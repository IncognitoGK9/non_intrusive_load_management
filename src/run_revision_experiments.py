"""
Revision experiments requested by the second review (October 2026). Resumable; run from nilm/src.

  --phase CNN   Calibrated same-protocol CNN. Same windows, houses, partition and network as phase D, but in every
                outer fold about 20% of the TRAINING houses are held out as validation houses. Training stops early
                on validation F1 (at most 12 epochs, patience 3) and the decision threshold is chosen on the
                validation houses only. Two input variants:
                  centred : the 99-sample window centred on the target sample (as in phase D)
                  causal  : the 50 samples up to and including the target sample (no future samples, comparable
                            with the trailing rolling window of the classical features)
                Output: results_all_experiments/phase_R_CNN/fold_results.csv

  --phase SENS  Data-quality sensitivity. LR, DT, XGBoost and LightGBM full factorial (160 configurations, fixed
                hyperparameters of plan_B.json) plus the pooled-selected fridge configuration (RF, FS-B, class
                weighting) on the reference house partition, with the two solar-affected houses (11 and 21) removed
                from every appliance and the House 4 kettle channel (0.004% ON) removed from the kettle data.
                Output: results_all_experiments/phase_R_SENS/fold_results.csv

USAGE:  bash run_revision.sh       (keeps the Mac awake, runs both phases in parallel, then the analysis)
"""
import argparse, os, sys, json, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from run_all_experiments import (Store, metrics, load, build, house_windows, find_raw, log, CFG, APPS, W, HALF)

EXCLUDE_ALL = {11, 21}                      # solar-affected aggregates (H11 fridge anomaly, H21 dishwasher anomaly, H21 combined kettle)
EXCLUDE_EXTRA = {"kettle": {4}}             # House 4 kettle channel ON for 0.004% of the time
MAX_EPOCHS, PATIENCE, VAL_SHARE = 12, 3, 0.2


def excluded(app):
    return EXCLUDE_ALL | EXCLUDE_EXTRA.get(app, set())


# ------------------------------------------------------------------ SENS
def run_sens():
    plan = json.load(open(os.path.join(CFG, "plan_B.json")))
    plan_a = json.load(open(os.path.join(CFG, "plan_A.json")))
    plan += [p for p in plan_a if (p["appliance"], p["arm"], p["model"], p["imbalance"]) == ("fridge", "B", "RandomForest", "class_weight")]
    order = {m: i for i, m in enumerate(["LogisticRegression", "DecisionTree", "XGBoost", "LightGBM", "RandomForest"])}
    imbs = ["none", "class_weight", "SMOTE", "ADASYN"]
    plan.sort(key=lambda p: (p["appliance"], p["arm"], order[p["model"]], imbs.index(p["imbalance"])))
    part = pd.read_csv(os.path.join(CFG, "house_partition.csv"))
    st = Store("R_SENS", ["seed", "appliance", "arm", "model", "imbalance", "fold"])
    total = len(plan) * 5
    for p in plan:
        app, arm, model, imb = p["appliance"], p["arm"], p["model"], p["imbalance"]
        if all(st.has(seed=42, appliance=app, arm=arm, model=model, imbalance=imb, fold=k) for k in range(1, 6)): continue
        X, y, g = load(app, arm); ex = excluded(app); keep = ~np.isin(g, list(ex))
        for _, r in part[part.appliance == app].sort_values("fold").iterrows():
            k = int(r.fold)
            if st.has(seed=42, appliance=app, arm=arm, model=model, imbalance=imb, fold=k): continue
            hs = [int(h) for h in str(r.test_houses).split() if int(h) not in ex]
            te = np.where(np.isin(g, hs) & keep)[0]; tr = np.where(~np.isin(g, hs) & keep)[0]
            t0 = time.time()
            est = build(model, imb, p["params"], y[tr]); est.fit(X[tr], y[tr]); pr = est.predict(X[te])
            st.add(dict(seed=42, appliance=app, arm=arm, model=model, imbalance=imb, fold=k, test_houses=" ".join(map(str, sorted(hs))),
                        excluded=" ".join(map(str, sorted(ex))), n_train=len(tr), **metrics(y[te], pr), elapsed_seconds=round(time.time() - t0, 1)))
            log(f"[SENS] {len(st.rows)}/{total} {app} {arm} {model} {imb} fold {k} F1={st.rows[-1]['f1']:.4f} {st.rows[-1]['elapsed_seconds']}s")


# ------------------------------------------------------------------ CNN
def best_threshold(y, p):
    from sklearn.metrics import precision_recall_curve
    if y.sum() == 0: return 0.5, 0.0
    pr, rc, th = precision_recall_curve(y, p)
    f = 2 * pr[:-1] * rc[:-1] / np.maximum(pr[:-1] + rc[:-1], 1e-12)
    i = int(np.nanargmax(f)); return float(th[i]), float(f[i])


def run_cnn(variants):
    import torch, torch.nn as nn
    from sklearn.metrics import f1_score
    raw = find_raw()
    cache = os.path.expanduser("~/nilm_dl_cache/windows"); os.makedirs(cache, exist_ok=True)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    part = pd.read_csv(os.path.join(CFG, "house_partition.csv"))
    st = Store("R_CNN", ["variant", "appliance", "fold"])

    class Net(nn.Module):  # identical to phase D
        def __init__(s):
            super().__init__()
            s.f = nn.Sequential(nn.Conv1d(2, 32, 7, padding=3), nn.ReLU(), nn.Conv1d(32, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                                nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2), nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
                                nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
        def forward(s, x): return s.f(x).squeeze(1)

    def prep(X, variant):
        if variant == "causal": X = X[:, :HALF + 1]          # samples t-49 ... t
        a = X / 1000.0; d = np.diff(X, axis=1, prepend=X[:, :1]) / 1000.0
        return np.stack([a, d], 1).astype(np.float32)

    def predict(net, X):
        net.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(X), 8192): out.append(torch.sigmoid(net(torch.from_numpy(X[i:i + 8192]).to(dev))).cpu().numpy())
        return np.concatenate(out)

    for variant in variants:
        for app in APPS:
            hs_all = sorted({int(h) for t in part[part.appliance == app].test_houses for h in str(t).split()})
            if all(st.has(variant=variant, appliance=app, fold=k) for k in range(1, 6)): continue
            missing = [h for h in hs_all if not os.path.exists(os.path.join(cache, f"{app}_h{h}.npz"))]
            if missing and raw is None:
                raise SystemExit(f"window cache missing for {app} houses {missing} and raw REFIT CSVs not found (~/refit_raw)")
            data = {h: house_windows(app, h, raw, cache) for h in hs_all}
            for _, r in part[part.appliance == app].sort_values("fold").iterrows():
                k = int(r.fold)
                if st.has(variant=variant, appliance=app, fold=k): continue
                t0 = time.time(); test = [int(h) for h in str(r.test_houses).split()]; train_all = [h for h in hs_all if h not in test]
                rng = np.random.default_rng(1000 + k); nval = max(1, int(round(VAL_SHARE * len(train_all))))
                for _ in range(50):  # validation houses must contain ON samples
                    val = sorted(rng.choice(train_all, nval, replace=False).tolist())
                    if sum(int(data[h][1].sum()) for h in val) > 0: break
                train = [h for h in train_all if h not in val]
                Xtr = prep(np.concatenate([data[h][0] for h in train]), variant); ytr = np.concatenate([data[h][1] for h in train]).astype(np.float32)
                Xva = prep(np.concatenate([data[h][0] for h in val]), variant); yva = np.concatenate([data[h][1] for h in val]).astype(int)
                Xte = prep(np.concatenate([data[h][0] for h in test]), variant); yte = np.concatenate([data[h][1] for h in test]).astype(int)
                torch.manual_seed(42); np.random.seed(42)
                net = Net().to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3)
                pw = torch.tensor([(len(ytr) - ytr.sum()) / max(ytr.sum(), 1)], device=dev)
                lossf = nn.BCEWithLogitsLoss(pos_weight=pw); n = len(ytr); bs = 1024
                Xt = torch.from_numpy(Xtr); yt = torch.from_numpy(ytr)
                best = (-1.0, None, 0.5, 0); bad = 0
                for ep in range(1, MAX_EPOCHS + 1):
                    net.train(); perm = torch.randperm(n)
                    for i in range(0, n, bs):
                        b = perm[i:i + bs]; opt.zero_grad()
                        l = lossf(net(Xt[b].to(dev)), yt[b].to(dev)); l.backward(); opt.step()
                    thr, vf = best_threshold(yva, predict(net, Xva))
                    if vf > best[0]:
                        best = (vf, {kk: v.detach().cpu().clone() for kk, v in net.state_dict().items()}, thr, ep); bad = 0
                    else:
                        bad += 1
                        if bad >= PATIENCE: break
                net.load_state_dict(best[1]); pte = predict(net, Xte); pr = (pte >= best[2]).astype(int)
                st.add(dict(variant=variant, appliance=app, model="CNN", fold=k, test_houses=" ".join(map(str, sorted(test))),
                            val_houses=" ".join(map(str, val)), n_train=n, best_epoch=best[3], threshold=best[2], val_f1=best[0],
                            f1_at_05=f1_score(yte, (pte >= 0.5).astype(int), zero_division=0), **metrics(yte, pr),
                            elapsed_seconds=round(time.time() - t0, 1)))
                log(f"[CNN] {len(st.rows)}/50 {variant} {app} fold {k} F1={st.rows[-1]['f1']:.4f} (F1@0.5 {st.rows[-1]['f1_at_05']:.4f}) thr={best[2]:.3f} ep={best[3]} {st.rows[-1]['elapsed_seconds']}s")
            del data


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", required=True, choices=["CNN", "SENS"]); a = ap.parse_args()
    if a.phase == "CNN": run_cnn(["centred", "causal"])
    if a.phase == "SENS": run_sens()
    log(f"PHASE R_{a.phase} DONE")
