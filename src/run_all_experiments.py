"""
All remaining leave-house-out experiments for the REFIT benchmark, in one resumable script.

Phases (each writes its own results folder and skips folds that are already finished):
  A  Full factorial of LightGBM, Random Forest and k-NN (both feature sets x four imbalance strategies,
     k-NN without class weighting), on the reference house partition used for the 35- and 120-configuration runs.
  B  Repeated house partitions: LR, DT, XGBoost and LightGBM full factorial on three further random
     house partitions (seeds 1, 2, 3).
  C  House-grouped hyperparameter tuning: LR, DT, XGBoost and LightGBM with engineered features and
     four imbalance strategies; hyperparameters re-tuned inside each training fold by 3-fold GroupKFold
     over the training houses only (the test houses are never used for tuning).
  D  Deep-learning baseline under the identical protocol: a 1D convolutional network on raw aggregate
     windows (99 samples, about 13 min), same houses, same partition, same F1 of the ON class.
  E  Full factorial of SVM-RBF (100,000-sample stratified training subsample per fold, as in the final run).

USAGE (from nilm/src):  python3 run_all_experiments.py --phase A   (or B, C, D, E)
Normally started by run_all_experiments.sh, which runs the phases in three parallel chains,
keeps the Mac awake with caffeinate and restarts any phase that stops.
"""
import argparse, os, sys, json, time, glob
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
FEAT = os.path.join(ROOT, "data", "features")
CFG = os.path.join(ROOT, "results_all_experiments", "config")
OUTROOT = os.path.join(ROOT, "results_all_experiments")
APPS = ["fridge", "washing_machine", "dishwasher", "kettle", "microwave"]
IMBS = ["none", "class_weight", "SMOTE", "ADASYN"]


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


class Store:
    """Fold-level results file with resume support."""
    def __init__(self, phase, keys):
        self.dir = os.path.join(OUTROOT, f"phase_{phase}"); os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, "fold_results.csv"); self.keys = keys
        self.rows = pd.read_csv(self.path).to_dict("records") if os.path.exists(self.path) else []
        self.done = {tuple(str(r[k]) for k in keys) for r in self.rows}

    def has(self, **kw):
        return tuple(str(kw[k]) for k in self.keys) in self.done

    def add(self, row):
        self.rows.append(row); self.done.add(tuple(str(row[k]) for k in self.keys))
        tmp = self.path + ".tmp"; pd.DataFrame(self.rows).to_csv(tmp, index=False); os.replace(tmp, self.path)


def metrics(yt, pr):
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    return dict(n_test=len(yt), n_test_on=int(yt.sum()), f1=f1_score(yt, pr, zero_division=0),
                precision=precision_score(yt, pr, zero_division=0), recall=recall_score(yt, pr, zero_division=0),
                accuracy=accuracy_score(yt, pr), tn=int(((yt == 0) & (pr == 0)).sum()), fp=int(((yt == 0) & (pr == 1)).sum()),
                fn=int(((yt == 1) & (pr == 0)).sum()), tp=int(((yt == 1) & (pr == 1)).sum()))


_cache = {}
def load(app, arm):
    k = (app, arm)
    if k not in _cache:
        _cache.clear()
        df = pd.read_csv(os.path.join(FEAT, f"{app}_arm{arm}.csv"))
        _cache[k] = (df.drop(columns=["label", "house_id"]).values, df["label"].values.astype(int), df["house_id"].values)
        del df
    return _cache[k]


def partitions(app, y, g, seed):
    """Yield (fold, test_houses, train_idx, test_idx). seed 42 = reference partition from house_partition.csv."""
    if seed == 42:
        part = pd.read_csv(os.path.join(CFG, "house_partition.csv"))
        for _, r in part[part.appliance == app].sort_values("fold").iterrows():
            hs = [int(h) for h in str(r.test_houses).split()]
            te = np.where(np.isin(g, hs))[0]; tr = np.where(~np.isin(g, hs))[0]
            yield int(r.fold), " ".join(map(str, sorted(hs))), tr, te
    else:
        from sklearn.model_selection import StratifiedGroupKFold
        for k, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, g), 1):
            yield k, " ".join(map(str, sorted(set(g[te])))), tr, te


def build(model, imb, params, ytr, n_jobs=3):
    from model_pipeline import _prepare_estimator, RANDOM_STATE
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.pipeline import Pipeline as ImbPipeline
    est = _prepare_estimator(model, imb, ytr)
    if params: est.set_params(**params)
    nj = {"RandomForest": "n_jobs", "XGBoost": "n_jobs", "LightGBM": "n_jobs", "kNN": "knn__n_jobs"}
    if model in nj: est.set_params(**{nj[model]: n_jobs})
    if imb in ("SMOTE", "ADASYN"):
        s = SMOTE(random_state=RANDOM_STATE) if imb == "SMOTE" else ADASYN(random_state=RANDOM_STATE)
        est = ImbPipeline([("sampler", s), ("model", est)])
    return est


def svm_subsample(tr, y):
    from sklearn.model_selection import train_test_split
    if len(tr) <= 100000: return tr
    t, _ = train_test_split(tr, train_size=100000, stratify=y[tr], random_state=42)
    return t


def run_fixed(phase, plan_file, seeds):
    plan = json.load(open(os.path.join(CFG, plan_file)))
    st = Store(phase, ["seed", "appliance", "arm", "model", "imbalance", "fold"])
    order = {m: i for i, m in enumerate(["LogisticRegression", "DecisionTree", "XGBoost", "LightGBM", "kNN", "RandomForest", "SVM_RBF"])}
    plan.sort(key=lambda p: (p["appliance"], p["arm"], order[p["model"]], IMBS.index(p["imbalance"])))
    total = len(plan) * 5 * len(seeds)
    for seed in seeds:
        for p in plan:
            app, arm, model, imb = p["appliance"], p["arm"], p["model"], p["imbalance"]
            if all(st.has(seed=seed, appliance=app, arm=arm, model=model, imbalance=imb, fold=k) for k in range(1, 6)): continue
            X, y, g = load(app, arm)
            for k, hs, tr, te in partitions(app, y, g, seed):
                if st.has(seed=seed, appliance=app, arm=arm, model=model, imbalance=imb, fold=k): continue
                t0 = time.time()
                trs = svm_subsample(tr, y) if model == "SVM_RBF" else tr
                est = build(model, imb, p["params"], y[trs]); est.fit(X[trs], y[trs]); pr = est.predict(X[te])
                st.add(dict(seed=seed, appliance=app, arm=arm, model=model, imbalance=imb, fold=k, test_houses=hs,
                            n_train=len(trs), **metrics(y[te], pr), elapsed_seconds=round(time.time() - t0, 1)))
                log(f"[{phase}] {len(st.rows)}/{total} seed {seed} {app} {arm} {model} {imb} fold {k} F1={st.rows[-1]['f1']:.4f} {st.rows[-1]['elapsed_seconds']}s")


def run_tuned():
    from sklearn.model_selection import GridSearchCV, GroupKFold
    from model_pipeline import MODELS
    st = Store("C", ["appliance", "model", "imbalance", "fold"])
    models = ["LogisticRegression", "DecisionTree", "XGBoost", "LightGBM"]
    total = len(APPS) * len(models) * len(IMBS) * 5
    for app in APPS:
        X, y, g = load(app, "B")
        for model in models:
            for imb in IMBS:
                for k, hs, tr, te in partitions(app, y, g, 42):
                    if st.has(appliance=app, model=model, imbalance=imb, fold=k): continue
                    t0 = time.time(); grid = MODELS[model][1]
                    est = build(model, imb, None, y[tr], n_jobs=1)
                    if imb in ("SMOTE", "ADASYN"): grid = {f"model__{a}": b for a, b in grid.items()}
                    gs = GridSearchCV(est, grid, scoring="f1", cv=GroupKFold(3), n_jobs=4, error_score=0.0)
                    gs.fit(X[tr], y[tr], groups=g[tr]); pr = gs.predict(X[te])
                    st.add(dict(appliance=app, arm="B", model=model, imbalance=imb, fold=k, test_houses=hs,
                                best_params=json.dumps({a.replace("model__", ""): b for a, b in gs.best_params_.items()}),
                                inner_f1=gs.best_score_, **metrics(y[te], pr), elapsed_seconds=round(time.time() - t0, 1)))
                    log(f"[C] {len(st.rows)}/{total} {app} {model} {imb} fold {k} F1={st.rows[-1]['f1']:.4f} {st.rows[-1]['elapsed_seconds']}s")


# ---------------- Phase D: deep-learning baseline
W = 99; HALF = W // 2; PER_HOUSE = 60000

def find_raw():
    cands = [os.path.join(ROOT, "..", "refit_raw"), os.path.expanduser("~/refit_raw"), os.path.join(ROOT, "data", "refit_raw"),
             os.path.expanduser("~/nilm_dl_cache/refit_raw")]
    for c in cands:
        if glob.glob(os.path.join(c, "CLEAN_House*.csv")): return os.path.abspath(c)
    arc = os.path.join(ROOT, "data", "CLEAN_REFIT_081116.7z"); dst = os.path.expanduser("~/nilm_dl_cache/refit_raw")
    if os.path.exists(arc):
        log("[D] extracting raw REFIT archive (one-off)"); os.makedirs(dst, exist_ok=True)
        try:
            import py7zr
        except ImportError:
            os.system(f"{sys.executable} -m pip install --quiet py7zr"); import py7zr
        with py7zr.SevenZipFile(arc, "r") as z: z.extractall(dst)
        for c in [dst] + glob.glob(os.path.join(dst, "*")):
            if glob.glob(os.path.join(c, "CLEAN_House*.csv")): return c
    return None


def house_windows(app, h, raw, cache):
    f = os.path.join(cache, f"{app}_h{h}.npz")
    if os.path.exists(f):
        z = np.load(f); return z["X"], z["y"]
    from data_refit import load_house, binarize_appliance
    df = load_house(h, app, raw_dir=raw)
    agg = df["Aggregate"].to_numpy(np.float32); lab = binarize_appliance(df[app], app).to_numpy(np.int8)
    idx = np.arange(HALF, len(agg) - HALF); yl = lab[idx]
    rng = np.random.default_rng(42 + h)
    pos, neg = idx[yl == 1], idx[yl == 0]
    n_pos = min(len(pos), max(1, round(PER_HOUSE * len(pos) / len(idx)))) if len(pos) else 0
    n_neg = min(len(neg), PER_HOUSE - n_pos)
    sel = np.sort(np.concatenate([rng.choice(pos, n_pos, replace=False) if n_pos else pos[:0], rng.choice(neg, n_neg, replace=False)]))
    Xw = np.lib.stride_tricks.sliding_window_view(agg, W)[sel - HALF].astype(np.float32)
    yw = lab[sel].astype(np.int8)
    np.savez(f, X=Xw, y=yw); return Xw, yw


def run_dl():
    try:
        import torch
    except ImportError:
        os.system(f"{sys.executable} -m pip install --quiet torch"); import torch
    import torch.nn as nn
    raw = find_raw()
    if raw is None:
        log("[D] raw REFIT CSV files not found; phase D skipped"); return
    log(f"[D] raw data: {raw}")
    cache = os.path.expanduser("~/nilm_dl_cache/windows"); os.makedirs(cache, exist_ok=True)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    st = Store("D", ["appliance", "fold"])
    part = pd.read_csv(os.path.join(CFG, "house_partition.csv"))

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.f = nn.Sequential(nn.Conv1d(2, 32, 7, padding=3), nn.ReLU(), nn.Conv1d(32, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                                nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2), nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
                                nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
        def forward(s, x): return s.f(x).squeeze(1)

    def prep(X):
        a = X / 1000.0; d = np.diff(X, axis=1, prepend=X[:, :1]) / 1000.0
        return np.stack([a, d], 1).astype(np.float32)

    for app in APPS:
        hs_all = sorted({int(h) for t in part[part.appliance == app].test_houses for h in str(t).split()})
        data = {h: house_windows(app, h, raw, cache) for h in hs_all}
        for _, r in part[part.appliance == app].sort_values("fold").iterrows():
            k = int(r.fold)
            if st.has(appliance=app, fold=k): continue
            t0 = time.time(); test = [int(h) for h in str(r.test_houses).split()]; train = [h for h in hs_all if h not in test]
            Xtr = prep(np.concatenate([data[h][0] for h in train])); ytr = np.concatenate([data[h][1] for h in train]).astype(np.float32)
            Xte = prep(np.concatenate([data[h][0] for h in test])); yte = np.concatenate([data[h][1] for h in test]).astype(int)
            torch.manual_seed(42); np.random.seed(42)
            net = Net().to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3)
            pw = torch.tensor([(len(ytr) - ytr.sum()) / max(ytr.sum(), 1)], device=dev)
            lossf = nn.BCEWithLogitsLoss(pos_weight=pw); n = len(ytr); bs = 1024
            Xt = torch.from_numpy(Xtr); yt = torch.from_numpy(ytr)
            for ep in range(6):
                net.train(); perm = torch.randperm(n)
                for i in range(0, n, bs):
                    b = perm[i:i + bs]; opt.zero_grad()
                    l = lossf(net(Xt[b].to(dev)), yt[b].to(dev)); l.backward(); opt.step()
            net.eval(); pr = []
            with torch.no_grad():
                for i in range(0, len(Xte), 8192): pr.append((net(torch.from_numpy(Xte[i:i + 8192]).to(dev)) > 0).cpu().numpy())
            pr = np.concatenate(pr).astype(int)
            st.add(dict(appliance=app, model="CNN", fold=k, test_houses=" ".join(map(str, sorted(test))), n_train=n,
                        **metrics(yte, pr), elapsed_seconds=round(time.time() - t0, 1)))
            log(f"[D] {len(st.rows)}/25 {app} fold {k} F1={st.rows[-1]['f1']:.4f} {st.rows[-1]['elapsed_seconds']}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", required=True, choices=list("ABCDE")); a = ap.parse_args()
    if a.phase == "A": run_fixed("A", "plan_A.json", [42])
    if a.phase == "B": run_fixed("B", "plan_B.json", [1, 2, 3])
    if a.phase == "C": run_tuned()
    if a.phase == "D": run_dl()
    if a.phase == "E": run_fixed("E", "plan_E.json", [42])
    log(f"PHASE {a.phase} DONE")
