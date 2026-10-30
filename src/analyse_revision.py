"""Analyses added in the revision (October 2026). Run after analyse_unseen_houses.py (and, when available, after
run_revision_experiments.py). Writes results_final/revision_summary.json and prints a readable summary.

  1. Matched imbalance effects (k-NN excluded, because k-NN has no class weighting)
  2. Two-way ANOVA without k-NN (full-rank design), rank-based ANOVA, residual diagnostics
  3. Iman-Davenport correction of the Friedman test
  4. Optimism against the mean of pooled and unseen scores (regression-to-the-mean check)
  5. Nested house-grouped selection from phase C (configuration chosen on the training houses only)
  6. Calibrated CNN (phase R_CNN) and data-quality sensitivity (phase R_SENS), if present
"""
import os, json, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf, statsmodels.api as sm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); OUT = os.path.join(ROOT, "results_final")
U = os.path.join(ROOT, "results_all_experiments")
K = ["appliance", "arm", "model", "imbalance"]
ON = {"fridge": 339857, "washing_machine": 35826, "dishwasher": 42545, "kettle": 5979, "microwave": 3632}
def wavg(d):
    d = dict(d); return sum(d[a] * ON[a] for a in d) / sum(ON[a] for a in d)
R = {}
P = pd.read_csv(os.path.join(OUT, "pooled_final_280.csv")); P = P[P.status == "ok"]
g = pd.read_csv(os.path.join(OUT, "lho270_summary.csv"))

# 1. matched imbalance effects
for col in ["pooled", "lho"]:
    i = g[g.model != "kNN"].groupby("imbalance")[col].mean()
    R[f"imb_matched_{col}"] = dict(i.round(4)); R[f"cw_gain_matched_{col}"] = 100 * (i["class_weight"] / i["none"] - 1)
    R[f"drop_imb_matched"] = g[g.model != "kNN"].groupby("imbalance")["drop"].mean().round(4).to_dict()
bc = P[P.model != "kNN"].groupby(["model", "imbalance"]).f1_mean.max().unstack().mean()
R["bestcase_imb_matched"] = bc.round(3).to_dict()

# 2. ANOVA
def shares(df, y, f, typ=2):
    m = smf.ols(f"{y} ~ {f}", data=df).fit(); a = sm.stats.anova_lm(m, typ=typ)
    return (100 * a.sum_sq / a.sum_sq.sum()).round(2).to_dict(), a["PR(>F)"].dropna().to_dict(), m.rsquared, m
main = "C(appliance)+C(model)+C(arm)+C(imbalance)"; two = f"({main})**2"
Q = P[P.model != "kNN"]
s2, p2, r2, _ = shares(Q, "f1_mean", two); R["anova_twoway_no_knn"] = dict(shares=s2, p=p2, r2=r2, n=len(Q))
sm_, _, _, mm = shares(Q, "f1_mean", main); R["anova_main_no_knn"] = sm_
P2 = P.copy(); P2["r"] = P2.f1_mean.rank(); R["anova_rank_main"] = shares(P2, "r", main)[0]
_, _, _, m1 = shares(P, "f1_mean", main)
R["shapiro_p"] = stats.shapiro(m1.resid).pvalue; R["levene_app_p"] = stats.levene(*[v.values for _, v in m1.resid.groupby(P.appliance)]).pvalue

# 3. Iman-Davenport
B = P.groupby(["model", "appliance"]).f1_mean.max().unstack(); k_, N_ = B.shape
chi = stats.friedmanchisquare(*[B.loc[m] for m in B.index]).statistic; FF = (N_ - 1) * chi / (N_ * (k_ - 1) - chi)
R["friedman_pooled"] = dict(chi2=chi, iman_davenport_F=FF, p=stats.f.sf(FF, k_ - 1, (k_ - 1) * (N_ - 1)))

# 4. optimism vs mean
R["rho_drop_vs_pooled"] = stats.spearmanr(g.pooled, g["drop"])[0]
r_, p_ = stats.spearmanr((g.pooled + g.lho) / 2, g["drop"]); R["rho_drop_vs_mean"] = [r_, p_]

# 5. nested house-grouped selection (phase C)
C = pd.read_csv(os.path.join(U, "phase_C", "fold_results.csv")); PB = P[P.arm == "B"]; rows = []
for (a, f), d in C.groupby(["appliance", "fold"]):
    gs = d.loc[d.inner_f1.idxmax()]; pp = PB[(PB.appliance == a) & PB.model.isin(d.model.unique())]; bp = pp.loc[pp.f1_mean.idxmax()]
    ps = d[(d.model == bp.model) & (d.imbalance == bp.imbalance)].iloc[0]; orc = d.loc[d.f1.idxmax()]
    rows.append(dict(appliance=a, fold=f, grouped_sel=f"{gs.model}/{gs.imbalance}", grouped_f1=gs.f1, pooled_sel=f"{bp.model}/{bp.imbalance}", pooled_f1=ps.f1, oracle_f1=orc.f1))
ns = pd.DataFrame(rows); ns.to_csv(os.path.join(OUT, "nested_selection_phaseC.csv"), index=False)
s = ns.groupby("appliance")[["grouped_f1", "pooled_f1", "oracle_f1"]].mean()
R["nested"] = dict(per_app=s.round(4).to_dict("index"), gain=(s.grouped_f1 - s.pooled_f1).round(4).to_dict(),
                   w_grouped=wavg(s.grouped_f1), w_pooled=wavg(s.pooled_f1), w_oracle=wavg(s.oracle_f1),
                   folds_better=int((ns.grouped_f1 > ns.pooled_f1).sum()), folds_worse=int((ns.grouped_f1 < ns.pooled_f1).sum()),
                   choices=ns.groupby("appliance").grouped_sel.agg(lambda x: x.value_counts().to_dict()).to_dict())

# 6a. calibrated CNN
f = os.path.join(U, "phase_R_CNN", "fold_results.csv")
if os.path.exists(f):
    D = pd.read_csv(f); out = {}
    for v, d in D.groupby("variant"):
        t = d.groupby("appliance").agg(f1=("f1", "mean"), sd=("f1", "std"), precision=("precision", "mean"), recall=("recall", "mean"),
                                       f1_at_05=("f1_at_05", "mean"), folds=("f1", "size"))
        out[v] = dict(per_app=t.round(4).to_dict("index"), weighted=wavg(t.f1) if len(t) == 5 else None, macro=t.f1.mean(),
                      weighted_at_05=wavg(t.f1_at_05) if len(t) == 5 else None, complete=bool(len(d) == 25))
    R["cnn_calibrated"] = out

# 6b. data-quality sensitivity
f = os.path.join(U, "phase_R_SENS", "fold_results.csv")
if os.path.exists(f):
    S = pd.read_csv(f); ref = pd.concat([pd.read_csv(os.path.join(ROOT, "results_lho_factorial", "fold_results.csv")),
                                         pd.read_csv(os.path.join(U, "phase_A", "fold_results.csv"))])
    ss = S.groupby(K).f1.mean().rename("sens").reset_index(); rr = ref.groupby(K).f1.mean().rename("ref").reset_index()
    m = ss.merge(rr, on=K).merge(P[K + ["f1_mean"]].rename(columns={"f1_mean": "pooled"}), on=K)
    fast = m[m.model.isin(["LogisticRegression", "DecisionTree", "XGBoost", "LightGBM"])]
    def eff(d, c):
        a = d.groupby("arm")[c].mean(); i = d.groupby("imbalance")[c].mean()
        return dict(fs_gain=100 * (a.B / a.A - 1), cw_gain=100 * (i["class_weight"] / i["none"] - 1))
    sel = {}
    for a in ON:
        pa = P[P.appliance == a]; bp = pa.loc[pa.f1_mean.idxmax()]; row = m[(m[K] == bp[K]).all(axis=1)]
        if len(row): sel[a] = dict(config=f"{bp.model}/{bp.arm}/{bp.imbalance}", ref=float(row.ref.iloc[0]), sens=float(row.sens.iloc[0]))
        x = fast[fast.appliance == a]
        if len(x): sel.setdefault(a, {}).update(best_fast_ref=f"{x.loc[x.ref.idxmax(), 'model']}/{x.loc[x.ref.idxmax(), 'arm']}/{x.loc[x.ref.idxmax(), 'imbalance']}",
                                                best_fast_sens=f"{x.loc[x.sens.idxmax(), 'model']}/{x.loc[x.sens.idxmax(), 'arm']}/{x.loc[x.sens.idxmax(), 'imbalance']}")
    R["sensitivity"] = dict(n_configs=len(m), complete=bool(len(S) == 161 * 5), eff_ref=eff(fast, "ref"), eff_sens=eff(fast, "sens"),
                            mean_change=float((m.sens - m.ref).mean()), rho_ref_sens=stats.spearmanr(m.ref, m.sens)[0],
                            rho_pooled_sens=stats.spearmanr(m.pooled, m.sens)[0], selections=sel,
                            w_pooledsel_ref=wavg({a: v["ref"] for a, v in sel.items() if "ref" in v}) if all("ref" in v for v in sel.values()) and len(sel) == 5 else None,
                            w_pooledsel_sens=wavg({a: v["sens"] for a, v in sel.items() if "sens" in v}) if all("sens" in v for v in sel.values()) and len(sel) == 5 else None,
                            per_app_change=(m.sens - m.ref).groupby(m.appliance).mean().round(4).to_dict())

json.dump(R, open(os.path.join(OUT, "revision_summary.json"), "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=lambda x: round(float(x), 4)))
