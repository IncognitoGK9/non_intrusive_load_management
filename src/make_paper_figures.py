"""Draws every figure of the paper (Figs 1 to 9) and the graphical abstract from the result files.
Run after consolidate_results.py and analyse_unseen_houses.py. Output: figures/"""
import os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")); P = lambda *a: os.path.join(ROOT, *a)
OUT = P("figures"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "axes.edgecolor": "#333333", "axes.linewidth": 0.7})
BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#9aa5b1"
MODELS = ["LogisticRegression", "DecisionTree", "kNN", "SVM_RBF", "RandomForest", "XGBoost", "LightGBM"]
AB = {"LogisticRegression": "LR", "DecisionTree": "DT", "kNN": "k-NN", "SVM_RBF": "SVM-RBF", "RandomForest": "RF", "XGBoost": "XGBoost", "LightGBM": "LightGBM"}
APPS = ["fridge", "washing_machine", "dishwasher", "kettle", "microwave"]
AN = {"fridge": "Fridge", "washing_machine": "Washing machine", "dishwasher": "Dishwasher", "kettle": "Kettle", "microwave": "Microwave"}
IMB = ["none", "class_weight", "SMOTE", "ADASYN"]; IN = {"none": "No correction", "class_weight": "Class weighting", "SMOTE": "SMOTE", "ADASYN": "ADASYN"}
fin = pd.read_csv(P("results_final", "pooled_final_280.csv")); fin = fin[fin.status == "ok"]
fp = pd.read_csv(P("results_final", "pooled_first_pass_280.csv")); fp = fp[fp.status == "ok"]
R = json.load(open(P("results_final", "analysis_unseen_houses.json"))); A = json.load(open(P("results_final", "anova_pooled_270.json")))

def clean(ax, grid="y"):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    if grid: ax.grid(axis=grid, color="#e3e3e3", lw=0.6); ax.set_axisbelow(True)
def save(fig, name): fig.savefig(os.path.join(OUT, name), dpi=300, bbox_inches="tight", pad_inches=0.03); plt.close(fig); print("wrote", name)

# Fig 1: framework (study design, including validation on unseen houses and robustness checks)
fig, ax = plt.subplots(figsize=(12, 3.6)); ax.set_xlim(0, 12); ax.set_ylim(0, 3.6); ax.axis("off")
cb, co, cg = ("#eaf2fb", "#3b78c4"), ("#fdf0e8", "#e07b39"), ("#ebf6ee", "#3a9a5b")
def box(x, y, w, h, title, body, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=c[0], ec=c[1], lw=1.4))
    ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=9.5, fontweight="bold")
    ax.text(x + w / 2, y + h - 0.52, body, ha="center", va="top", fontsize=8, linespacing=1.35)
def arr(x1, y1, x2, y2): ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.1))
w, h, y1 = 2.65, 1.2, 2.3; xs = [0.15 + i * 3.0 for i in range(4)]
top = [("REFIT dataset", "20 UK households\naggregate and appliance\nchannels, about 8 s"),
       ("Labelling", "verified channel map\n5 target appliances\nfixed ON-power thresholds"),
       ("Feature sets", "FS-A: power, hour of day\nFS-B: 15-sample rolling\nstatistics, difference, day"),
       ("Pooling", "60,000 stratified samples\nper house; 0.90 to 1.14\nmillion per appliance")]
for i, (t, b) in enumerate(top):
    box(xs[i], y1, w, h, t, b, cb)
    if i: arr(xs[i - 1] + w, y1 + h / 2, xs[i], y1 + h / 2)
w2, y2 = 2.2, 0.15; xs2 = [0.15 + i * 2.4 for i in range(5)]
bot = [("Factorial grid", "7 models x 2 feature sets\nx 4 imbalance strategies\nx 5 appliances\n270 fitted configurations", co),
       ("Pooled nested CV", "5 outer stratified folds\n3-fold inner grid search\nfirst pass and final run\n(training-set size)", co),
       ("Unseen houses", "leave-house-out validation\nof all 270 configurations\n5 folds, 3 or 4 test\nhouses per fold", cg),
       ("Robustness checks", "4 house partitions\ngrouped tuning, selection\nsame-protocol CNN\ndata-quality check", cg),
       ("Analysis", "Friedman and Nemenyi\nanalysis of variance\nweighted F1; comparison\nwith published results", co)]
for i, (t, b, c) in enumerate(bot):
    box(xs2[i], y2, w2, 1.65, t, b, c)
    if i: arr(xs2[i - 1] + w2, y2 + 0.82, xs2[i], y2 + 0.82)
ax.plot([xs[3] + w / 2] * 2, [y1, 2.05], color="#333", lw=1.1); ax.plot([xs[3] + w / 2, xs2[0] + w2 / 2], [2.05, 2.05], color="#333", lw=1.1)
arr(xs2[0] + w2 / 2, 2.05, xs2[0] + w2 / 2, y2 + 1.65)
fig.savefig(os.path.join(OUT, "fig01_framework.pdf"), bbox_inches="tight")
save(fig, "fig01_framework.png")

best = fin.groupby(["appliance", "model"]).f1_mean.max().unstack()[MODELS]
# Fig 2: model comparison
fig, ax = plt.subplots(figsize=(6.0, 2.6)); x = np.arange(len(MODELS))
ax.bar(x, best.mean().values, 0.62, color=BLUE, yerr=best.std().values, capsize=3, error_kw=dict(lw=0.8, ecolor="#333333"))
ax.set_xticks(x); ax.set_xticklabels([AB[m] for m in MODELS]); ax.set_ylabel("Best F1 (mean over appliances)"); ax.set_ylim(0, 0.75); clean(ax)
save(fig, "fig02_model_comparison.png")
# Fig 3: critical difference
ranks = best.rank(axis=1, ascending=False).mean().sort_values(); k, N = 7, 5; CD = 2.949 * np.sqrt(k * (k + 1) / (6 * N))
fig, ax = plt.subplots(figsize=(6.0, 1.9)); ax.set_xlim(0.8, 7.2); ax.set_ylim(-1.4, 1.1); ax.axis("off")
ax.plot([1, 7], [0, 0], color="#222222", lw=0.9)
for r in range(1, 8): ax.plot([r, r], [0, 0.1], color="#222222", lw=0.9); ax.text(r, 0.2, str(r), ha="center", fontsize=7.5)
ax.plot([1, 1 + CD], [0.75, 0.75], color="#222222", lw=1.2); ax.text(1 + CD / 2, 0.85, f"CD = {CD:.2f}", ha="center", fontsize=7.5)
for i, (m, r) in enumerate(ranks.items()):
    left = i < 4; y = -0.45 - (i if left else i - 4) * 0.26
    ax.plot([r, r], [0, y], color="#444444", lw=0.8); xe = 0.95 if left else 7.05
    ax.plot([r, xe], [y, y], color="#444444", lw=0.8)
    ax.text(xe - 0.03 if left else xe + 0.03, y, f"{AB[m]} ({r:.1f})", ha="right" if left else "left", va="center", fontsize=7.5)
vals = ranks.values; names = list(ranks.index); yb = -1.25
for i in range(len(vals)):
    j = max(jj for jj in range(len(vals)) if vals[jj] - vals[i] <= CD)
    if j > i and not any(vals[i2] - vals[i] <= CD and j2 >= j and i2 < i for i2, j2 in []):
        pass
grp = []
for i in range(len(vals)):
    j = max(jj for jj in range(len(vals)) if vals[jj] - vals[i] <= CD)
    if j > i and not any(a <= i and b >= j for a, b in grp): grp.append((i, j))
for n, (i, j) in enumerate(grp):
    yy = -0.12 - n * 0.1; ax.plot([vals[i] - 0.05, vals[j] + 0.05], [yy, yy], color=ORANGE, lw=3)
save(fig, "fig03_critical_difference.png")
# Fig 4: feature sets
fa = fin.groupby(["model", "arm"]).f1_mean.max().unstack().loc[MODELS]
fig, ax = plt.subplots(figsize=(6.0, 2.6)); w = 0.38
ax.bar(x - w / 2, fa["A"], w, color=BLUE, label="FS-A (basic)"); ax.bar(x + w / 2, fa["B"], w, color=ORANGE, label="FS-B (engineered)")
ax.set_xticks(x); ax.set_xticklabels([AB[m] for m in MODELS]); ax.set_ylabel("Best F1"); ax.set_ylim(0, 0.7); ax.legend(frameon=False, ncol=2, loc="upper left"); clean(ax)
save(fig, "fig04_feature_sets.png")
# Fig 5: imbalance
ib = fin.groupby(["model", "imbalance"]).f1_mean.max().unstack().reindex(index=MODELS, columns=IMB)
fig, ax = plt.subplots(figsize=(6.0, 2.6)); w = 0.2
for j, (i_, c) in enumerate(zip(IMB, [GRAY, BLUE, AQUA, YELLOW])): ax.bar(x + (j - 1.5) * w, np.nan_to_num(ib[i_].values), w, color=c, label=IN[i_])
ax.set_xticks(x); ax.set_xticklabels([AB[m] for m in MODELS]); ax.set_ylabel("Best F1"); ax.set_ylim(0, 0.75); ax.legend(frameon=False, ncol=4, loc="upper left", fontsize=7); clean(ax)
save(fig, "fig05_imbalance.png")
# Fig 6: heatmap
H = best.loc[APPS]; fig, ax = plt.subplots(figsize=(6.0, 2.6)); im = ax.imshow(H.values, cmap="Blues", vmin=0, vmax=0.6, aspect="auto")
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        v = H.values[i, j]; ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7, color="white" if v > 0.45 else "#111111",
                                   weight="bold" if v == H.values[i].max() else "normal")
ax.set_xticks(range(7)); ax.set_xticklabels([AB[m] for m in MODELS]); ax.set_yticks(range(5)); ax.set_yticklabels([AN[a] for a in APPS])
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.set_label("F1", fontsize=8)
save(fig, "fig06_heatmap.png")
# Fig 7: change between first pass and final run
K = ["appliance", "arm", "model", "imbalance"]; RE = ["SVM_RBF", "RandomForest", "XGBoost", "LightGBM"]
m = fin[fin.model.isin(RE)].merge(fp[fp.model.isin(RE)], on=K, suffixes=("_f", "_p")); m["d"] = m.f1_mean_f - m.f1_mean_p
g = m.groupby("model").agg(d=("d", "mean"), up=("d", lambda s: int((s > 0).sum())), n=("d", "size")).loc[RE]
fig, ax = plt.subplots(figsize=(6.0, 2.4)); xx = np.arange(4)
ax.bar(xx, g.d, 0.55, color=[BLUE if v >= 0 else ORANGE for v in g.d]); ax.axhline(0, color="#333333", lw=0.7)
for i, (v, u, n) in enumerate(zip(g.d, g.up, g.n)): ax.text(i, v + (0.002 if v >= 0 else -0.002), f"{v:+.3f}\n{u} of {n} improved", ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
ax.set_xticks(xx); ax.set_xticklabels([AB[r] for r in RE]); ax.set_ylabel("Mean change in F1"); ax.set_ylim(-0.075, 0.03); clean(ax)
save(fig, "fig07_first_pass_vs_final.png"); print(g.round(3).to_dict())
# Fig 8: variance decomposition
NM = {"C(appliance)": "Appliance", "C(model)": "Model", "C(arm)": "Feature set", "C(imbalance)": "Imbalance strategy", "Residual": "Residual"}
nm = lambda k: " × ".join(NM[p] for p in k.split(":")) if k != "Residual" else "Residual"
_rv = P("results_final", "revision_summary.json")
if os.path.exists(_rv):  # revision: two-way model fitted without k-NN (full-rank design)
    _t = json.load(open(_rv))["anova_twoway_no_knn"]; e = pd.Series(_t["shares"]).sort_values(); pv = _t["p"]
else:
    e = pd.Series(A["twoway"]).sort_values(); pv = A["p"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.0, 2.7), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.55))
a1.barh(range(len(e)), e.values, color=[GRAY if k == "Residual" else (BLUE if ":" not in k else "#7fb0ea") for k in e.index], height=0.65)
for i, (k_, v) in enumerate(e.items()): a1.text(v + 0.6, i, f"{v:.1f}" + ("" if k_ == "Residual" or (pv.get(k_) or 0) < 0.05 else " (n.s.)"), va="center", fontsize=7)
a1.set_yticks(range(len(e))); a1.set_yticklabels([nm(k_) for k_ in e.index], fontsize=7.2); a1.set_xlabel("Share of variance in F1 (%)"); a1.set_xlim(0, 45); clean(a1, "x")
a1.text(-0.02, 1.03, "(a)", transform=a1.transAxes, fontsize=9, weight="bold", ha="right")
keys = ["C(appliance)", "C(arm)", "C(model)", "C(imbalance)", "Residual"]; y = np.arange(5); ep, el = R["eta"]["pooled"], R["eta"]["lho"]
a2.barh(y - 0.18, [ep[k_] for k_ in keys], 0.36, color=BLUE, label="Pooled CV"); a2.barh(y + 0.18, [el[k_] for k_ in keys], 0.36, color=ORANGE, label="Unseen houses")
for i, k_ in enumerate(keys): a2.text(ep[k_] + 0.6, i - 0.18, f"{ep[k_]:.1f}", va="center", fontsize=7); a2.text(el[k_] + 0.6, i + 0.18, f"{el[k_]:.1f}", va="center", fontsize=7)
a2.set_yticks(y); a2.set_yticklabels([NM[k_] for k_ in keys], fontsize=7.2); a2.invert_yaxis(); a2.set_xlabel("Share of variance in F1 (%)"); a2.set_xlim(0, 48); clean(a2, "x")
a2.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7); a2.text(-0.02, 1.03, "(b)", transform=a2.transAxes, fontsize=9, weight="bold", ha="right")
save(fig, "fig08_variance.png")
# Fig 9: pooled vs unseen
g9 = pd.read_csv(P("results_final", "lho270_summary.csv")); fig, ax = plt.subplots(figsize=(4.2, 3.6)); C = {"none": ORANGE, "class_weight": BLUE, "SMOTE": AQUA, "ADASYN": YELLOW}
for i_, c in C.items():
    for arm, mk in [("A", "o"), ("B", "s")]:
        d = g9[(g9.imbalance == i_) & (g9.arm == arm)]
        ax.scatter(d.pooled, d.lho, s=13, marker=mk, facecolor=c if arm == "B" else "white", edgecolor=c, lw=0.8, label=f"{IN[i_]}, FS-{arm}")
ax.plot([0, 0.62], [0, 0.62], color="#444444", lw=0.8, ls="--"); ax.text(0.53, 0.585, "y = x", fontsize=7, color="#444444")
ax.set_xlim(0, 0.62); ax.set_ylim(0, 0.62); ax.set_xlabel("F1, pooled cross-validation"); ax.set_ylabel("F1, unseen houses"); clean(ax, "both")
ax.legend(frameon=False, fontsize=6.3, loc="upper left", handletextpad=0.3); save(fig, "fig09_pooled_vs_unseen.png")
# Graphical abstract
B_, O_, G_, INK, MUTE = "#2b6cb0", "#e8743b", "#2f855a", "#1a202c", "#718096"
fig = plt.figure(figsize=(13.28, 5.31), dpi=200); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 13.28); ax.set_ylim(0, 5.31); ax.axis("off")
def box(x0, w0, t):
    ax.add_patch(FancyBboxPatch((x0, 0.35), w0, 4.15, boxstyle="round,pad=0.02,rounding_size=0.12", fc="#f7fafc", ec="#cbd5e0", lw=1.2))
    ax.text(x0 + w0 / 2, 4.15, t, ha="center", va="center", fontsize=12.5, fontweight="bold", color=INK)
ax.text(6.64, 4.98, "What matters in classical non-intrusive load monitoring?", ha="center", va="center", fontsize=14.5, fontweight="bold", color=INK)
box(0.2, 2.9, "1. Data")
for i, t in enumerate(["REFIT: 20 UK households", "Aggregate meter, about 8 s", "Five appliances:", "fridge, washing machine,", "dishwasher, kettle, microwave"]):
    ax.text(0.45, 3.5 - i * 0.52, t, fontsize=10.5, color=INK if i != 2 else MUTE, va="center")
ax.text(0.45, 0.85, "ON/OFF labels from appliance channels", fontsize=9, color=MUTE, va="center")
box(3.35, 3.0, "2. Factorial benchmark")
for i, (n_, t) in enumerate([("7", "classical models"), ("2", "feature sets"), ("4", "imbalance strategies")]):
    ax.text(3.65, 3.5 - i * 0.62, n_, fontsize=21, fontweight="bold", color=B_, va="center"); ax.text(4.2, 3.5 - i * 0.62, t, fontsize=11, color=INK, va="center")
ax.text(3.65, 1.55, "= 270 configurations", fontsize=13, fontweight="bold", color=O_, va="center"); ax.text(3.65, 0.95, "All re-tested on unseen houses", fontsize=9, color=MUTE, va="center")
box(6.6, 3.1, "3. What drives F1?"); am = A["main"]
for i, (n_, k_) in enumerate([("Appliance", "C(appliance)"), ("Feature set", "C(arm)"), ("Model", "C(model)"), ("Imbalance", "C(imbalance)")]):
    v = am[k_]; yy = 3.45 - i * 0.6; ax.text(6.8, yy, n_, fontsize=10.5, va="center", color=INK)
    ax.add_patch(plt.Rectangle((8.05, yy - 0.15), 1.4 * v / 40, 0.30, fc=B_, ec="none")); ax.text(8.1 + 1.4 * v / 40, yy, f"{v:.0f}%", fontsize=10, va="center", color=INK)
ax.text(6.8, 0.85, "Share of variance, 270 configurations", fontsize=9, color=MUTE, va="center")
box(9.95, 3.1, "4. Pooled CV vs new houses")
_RV = json.load(open(_rv)) if os.path.exists(_rv) else {}  # revision: k-NN excluded (no class weighting)
_cwp = _RV.get("cw_gain_matched_pooled", R['eff_pooled']['cw_gain']); _cwl = _RV.get("cw_gain_matched_lho", R['eff_lho']['cw_gain'])
rows = [("Weighted F1", "0.55", f"{R['w_pooledsel']:.2f}"), ("Model gaps", "sig.", "n.s."), ("Feature gain", f"+{R['eff_pooled']['gain']:.0f}%", f"+{R['eff_lho']['gain']:.0f}%"),
        ("Weighting gain", f"+{_cwp:.0f}%", f"+{_cwl:.0f}%"), ("CNN, same protocol", "", f"{_RV['cnn_calibrated']['centred']['weighted']:.2f}" if _RV.get('cnn_calibrated', {}).get('centred', {}).get('weighted') else f"{R['D_w']:.2f}")]
ax.text(11.75, 3.62, "Pooled", fontsize=9, color=B_, ha="center", fontweight="bold"); ax.text(12.6, 3.62, "Unseen", fontsize=9, color=O_, ha="center", fontweight="bold")
for i, (n_, a_, b_) in enumerate(rows):
    yy = 3.18 - i * 0.46; ax.text(10.12, yy, n_, fontsize=9.4, va="center", color=INK)
    ax.text(11.75, yy, a_, fontsize=10, va="center", ha="center", color=B_); ax.text(12.6, yy, b_, fontsize=10, va="center", ha="center", color=O_, fontweight="bold")
ax.text(10.12, 0.75, ("House-grouped selection better: " + str(sum(v > 0.01 for v in _RV["nested"]["gain"].values())) + " of 5" if "nested" in _RV else f"Pooled choice not best: {R['n_changed']} of 5 appliances"), fontsize=8.8, color=G_, va="center", fontweight="bold")
for x0 in (3.18, 6.43, 9.78): ax.add_patch(FancyArrowPatch((x0 - 0.05, 2.4), (x0 + 0.15, 2.4), arrowstyle="-|>", mutation_scale=14, color=MUTE, lw=1.4))
fig.savefig(os.path.join(OUT, "graphical_abstract.png"), dpi=200, facecolor="white"); plt.close(fig); print("wrote graphical_abstract.png")
print("ranks", ranks.round(1).to_dict(), "best mean", best.mean().round(3).to_dict())
