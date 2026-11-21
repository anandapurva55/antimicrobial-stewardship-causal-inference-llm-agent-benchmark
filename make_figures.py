"""Render the README figures into docs/ from golden_results.json, ground_truth.json and runs/.

    python make_figures.py
"""
import glob, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from asp_common import load_results
from generate_data import EFFECT, ELDERLY_MULT

os.makedirs("docs", exist_ok=True)
BLUE, ORANGE, GREY, TEAL, AMBER, PURPLE, NAVY = "#2a6fdb", "#e0862b", "#6b7280", "#1b9e8a", "#f2b134", "#7b4fd6", "#1f2a44"
plt.rcParams.update({"text.color": NAVY, "axes.labelcolor": NAVY, "axes.edgecolor": NAVY, "xtick.color": NAVY, "ytick.color": NAVY,
                     "axes.titlecolor": NAVY, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})
df = pd.read_csv("asp_study_data.csv")
truth = json.load(open("ground_truth.json"))
G = load_results("golden_results.json")


# 1. event study vs truth ------------------------------------------------------------
def true_by_event_time():
    t = df[df.asp_active == 1].copy()
    t["k"] = t.year - t.asp_adoption_year
    t["tau"] = [EFFECT[g][k] * (ELDERLY_MULT if a >= 65 else 1) for g, k, a in zip(t.asp_adoption_year, t.k, t.age)]
    return t.groupby("k").apply(lambda x: x.tau.mean())


es = G["event_study"].sort_values("event_time")
tr = true_by_event_time()
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.errorbar(es.event_time, es.coef, yerr=[es.coef - es.ci_low, es.ci_high - es.coef], fmt="o", color=BLUE,
            capsize=4, label="Callaway-Sant'Anna estimate (95% bootstrap CI)")
ax.plot(list(tr.index) + [-1], list(tr.values) + [0], "D", color=ORANGE, mfc="none", ms=8, label="True effect (generator)")
ax.axhline(0, color=NAVY, lw=.8); ax.axvspan(-4.6, -0.5, color=GREY, alpha=.07)
ax.text(-4.5, -4.6, "pre-adoption leads\n(parallel-trends check)", color=GREY, fontsize=8, va="bottom")
ax.set_xlabel("Years since ASP adoption"); ax.set_ylabel("Effect on antibiotic duration (days)")
ax.set_title("Event study: leads are flat, post-adoption effects track the truth")
ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1, 0.62), fontsize=8.5); fig.tight_layout(); fig.savefig("docs/event_study.png"); plt.close(fig)

# 2. TWFE vs staggered-robust vs truth -----------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 3.4))
names = ["True ATT", "Plain TWFE", "Callaway-Sant'Anna"]
vals = [truth["true_att_treated_days"], G["twfe_att_estimate"], G["did_att_estimate"]]
cols = [ORANGE, AMBER, BLUE]
ax.barh(names, vals, color=cols, height=.55)
for i, v in enumerate(vals):
    ax.text(v - 0.05, i, f"{v:.2f}", va="center", ha="right", color="white", fontweight="bold")
ax.set_xlim(-3.8, 0); ax.set_xlabel("ATT on antibiotic duration (days)")
ax.set_title("Plain TWFE is biased under staggered, dynamic effects"); fig.tight_layout(); fig.savefig("docs/att_comparison.png"); plt.close(fig)

# 3. Kaplan-Meier ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4))
for g, lab, c in ((0, "No ASP", ORANGE), (1, "ASP active", BLUE)):
    m = df.asp_active == g
    KaplanMeierFitter(label=lab).fit(df.time_to_improvement[m], df.event_observed[m]).plot_survival_function(ax=ax, color=c, ci_alpha=.12)
ax.set_xlabel("Days"); ax.set_ylabel("P(not yet improved)"); ax.set_title(f"Kaplan-Meier (Cox HR {G['cox_hr_asp']:.2f}, robust SE {G['cox_se_asp']:.3f})")
ax.legend(frameon=False); fig.tight_layout(); fig.savefig("docs/km_curves.png"); plt.close(fig)

# 4. CATE by age group ----------------------------------------------------------------
df["cate"] = G["cate_estimates"]
fig, ax = plt.subplots(figsize=(6.5, 4))
for grp, c in (("<40", BLUE), ("40-65", TEAL), ("65+", PURPLE)):
    ax.hist(df.loc[(df.age_group == grp) & (df.asp_active == 1), "cate"], bins=50, alpha=.55, color=c, density=True,
            label=f"{grp} (mean {df.loc[(df.age_group == grp) & (df.asp_active == 1), 'cate'].mean():.2f})")
lo, hi = G["cate_elderly_minus_young_ci"]
ax.set_xlabel("Estimated treatment effect, treated patients (days)"); ax.set_ylabel("Density")
ax.set_title(f"X-learner CATE: 65+ vs <40 difference, 95% CI [{lo:.2f}, {hi:.2f}]", fontsize=10)
ax.legend(frameon=False, title="Age group"); fig.tight_layout(); fig.savefig("docs/cate_by_age.png"); plt.close(fig)

# 5. golden unit-test output as an image (stands in for a screenshot) -----------------
from asp_tests import run_tests
rows = run_tests(G)
fig, ax = plt.subplots(figsize=(10, 0.24 * (len(rows) + 2) + .4)); ax.axis("off"); fig.patch.set_facecolor("#f4f6fb")
for i, r in enumerate(rows):
    ax.text(0, 1 - i * 0.07, f"{r['id']:>3}  {'PASS' if r['passed'] else 'FAIL'}  {r['desc']}", family="monospace", fontsize=8.5,
            color="#0f6b5e" if r["passed"] else "#9a6700", va="top", transform=ax.transAxes)
ax.text(0, 1 - (len(rows) + 1) * 0.07, f"SCORE: {sum(r['passed'] for r in rows)}/{len(rows)}", family="monospace", fontsize=10,
        fontweight="bold", color=NAVY, va="top", transform=ax.transAxes)
fig.savefig("docs/golden_unit_tests.png", facecolor=fig.get_facecolor(), bbox_inches="tight"); plt.close(fig)

# 6. agent runs -----------------------------------------------------------------------
runs = []
for sp in sorted(glob.glob("runs/*/*/trial_*/score.json")):
    d = os.path.dirname(sp); parts = d.split(os.sep)
    s_ = json.load(open(sp)); m = json.load(open(os.path.join(d, "meta.json")))
    runs.append(dict(model=parts[1].replace("gemini__", ""), cond=m["condition"], trial=m["trial"], ok=bool(m["executed_ok"] and os.path.exists(os.path.join(d, "results.json")) and json.load(open(os.path.join(d, "results.json")))),
                     score=s_["score"], tests=s_["tests"], attempts=m["attempts"]))
ALL = pd.DataFrame(runs)
BASE = "gemini-2.5-pro"

if len(ALL):
    # 6a/6b: Gemini 2.5 Pro, original vs pinned
    R = ALL[ALL.model == BASE]
    fig, ax = plt.subplots(figsize=(7, 4)); rng = np.random.default_rng(0)
    for i, c in enumerate(["original", "pinned"]):
        sub = R[R.cond == c]
        ax.scatter(i + rng.uniform(-.12, .12, len(sub)), sub.score, c=[BLUE if o else AMBER for o in sub.ok], s=55, zorder=3, edgecolor="white")
        ax.hlines(sub.score.mean(), i - .25, i + .25, color=NAVY, lw=2, zorder=4)
        ax.text(i + .28, sub.score.mean(), f"mean {sub.score.mean():.1f}", va="center", fontsize=9)
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"original (n={sum(R.cond == 'original')})", f"pinned outcome (n={sum(R.cond == 'pinned')})"])
    ax.set_xlim(-.5, 1.7); ax.set_ylim(-.5, 14.8); ax.axhline(14, color=TEAL, ls="--", lw=1); ax.text(1.68, 14.05, "golden 14/14", ha="right", fontsize=8, color=TEAL)
    ax.set_ylabel("Score (of 14 tests)"); ax.set_title("Gemini 2.5 Pro, per trial")
    ax.scatter([], [], c=BLUE, label="valid results"); ax.scatter([], [], c=AMBER, label="no valid results (score 0)")
    ax.legend(frameon=False, loc="lower right", fontsize=8.5); fig.tight_layout(); fig.savefig("docs/agent_scores.png"); plt.close(fig)

    ids = [f"S{i}" for i in range(1, 15)]
    passed = lambda r, i: next(t["passed"] for t in r.tests if t["id"] == i)
    Rs = R.sort_values(["cond", "trial"]); M = np.array([[passed(r, i) for r in Rs.itertuples()] for i in ids], float)
    fig, ax = plt.subplots(figsize=(0.42 * len(Rs) + 2.2, 5))
    ax.imshow(M, cmap=matplotlib.colors.ListedColormap([AMBER, TEAL]), aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(range(14)); ax.set_yticklabels(ids); ax.set_xticks(range(len(Rs)))
    ax.set_xticklabels([f"{r.cond[0].upper()}{r.trial}" for r in Rs.itertuples()], fontsize=8)
    ax.set_xlabel("Trial (O = original, P = pinned)"); ax.set_title("Gemini 2.5 Pro: tests passed (green) or failed (red) per trial")
    ax.spines[:].set_visible(False); fig.tight_layout(); fig.savefig("docs/test_matrix.png"); plt.close(fig)

    # 6c/6d: comparison across models, original prompt only
    COMPARE_N = 5  # equal trials per model (trials 0-4) so the comparison is balanced
    O = ALL[(ALL.cond == "original") & (ALL.trial < COMPARE_N)]
    if O.model.nunique() > 1:
        fam = lambda m: GREY if "2.5" in m else TEAL if "-3" in m else BLUE
        lb = O.groupby("model").agg(mean=("score", "mean"), ran=("ok", "sum"), n=("score", "size"), att=("attempts", "mean")).sort_values("mean")
        fig, ax = plt.subplots(figsize=(9, 0.6 * len(lb) + 1.6))
        ax.barh(lb.index, lb["mean"], color=[fam(m) for m in lb.index], height=.62)
        for i, (m, r) in enumerate(lb.iterrows()):
            rank = 1 + sum(round(x, 2) > round(r["mean"], 2) for x in lb["mean"])  # tied models share a rank
            ax.text(r["mean"] + .2, i, f"#{rank}  {r['mean']:.1f}/14  |  valid {int(r['ran'])}/{int(r['n'])}  |  {r['att']:.1f} attempts", va="center", fontsize=9)
        ax.set_xlim(0, 21); ax.set_xticks(range(0, 15, 2)); ax.set_xlabel("Mean score over 5 trials (of 14 tests)")
        ax.set_title(f"Leaderboard: 5 trials per model, original prompt (higher is better)")
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=c, label=l) for l, c in (("Gemini 2.5", GREY), ("Gemini 3.x", TEAL), ("Gemini 4", BLUE))],
                  frameon=False, loc="lower right", fontsize=8.5); fig.tight_layout(); fig.savefig("docs/model_leaderboard.png"); plt.close(fig)

        order = list(O.groupby("model").score.mean().sort_values().index)
        fig, ax = plt.subplots(figsize=(8, 0.62 * len(order) + 1.4)); rng = np.random.default_rng(1)
        for i, m in enumerate(order):
            sub = O[O.model == m]
            ax.scatter(sub.score, i + rng.uniform(-.15, .15, len(sub)), c=[BLUE if o else AMBER for o in sub.ok], s=48, zorder=3, edgecolor="white")
            ax.vlines(sub.score.mean(), i - .3, i + .3, color=NAVY, lw=2, zorder=4)
            ax.text(14.9, i, f"mean {sub.score.mean():.1f} | valid {int(sub.ok.sum())}/{len(sub)}", va="center", fontsize=8.5)
        ax.set_yticks(range(len(order))); ax.set_yticklabels(order); ax.set_xlim(-.5, 19.5); ax.set_xticks(range(0, 15, 2))
        ax.axvline(14, color=TEAL, ls="--", lw=1); ax.set_xlabel("Score (of 14 tests); dark tick = mean")
        ax.set_title("Model comparison: 5 trials per model, original prompt (blue: valid results, amber: none)", fontsize=10.5)
        fig.tight_layout(); fig.savefig("docs/model_comparison_scores.png"); plt.close(fig)

        ex = O[O.ok]; models = [m for m in order[::-1] if (ex.model == m).any()]
        M = np.array([[np.mean([passed(r, i) for r in ex[ex.model == m].itertuples()]) for i in ids] for m in models])
        fig, ax = plt.subplots(figsize=(10, 0.55 * len(models) + 1.6))
        im = ax.imshow(M, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("pf", ["#f7c873", "#f4f6fb", "#1b9e8a"]), vmin=0, vmax=1, aspect="auto")
        for a in range(M.shape[0]):
            for b in range(M.shape[1]):
                ax.text(b, a, f"{M[a, b]:.0%}", ha="center", va="center", fontsize=7.5, color=NAVY)
        ax.set_xticks(range(14)); ax.set_xticklabels(ids); ax.set_yticks(range(len(models)))
        ax.set_yticklabels([f"{m} (n={int((ex.model == m).sum())})" for m in models])
        ax.set_title("Pass rate per test among runs with valid results (5 trials per model, original prompt)")
        ax.spines[:].set_visible(False); fig.tight_layout(); fig.savefig("docs/model_comparison_tests.png"); plt.close(fig)
print("figures:", sorted(os.listdir("docs")))
