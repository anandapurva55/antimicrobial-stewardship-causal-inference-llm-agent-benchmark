"""Render the README figures into docs/ from the golden results, the ground truth and the stored agent trials.

    python scripts/make_figures.py

Design rules: one validated categorical palette (blue, orange, aqua, yellow, magenta, violet; no red, no black),
text in ink (never in a series colour), thin rounded marks, hairline grids, direct labels, and a colour that always
belongs to the same entity (each model keeps its colour in every figure).
"""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from scipy.stats import gaussian_kde

from asp_benchmark.common import load_results
from asp_benchmark.data_generation import ELDERLY_MULT, EFFECT
from asp_benchmark.paths import CSV, DOCS_DIR, GOLDEN_RESULTS, RESULTS_DIR, RUNS_DIR, TRUTH
from asp_benchmark.scoring import run_tests
from asp_benchmark.stats import bootstrap_ci

# ---------------------------------------------------------------- palette and style
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#1f2a44", "#55607a", "#8b93a7", "#e6e9f0"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET = "#2a78d6", "#e8872a", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"
PASS, FAIL = "#1b9e8a", "#f2b134"
MODEL_COLORS = {"gemini-4-argon": BLUE, "gemini-4-carbon": ORANGE, "gemini-3.8-flash": AQUA,
                "gemini-4-hydrogen": YELLOW, "gemini-4-neon": MAGENTA, "gemini-2.5-pro": VIOLET}
PASS_RAMP = LinearSegmentedColormap.from_list("pass", [(0.0, "#f2b134"), (0.5, "#f3ead3"), (1.0, "#7fcfb8")])
TEST_NAMES = {1: "Cox HR", 2: "Cox SE", 3: "PH test", 4: "Panel", 5: "Event study", 6: "Pre-trend", 7: "TWFE", 8: "Staggered\nATT",
              9: "CATE\nspread", 10: "CATE\nby age", 11: "CATE CI", 12: "ICC", 13: "AIC", 14: "Mixed\nmodel"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "savefig.dpi": 200,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.edgecolor": GRID, "axes.linewidth": 1,
    "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.8,
})

DOCS_DIR.mkdir(exist_ok=True)
df = pd.read_csv(CSV)
truth = json.load(open(TRUTH))
G = load_results(GOLDEN_RESULTS)


def canvas(w, h, left=0.9, right=0.4, top=1.15, bottom=0.75):
    """A figure with a fixed axes box (inches), so mark geometry can be computed exactly."""
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([left / w, bottom / h, (w - left - right) / w, (h - bottom - top) / h])
    return fig, ax


def heading(fig, title, subtitle=None):
    w, h = fig.get_size_inches()
    fig.text(0.4 / w, 1 - 0.3 / h, title, fontsize=15, fontweight="bold", va="top", ha="left")
    if subtitle:
        fig.text(0.4 / w, 1 - 0.64 / h, subtitle, fontsize=9.5, color=INK2, va="top", ha="left")


def save(fig, name):
    fig.savefig(DOCS_DIR / name)
    plt.close(fig)


def rbar(ax, y, x0, x1, thick_pt, color, z=3):
    """Horizontal bar with a rounded data end and a square baseline end."""
    fig = ax.figure
    axw = ax.get_position().width * fig.get_figwidth()
    span = abs(ax.get_xlim()[1] - ax.get_xlim()[0])
    r = (thick_pt / 2) / 72 * span / axw
    xe = x1 - np.sign(x1 - x0) * r
    ax.plot([x0, xe], [y, y], lw=thick_pt, color=color, solid_capstyle="butt", zorder=z)
    ax.plot([xe], [y], marker="o", ms=thick_pt, mfc=color, mec=color, mew=0, zorder=z)


def vgrid(ax):
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)


def cell(ax, x, y, color, w=1.0, h=1.0, gap=0.08, rounding=0.16):
    ax.add_patch(FancyBboxPatch((x + gap, y + gap), w - 2 * gap, h - 2 * gap, boxstyle=f"round,pad=0,rounding_size={rounding}",
                                fc=color, ec="none", zorder=2))


# ---------------------------------------------------------------- stored agent trials
runs = []
for sp in sorted(glob.glob(str(RUNS_DIR / "*/*/trial_*/score.json"))):
    d = os.path.dirname(sp)
    parts = d.split(os.sep)[-3:]
    score = json.load(open(sp))
    meta = json.load(open(os.path.join(d, "meta.json")))
    rp = os.path.join(d, "results.json")
    valid = bool(meta["executed_ok"] and os.path.exists(rp) and json.load(open(rp)))
    runs.append(dict(model=parts[0].replace("gemini__", ""), cond=meta["condition"], trial=meta["trial"], ok=valid,
                     score=score["score"], tests=score["tests"], attempts=meta["attempts"]))
ALL = pd.DataFrame(runs)
passed = lambda r, i: next(t["passed"] for t in r.tests if t["id"] == i)


# ================================================================ 1. event study
def true_by_event_time():
    t = df[df.asp_active == 1].copy()
    t["k"] = t.year - t.asp_adoption_year
    t["tau"] = [EFFECT[g][k] * (ELDERLY_MULT if a >= 65 else 1) for g, k, a in zip(t.asp_adoption_year, t.k, t.age)]
    return t.groupby("k").tau.mean()


es = G["event_study"].sort_values("event_time")
ref = pd.DataFrame([dict(event_time=-1, coef=0.0, ci_low=0.0, ci_high=0.0)])
es_full = pd.concat([es[["event_time", "coef", "ci_low", "ci_high"]], ref]).sort_values("event_time")
tr = true_by_event_time()
fig, ax = canvas(8.6, 5.0, left=0.95, right=0.5)
ax.axvspan(-4.6, -0.5, color=BLUE, alpha=0.05, lw=0)
ax.axhline(0, color=GRID, lw=1.2, zorder=1)
ax.fill_between(es_full.event_time, es_full.ci_low, es_full.ci_high, color=BLUE, alpha=0.14, lw=0, zorder=2)
ax.plot(es_full.event_time, es_full.coef, color=BLUE, lw=2, zorder=3)
ax.scatter(es.event_time, es.coef, s=70, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=4)
ax.scatter([-1], [0], s=70, color=SURFACE, edgecolor=BLUE, linewidth=2, zorder=4)
tx = list(tr.index) + [-1]
ty = list(tr.values) + [0.0]
ax.scatter(tx, ty, s=60, marker="D", color=SURFACE, edgecolor=ORANGE, linewidth=2, zorder=5)
ax.axvline(-0.5, color=INK2, lw=1, zorder=1)
ax.text(-0.42, 0.55, "ASP adopted", fontsize=9, color=INK2, va="center")
ax.text(-4.45, 0.55, "Before adoption: coefficients near 0", fontsize=9, color=INK2, va="center")
ax.set_xlim(-4.6, 2.4)
ax.set_ylim(-5.4, 1.1)
ax.set_xticks(range(-4, 3))
ax.set_xlabel("Years since ASP adoption")
ax.set_ylabel("Effect on antibiotic duration (days)")
ax.grid(axis="y", zorder=0)
ax.set_axisbelow(True)
handles = [Line2D([], [], color=BLUE, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, label="Estimate with 95% interval"),
           Line2D([], [], ls="", marker="D", ms=7, mfc=SURFACE, mec=ORANGE, mew=2, label="True effect built into the data")]
ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9.5, labelcolor=INK, bbox_to_anchor=(1.0, 0.88))
heading(fig, "Event study: flat before adoption, then tracking the true effect",
        "Callaway-Sant'Anna estimates by years since adoption")
save(fig, "event_study.png")

# ================================================================ 2. TWFE against staggered estimator
fig, ax = canvas(8.4, 3.9, left=2.5, right=0.5, top=1.15, bottom=0.8)
rows = [("True effect", truth["true_att_treated_days"], ORANGE, None),
        ("Callaway-Sant'Anna", G["did_att_estimate"], BLUE, None),
        ("Plain TWFE", G["twfe_att_estimate"], VIOLET, "understates by %.2f days" % abs(G["twfe_att_estimate"] - truth["true_att_treated_days"]))]
ax.set_xlim(0, 5.4)
ax.set_ylim(2.6, -0.6)
ax.set_xticks(range(0, 5))
for i, (name, v, c, note) in enumerate(rows):
    rbar(ax, i, 0, abs(v), 20, c)
    ax.text(3.55, i - 0.05, f"{v:.2f} days", fontsize=11, fontweight="bold", va="center")
    if note:
        ax.text(3.55, i + 0.33, note, fontsize=8.5, color=INK2, va="center")
lo_hi = tuple(sorted(abs(v) for v in json.load(open(RESULTS_DIR / "golden_extras.json"))["did_att_ci"]))
ax.plot(lo_hi, [1, 1], color=INK, lw=1.4, zorder=5, solid_capstyle="round")
for x in lo_hi:
    ax.plot([x, x], [0.88, 1.12], color=INK, lw=1.4, zorder=5)
ax.set_yticks(range(3))
ax.set_yticklabels([r[0] for r in rows], fontsize=11, color=INK)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_xlabel("Reduction in antibiotic duration (days)")
vgrid(ax)
heading(fig, "Plain TWFE understates the effect; Callaway-Sant'Anna recovers it",
        "Whisker: 95% bootstrap interval of the Callaway-Sant'Anna estimate")
save(fig, "att_comparison.png")

# ================================================================ 3. Kaplan-Meier
fig, ax = canvas(8.6, 5.2, left=0.95, right=1.6)
curves = []
for g, lab, c in ((0, "No ASP", ORANGE), (1, "ASP active", BLUE)):
    m = df.asp_active == g
    k = KaplanMeierFitter().fit(df.time_to_improvement[m], df.event_observed[m])
    sf, ci = k.survival_function_.iloc[:, 0], k.confidence_interval_
    ax.fill_between(sf.index, ci.iloc[:, 0], ci.iloc[:, 1], step="post", color=c, alpha=0.12, lw=0, zorder=2)
    ax.step(sf.index, sf.values, where="post", color=c, lw=2, zorder=3)
    med = k.median_survival_time_
    ax.plot([med, med], [0, 0.5], color=c, lw=1, zorder=2)
    ax.scatter([med], [0.5], s=60, color=c, edgecolor=SURFACE, linewidth=2, zorder=4)
    curves.append((lab, c, sf, med))
ax.axhline(0.5, color=GRID, lw=1, zorder=1)
for lab, c, sf, med in curves:
    end_x = min(28, sf.index.max())
    ax.text(28.6, float(sf[sf.index <= end_x].iloc[-1]), lab, fontsize=10, fontweight="bold", va="center")
    ax.text(med + (-0.35 if lab == "ASP active" else 0.35), 0.06, f"median {med:.1f} days", fontsize=9, color=INK2, va="center", ha="right" if lab == "ASP active" else "left")
ax.set_xlim(0, 28)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Days since admission")
ax.set_ylabel("Share of patients not yet improved")
ax.grid(axis="y", zorder=0)
ax.set_axisbelow(True)
heading(fig, "Patients under ASP improve sooner",
        f"Kaplan-Meier curves with 95% intervals; Cox hazard ratio {G['cox_hr_asp']:.2f}, standard error {G['cox_se_asp']:.3f} (clustered by hospital)")
save(fig, "km_curves.png")

# ================================================================ 4. effect by age group (ridgeline)
df["cate"] = G["cate_estimates"]
treated = df[df.asp_active == 1]
ramp = {"<40": "#86b6ef", "40-65": "#3987e5", "65+": "#1c5cab"}
fig, ax = canvas(8.4, 5.0, left=1.9, right=1.6, top=1.25, bottom=0.8)
grid = np.linspace(-6.2, 1.2, 400)
for i, grp in enumerate(["<40", "40-65", "65+"]):
    vals = treated.loc[treated.age_group == grp, "cate"].values
    dens = gaussian_kde(vals)(grid)
    dens = dens / dens.max() * 0.86
    base = i + 1.0
    ax.fill_between(grid, base, base - dens, color=ramp[grp], alpha=0.30, lw=0, zorder=2)
    ax.plot(grid, base - dens, color=ramp[grp], lw=2, zorder=3)
    ax.plot([grid[0], grid[-1]], [base, base], color=GRID, lw=1, zorder=1)
    mu = vals.mean()
    ax.plot([mu, mu], [base, base - np.interp(mu, grid, dens)], color=INK, lw=1.2, zorder=4)
    ax.scatter([mu], [base], s=55, color=ramp[grp], edgecolor=SURFACE, linewidth=2, zorder=5)
    ax.text(1.35, base - 0.3, f"mean {mu:.2f} days", fontsize=9.5, va="center")
    label = {"<40": "Under 40", "40-65": "40 to 64", "65+": "65 and over"}[grp]
    ax.text(-0.02, base - 0.36, label, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=11, color=INK)
    ax.text(-0.02, base - 0.12, f"{len(vals):,} treated", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=8.5, color=MUTED)
ax.set_xlim(-6.2, 3.4)
ax.set_ylim(3.05, 0.0)
ax.set_xticks(range(-6, 2))
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlabel("Estimated effect on antibiotic duration (days; more negative is a larger reduction)")
vgrid(ax)
lo, hi = G["cate_elderly_minus_young_ci"]
heading(fig, "Older patients benefit more from ASP",
        f"X-learner effects for treated patients. 65 and over minus under 40: bootstrap 95% interval {lo:.2f} to {hi:.2f} days")
save(fig, "cate_by_age.png")

# ================================================================ 5. golden unit tests as a report card
test_rows = run_tests(G)
n_ok = sum(r["passed"] for r in test_rows)
W, H = 10.4, 0.62 + 0.36 * len(test_rows) + 1.05
fig = plt.figure(figsize=(W, H))
fig.text(0.4 / W, 1 - 0.3 / H, "Golden solution: all 14 scored tests passed" if n_ok == len(test_rows) else f"Golden solution: {n_ok} of {len(test_rows)} tests passed",
         fontsize=15, fontweight="bold", va="top")
bar = fig.add_axes([0.4 / W, 1 - 0.95 / H, 4.0 / W, 0.2 / H])
bar.set_xlim(0, 1)
bar.set_ylim(0, 1)
bar.axis("off")
bar.plot([0.02, 0.98], [0.5, 0.5], lw=11, color=GRID, solid_capstyle="round")
bar.plot([0.02, 0.02 + 0.96 * n_ok / len(test_rows)], [0.5, 0.5], lw=11, color=PASS, solid_capstyle="round")
fig.text(4.65 / W, 1 - 0.85 / H, f"{n_ok} of {len(test_rows)}", fontsize=11, fontweight="bold", va="center")
body = fig.add_axes([0.4 / W, 0.25 / H, (W - 0.8) / W, (H - 1.4) / H])
body.set_xlim(0, W - 0.8)
body.set_ylim(len(test_rows), 0)
body.axis("off")
for i, r in enumerate(test_rows):
    if i % 2 == 0:
        body.add_patch(FancyBboxPatch((0, i + 0.05), W - 0.8, 0.9, boxstyle="round,pad=0,rounding_size=0.12", fc="#f2f4f8", ec="none", mutation_aspect=1))
    body.text(0.15, i + 0.5, r["id"], fontsize=9.5, fontweight="bold", color=INK2, va="center")
    body.text(0.85, i + 0.5, r["desc"], fontsize=9.5, va="center")
    body.add_patch(FancyBboxPatch((W - 0.8 - 0.95, i + 0.22), 0.8, 0.56, boxstyle="round,pad=0,rounding_size=0.25", fc=PASS if r["passed"] else FAIL, ec="none"))
    body.text(W - 0.8 - 0.55, i + 0.5, "PASS" if r["passed"] else "FAIL", fontsize=8.5, fontweight="bold", color="white", ha="center", va="center")
save(fig, "golden_unit_tests.png")

# ================================================================ agent trials: shared helpers
LEADERBOARD = ALL[(ALL.cond == "original") & (ALL.trial < 5)]


def model_stats(sub):
    sc = sub.score.values
    lo, hi = bootstrap_ci(sc)
    return dict(mean=sc.mean(), lo=lo, hi=hi, valid=int(sub.ok.sum()), n=len(sub), att=sub.attempts.mean())


if len(ALL):
    stats = {m: model_stats(LEADERBOARD[LEADERBOARD.model == m]) for m in LEADERBOARD.model.unique()}
    order = sorted(stats, key=lambda m: (-stats[m]["mean"], stats[m]["att"]))
    ranks = {m: 1 + sum(round(stats[o]["mean"], 2) > round(stats[m]["mean"], 2) for o in stats) for m in order}

    # ============================================================ 6. leaderboard
    fig, ax = canvas(9.6, 5.3, left=2.9, right=0.5, top=1.2, bottom=0.8)
    ax.set_xlim(0, 20.5)
    ax.set_ylim(len(order) - 0.45, -0.85)
    ax.set_xticks(range(0, 15, 2))
    vgrid(ax)
    ax.axvline(14, color=PASS, lw=1.2, zorder=1)
    ax.text(14.15, -0.62, "all 14 tests", fontsize=8.5, color=INK2, va="center")
    for i, m in enumerate(order):
        s = stats[m]
        rbar(ax, i, 0, max(s["mean"], 0.01), 22, MODEL_COLORS[m])
        if s["hi"] - s["lo"] > 1e-9:
            ax.plot([s["lo"], s["hi"]], [i, i], color=INK, lw=1.4, zorder=5, solid_capstyle="round")
            for x in (s["lo"], s["hi"]):
                ax.plot([x, x], [i - 0.12, i + 0.12], color=INK, lw=1.4, zorder=5)
        x_label = 14.45  # one aligned column of labels, clear of the reference line
        ax.text(x_label, i - 0.13, f"{s['mean']:.1f}", fontsize=12, fontweight="bold", va="center")
        ax.text(x_label, i + 0.2, f"valid {s['valid']}/{s['n']}  |  {s['att']:.1f} attempts", fontsize=8.5, color=INK2, va="center")
        ax.text(-0.02, i, m, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=11, color=INK)
        ax.scatter([-0.36], [i], s=330, transform=ax.get_yaxis_transform(), color="#eef1f7", clip_on=False, zorder=1)
        ax.text(-0.36, i, str(ranks[m]), transform=ax.get_yaxis_transform(), ha="center", va="center", fontsize=9.5, fontweight="bold", color=INK2)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Mean score over 5 trials (14 tests). Whisker: 95% bootstrap interval")
    heading(fig, "Leaderboard: six Gemini models on the causal inference task",
            "Original prompt, trials 0 to 4 of every model. Tied models share a rank")
    save(fig, "leaderboard.png")

    # ============================================================ 7. every trial of every model
    fig, ax = canvas(9.6, 5.3, left=2.0, right=2.6, top=1.2, bottom=0.8)
    ax.set_xlim(-0.6, 14.6)
    ax.set_ylim(len(order) - 0.45, -0.55)
    ax.set_xticks(range(0, 15, 2))
    vgrid(ax)
    rng = np.random.default_rng(1)
    for i, m in enumerate(order):
        sub = LEADERBOARD[LEADERBOARD.model == m]
        ordered = sub.sort_values(["score", "trial"]).reset_index(drop=True)
        seen = {}
        for _, r in ordered.iterrows():
            k = seen.get(r.score, 0)
            seen[r.score] = k + 1
            same = int((ordered.score == r.score).sum())
            j = (k - (same - 1) / 2) * 0.17  # dots with the same score stack in a column, never overlapping
            if r.ok:
                ax.scatter([r.score], [i + j], s=62, color=MODEL_COLORS[m], edgecolor=SURFACE, linewidth=1.6, zorder=4)
            else:
                ax.scatter([r.score], [i + j], s=62, facecolor=SURFACE, edgecolor=MODEL_COLORS[m], linewidth=2, zorder=4)
        s = stats[m]
        ax.text(14.95, i, f"mean {s['mean']:.1f}", fontsize=10, fontweight="bold", va="center")
        ax.text(-0.02, i, m, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=11, color=INK)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Score of each trial (14 tests)")
    handles = [Line2D([], [], marker="o", ms=9, mfc=INK2, mec=SURFACE, mew=2, ls="", label="valid results"),
               Line2D([], [], marker="o", ms=9, mfc=SURFACE, mec=INK2, mew=2, ls="", label="no valid results (score 0)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False, fontsize=9, labelcolor=INK2)
    heading(fig, "Every trial, every model", "Five trials per model on the original prompt")
    save(fig, "model_comparison_scores.png")

    # ============================================================ 8. pass rate of each test by model
    valid_runs = LEADERBOARD[LEADERBOARD.ok]
    ids = [f"S{i}" for i in range(1, 15)]
    fig, ax = canvas(11.4, 4.6, left=2.3, right=0.35, top=1.2, bottom=0.95)
    ax.set_xlim(0, 14)
    ax.set_ylim(len(order), 0)
    ax.axis("off")
    for i, m in enumerate(order):
        sub = valid_runs[valid_runs.model == m]
        ax.text(-0.15, i + 0.4, m, ha="right", va="center", fontsize=10.5, color=INK)
        ax.text(-0.15, i + 0.72, f"{len(sub)} valid trials", ha="right", va="center", fontsize=8, color=MUTED)
        for j, tid in enumerate(ids):
            rate = float(np.mean([passed(r, tid) for r in sub.itertuples()])) if len(sub) else 0.0
            cell(ax, j, i, PASS_RAMP(rate))
            if rate < 1:
                ax.text(j + 0.5, i + 0.5, f"{rate:.0%}", ha="center", va="center", fontsize=8.5, fontweight="bold", color=INK)
    for j in range(14):
        ax.text(j + 0.5, len(order) + 0.12, f"S{j + 1}", ha="center", va="top", fontsize=8.5, fontweight="bold", color=INK2)
        ax.text(j + 0.5, len(order) + 0.5, TEST_NAMES[j + 1], ha="center", va="top", fontsize=7.5, color=MUTED, linespacing=1.1)
    heading(fig, "Which tests each model passes",
            "Pass rate among trials with valid results: teal is all passed, amber is none, and a percentage is shown wherever a test was missed")
    save(fig, "model_comparison_tests.png")

    # ============================================================ 9. gemini-2.5-pro, original against pinned prompt
    P = ALL[ALL.model == "gemini-2.5-pro"]
    fig, ax = canvas(8.2, 5.2, left=0.95, right=0.4, top=1.3, bottom=0.85)
    ax.set_xlim(-0.5, 1.95)
    ax.set_ylim(-0.6, 14.6)
    ax.set_yticks(range(0, 15, 2))
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(14, color=PASS, lw=1.2, zorder=1)
    for i, c in enumerate(["original", "pinned"]):
        sub = P[P.cond == c]
        ordered = sub.sort_values(["score", "trial"]).reset_index(drop=True)
        jitter = np.linspace(-0.3, 0.3, len(ordered))
        for (_, r), j in zip(ordered.iterrows(), jitter):
            if r.ok:
                ax.scatter([i + j], [r.score], s=95, color=VIOLET, edgecolor=SURFACE, linewidth=2, zorder=4)
            else:
                ax.scatter([i + j], [r.score], s=95, facecolor=SURFACE, edgecolor=VIOLET, linewidth=2.2, zorder=4)
        mean, (lo, hi) = sub.score.mean(), bootstrap_ci(sub.score.values)
        ax.plot([i + 0.46, i + 0.46], [lo, hi], color=INK, lw=1.4, zorder=5, solid_capstyle="round")
        for y in (lo, hi):
            ax.plot([i + 0.42, i + 0.5], [y, y], color=INK, lw=1.4, zorder=5)
        ax.scatter([i + 0.46], [mean], s=70, marker="D", color=INK, edgecolor=SURFACE, linewidth=1.6, zorder=6)
        ax.text(i + 0.56, mean + 0.35, f"mean {mean:.1f}", fontsize=10.5, fontweight="bold", va="center")
        ax.text(i + 0.56, mean - 0.55, f"interval {lo:.1f} to {hi:.1f}", fontsize=8.5, color=INK2, va="center")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Original prompt\n({int((P.cond == 'original').sum())} trials)", f"Outcome stated in the prompt\n({int((P.cond == 'pinned').sum())} trials)"], fontsize=10.5)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Score (14 tests)")
    handles = [Line2D([], [], marker="o", ms=9, mfc=VIOLET, mec=SURFACE, mew=2, ls="", label="valid results"),
               Line2D([], [], marker="o", ms=9, mfc=SURFACE, mec=VIOLET, mew=2, ls="", label="no valid results (score 0)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=2, frameon=False, fontsize=9, labelcolor=INK2)
    heading(fig, "gemini-2.5-pro: naming the outcome does not change the result",
            "One dot per trial; the diamond is the mean and the whisker its 95% bootstrap interval")
    save(fig, "agent_scores.png")

    # ============================================================ 10. gemini-2.5-pro, tests passed in each trial
    Ps = P.sort_values(["cond", "trial"]).reset_index(drop=True)
    fig, ax = canvas(11.6, 6.1, left=1.9, right=0.35, top=1.55, bottom=0.85)
    n_orig = int((Ps.cond == "original").sum())
    xs = [k + (0.5 if k >= n_orig else 0) for k in range(len(Ps))]
    ax.set_xlim(-0.1, len(Ps) + 0.6)
    ax.set_ylim(14, 0)
    ax.axis("off")
    for k, r in enumerate(Ps.itertuples()):
        for i, tid in enumerate(ids):
            ok = passed(r, tid)
            cell(ax, xs[k], i, PASS if ok else FAIL, gap=0.07, rounding=0.2)
            if not ok:
                ax.text(xs[k] + 0.5, i + 0.52, "×", ha="center", va="center", fontsize=9, fontweight="bold", color="white")
        ax.text(xs[k] + 0.5, 14.15, f"{r.cond[0].upper()}{r.trial}", ha="center", va="top", fontsize=8.5, color=INK2)
    for i, tid in enumerate(ids):
        ax.text(-0.15, i + 0.52, f"{tid}  {TEST_NAMES[i + 1].replace(chr(10), ' ')}", ha="right", va="center", fontsize=9, color=INK2)
    for label, a, b in (("Original prompt", 0, n_orig - 1), ("Outcome stated in the prompt", n_orig, len(Ps) - 1)):
        ax.plot([xs[a] + 0.1, xs[b] + 0.9], [-0.35, -0.35], color=INK2, lw=1.2, clip_on=False, solid_capstyle="round")
        ax.text((xs[a] + xs[b] + 1) / 2, -0.55, label, ha="center", va="bottom", fontsize=10, fontweight="bold", clip_on=False)
    heading(fig, "gemini-2.5-pro: the tests each trial passed",
            "Teal passed, amber (with a cross) failed. A fully amber column is a trial whose code produced no valid results")
    save(fig, "test_matrix.png")

print("figures:", sorted(f for f in os.listdir(DOCS_DIR) if f.endswith(".png")))
