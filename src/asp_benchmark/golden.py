# %% [markdown]
# # Multi-Hospital ASP Causal Study: Expert Solution
# Primary outcome throughout: **`antibiotic_duration`** (survival uses `time_to_improvement`).
# Source of truth for `notebooks/golden_notebook.ipynb` (built with `scripts/build_notebooks.py`).
# Run as a script with: python -m asp_benchmark.golden

# %%
import json, os, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test, logrank_test
from linearmodels.panel import PanelOLS
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from asp_benchmark.common import save_results
from asp_benchmark.paths import CSV, GOLDEN_RESULTS, RESULTS_DIR, TRUTH
warnings.filterwarnings("ignore")
OUTCOME = "antibiotic_duration"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(CSV)
truth = json.load(open(TRUTH)) if TRUTH.exists() else {}
assert df.shape[0] > 10_000 and df.hospital_id.nunique() == 15 and set(df.year) == set(range(2017, 2023))
df["sex_binary"] = (df.sex == "M").astype(int)
for k in ("bloodstream", "respiratory", "urinary"):
    df[f"infection_{k}"] = (df.infection_type == k).astype(int)
df["ward_icu"] = (df.ward == "ICU").astype(int)
print(df.shape, "censoring:", round(1 - df.event_observed.mean(), 3))

# %% [markdown]
# ## 1. Survival analysis
# Kaplan-Meier + Cox PH. Patients are nested in hospitals, so the Cox SE is **clustered by hospital**
# (naive SE shown for comparison). PH assumption: Schoenfeld-residual test.

# %%
COX_X = ["asp_active", "age", "severity_score", "comorbidity_score", "immune_compromised",
         "sex_binary", "infection_bloodstream", "infection_respiratory", "infection_urinary", "ward_icu"]
cox_df = df[COX_X + ["time_to_improvement", "event_observed", "hospital_id"]]
cph_naive = CoxPHFitter().fit(cox_df.drop(columns="hospital_id"), "time_to_improvement", "event_observed")
cph = CoxPHFitter().fit(cox_df, "time_to_improvement", "event_observed", cluster_col="hospital_id")
cox_hr_asp = float(np.exp(cph.params_["asp_active"]))
cox_se_asp = float(cph.standard_errors_["asp_active"])
print(f"HR={cox_hr_asp:.3f}  robust SE={cox_se_asp:.4f}  naive SE={cph_naive.standard_errors_['asp_active']:.4f}")
ph = proportional_hazard_test(cph_naive, cox_df.drop(columns="hospital_id"), time_transform="rank")
ph_test_pvalues = {k: float(v) for k, v in ph.summary["p"].items()}
print({k: round(v, 3) for k, v in ph_test_pvalues.items()})

fig, ax = plt.subplots(figsize=(7, 5))
for g, lab in ((1, "ASP active"), (0, "No ASP")):
    m = df.asp_active == g
    KaplanMeierFitter(label=lab).fit(df.time_to_improvement[m], df.event_observed[m]).plot_survival_function(ax=ax)
ax.set_title("Kaplan-Meier: not yet improved"); plt.savefig(RESULTS_DIR / "km_curves.png", dpi=100); plt.close()

# %% [markdown]
# ## 2. Difference-in-differences (staggered adoption)
# Hospital-year panel of mean `antibiotic_duration` (90 rows: 15 hospitals x 6 years). Plain TWFE is reported, but with staggered
# adoption and dynamic/heterogeneous effects it can be biased, so the headline ATT is a Callaway-Sant'Anna
# estimator (never-treated controls, base period g-1, weights = treated patients per cell), with a
# hospital-cluster bootstrap. The 2017-2022 window gives the 2020 and 2021 cohorts 3 and 4 pre-periods, so the
# event study has several leads. The pre-trend test is the average of all leads (event time <= -2), with a bootstrap CI.

# %%
panel = (df.groupby(["hospital_id", "year"])
           .agg(mean_abx_dur=(OUTCOME, "mean"), n=(OUTCOME, "size"),
                asp_adoption_year=("asp_adoption_year", "first")).reset_index())
assert len(panel) == 90
did_panel = panel
panel["post_treated"] = (panel.year >= panel.asp_adoption_year).astype(int)
twfe = PanelOLS.from_formula("mean_abx_dur ~ post_treated + EntityEffects + TimeEffects",
                             data=panel.set_index(["hospital_id", "year"])).fit(cov_type="clustered", cluster_entity=True)
twfe_att_estimate = float(twfe.params["post_treated"])
print(f"TWFE ATT = {twfe_att_estimate:.3f}")

Y = panel.pivot(index="hospital_id", columns="year", values="mean_abx_dur")
N = panel.pivot(index="hospital_id", columns="year", values="n")
adopt = panel.groupby("hospital_id").asp_adoption_year.first()
COHORTS = [2020, 2021]
YEARS = sorted(panel.year.unique())


def cs(hosp_by_group, controls):
    """Return {(g,t): (att, weight)} using Y/N panels for the given hospital id lists."""
    out = {}
    for g in COHORTS:
        H = hosp_by_group[g]
        for t in YEARS:
            if t == g - 1:
                continue
            d = (Y.loc[H, t] - Y.loc[H, g - 1]).mean() - (Y.loc[controls, t] - Y.loc[controls, g - 1]).mean()
            out[(g, t)] = (d, float(N.loc[H, t].sum()))
    return out


def aggregate(est):
    post = [(v, w) for (g, t), (v, w) in est.items() if t >= g]
    overall = sum(v * w for v, w in post) / sum(w for _, w in post)
    pre = [(v, w) for (g, t), (v, w) in est.items() if t < g - 1]
    ev = {"pre": sum(v * w for v, w in pre) / sum(w for _, w in pre)}
    for e in sorted({t - g for (g, t) in est}):
        cells = [(v, w) for (g, t), (v, w) in est.items() if t - g == e]
        ev[e] = sum(v * w for v, w in cells) / sum(w for _, w in cells)
    return overall, ev


groups = {g: list(adopt[adopt == g].index) for g in COHORTS}
never = list(adopt[adopt == 9999].index)
overall, ev = aggregate(cs(groups, never))
rng = np.random.default_rng(0)
boots, boots_ev = [], []
for _ in range(1000):
    gb = {g: list(rng.choice(h, len(h))) for g, h in groups.items()}
    # bootstrap with replacement: .loc with duplicate ids is fine for means
    o, e = aggregate(cs(gb, list(rng.choice(never, len(never)))))
    boots.append(o); boots_ev.append(e)
did_att_estimate = float(overall)
ci = np.percentile(boots, [2.5, 97.5])
print(f"Callaway-Sant'Anna ATT = {did_att_estimate:.3f}  95% CI [{ci[0]:.2f}, {ci[1]:.2f}]  "
      f"(truth {truth.get('true_att_treated_days', float('nan')):.3f})")

rows = []
for e, v in ev.items():
    if e == "pre":
        continue
    b = np.array([x[e] for x in boots_ev])
    rows.append(dict(event_time=e, coef=float(v), ci_low=float(np.percentile(b, 2.5)),
                     ci_high=float(np.percentile(b, 97.5)), n_cohorts=sum(1 for g in COHORTS if (g + e) in YEARS)))
event_study = pd.DataFrame(rows)
print(event_study.round(3).to_string(index=False))
b = np.array([x["pre"] for x in boots_ev])
pre_trend_coeff = float(ev["pre"])
pre_trend_ci = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
pre_trend_pvalue = float(2 * (1 - __import__("scipy.stats").stats.norm.cdf(abs(pre_trend_coeff) / b.std())))
print(f"Pre-trend (average of all leads): {pre_trend_coeff:.3f}, p={pre_trend_pvalue:.3f}, CI {np.round(pre_trend_ci, 3)}; "
      f"min detectable effect ~{2.8 * b.std():.2f} days")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.errorbar(event_study.event_time, event_study.coef,
            yerr=[event_study.coef - event_study.ci_low, event_study.ci_high - event_study.coef], fmt="o", capsize=4)
ax.axhline(0, color="#1f2a44", lw=.8); ax.axvline(-0.5, color="#7b4fd6", ls="--")
ax.set_title("Event study (Callaway-Sant'Anna, bootstrap 95% CI)"); plt.savefig(RESULTS_DIR / "event_study.png", dpi=100); plt.close()

# %% [markdown]
# ## 3. Heterogeneous treatment effects
# T- and X-learner (gradient boosting). Because treatment varies by hospital-year, the covariates include
# hospital and year, so effects are identified from within-hospital before/after contrasts and secular trends.
# A patient-level bootstrap gives a CI for the elderly-vs-young CATE difference.

# %%
def design(d):
    X = d[["age", "severity_score", "comorbidity_score", "immune_compromised", "sex_binary",
           "infection_bloodstream", "infection_respiratory", "infection_urinary", "ward_icu", "year"]].copy()
    for h in range(1, 16):
        X[f"h{h}"] = (d.hospital_id == h).astype(int)
    return X.values.astype(float)


def gb(): return HistGradientBoostingRegressor(max_iter=150, max_depth=4, learning_rate=0.06, random_state=0)


def xlearner(X, y, T):
    mu1, mu0 = gb().fit(X[T == 1], y[T == 1]), gb().fit(X[T == 0], y[T == 0])
    d1, d0 = y[T == 1] - mu0.predict(X[T == 1]), mu1.predict(X[T == 0]) - y[T == 0]
    t1, t0 = gb().fit(X[T == 1], d1), gb().fit(X[T == 0], d0)
    e = np.clip(HistGradientBoostingClassifier(max_iter=100, random_state=0).fit(X, T).predict_proba(X)[:, 1], 0.05, 0.95)
    return (1 - e) * t1.predict(X) + e * t0.predict(X), mu1.predict(X) - mu0.predict(X)


X, y, T = design(df), df[OUTCOME].values, df.asp_active.values
cate_x, cate_t = xlearner(X, y, T)
cate_estimates = cate_x
df["cate"] = cate_x
age = df.age.values
gap = lambda c: c[age >= 65].mean() - c[age < 40].mean()
print(f"X-learner mean CATE (treated) {cate_x[T == 1].mean():.3f}, T-learner {cate_t[T == 1].mean():.3f}, "
      f"truth {truth.get('true_att_treated_days', float('nan')):.3f}")
print(f"Elderly-minus-young CATE: {gap(cate_x):.3f} (truth {truth.get('true_att_elderly', 0) - truth.get('true_att_cohort_agg_unweighted_young', 0):.3f})")

rng = np.random.default_rng(1)
gaps = []
for _ in range(30):
    i = rng.integers(0, len(df), len(df))
    c, _ = xlearner(X[i], y[i], T[i])
    pred_all = c  # CATE evaluated on bootstrap sample
    a = age[i]
    gaps.append(pred_all[a >= 65].mean() - pred_all[a < 40].mean())
cate_elderly_minus_young_ci = [float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))]
print("Bootstrap 95% CI for elderly-minus-young:", np.round(cate_elderly_minus_young_ci, 3))

# %% [markdown]
# ## 4. Mixed-effects model (same outcome: `antibiotic_duration`)

# %%
F = (f"{OUTCOME} ~ asp_active + age + severity_score + comorbidity_score + immune_compromised "
     "+ C(ward) + C(infection_type)")
naive = smf.ols(F, df).fit()
reml = smf.mixedlm(F, df, groups=df.hospital_id).fit(reml=True)
ml = smf.mixedlm(F, df, groups=df.hospital_id).fit(reml=False)
random_effect_variance = float(reml.cov_re.iloc[0, 0])
icc_value = random_effect_variance / (random_effect_variance + float(reml.scale))
mixed_model_aic, naive_ols_aic = float(ml.aic), float(naive.aic)
mixed_asp_coeff = float(reml.params["asp_active"])
print(f"ICC={icc_value:.3f}  RE var={random_effect_variance:.3f}  AIC mixed {mixed_model_aic:.1f} vs OLS {naive_ols_aic:.1f}"
      f"  ASP coef {mixed_asp_coeff:.3f}")

# %% [markdown]
# ## 5. Save results (JSON, no pickle)

# %%
print("saved:", save_results(GOLDEN_RESULTS, globals()))
# Values the figures need that are not part of the required outputs
json.dump({"did_att_ci": [float(ci[0]), float(ci[1])]}, open(RESULTS_DIR / "golden_extras.json", "w"))
