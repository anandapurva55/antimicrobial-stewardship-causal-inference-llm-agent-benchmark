from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from sklearn.ensemble import GradientBoostingRegressor
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 1. SURVIVAL ANALYSIS
# ---------------------------------------------------------
kmf = KaplanMeierFitter()
kmf.fit(df["time_to_improvement"], event_observed=df["event_observed"])

cox_df = df[["time_to_improvement", "event_observed", "asp_active", "hospital_id"]].copy()
cph = CoxPHFitter()
cph.fit(
    cox_df,
    duration_col="time_to_improvement",
    event_col="event_observed",
    cluster_col="hospital_id"
)

cox_hr_asp = float(cph.hazard_ratios_["asp_active"])
cox_se_asp = float(cph.standard_errors_["asp_active"])

ph_test = proportional_hazard_test(cph, cox_df, time_transform="rank")
ph_test_pvalues = {str(k): float(v) for k, v in ph_test.summary["p"].items()}

# ---------------------------------------------------------
# 2. CAUSAL IMPACT (DiD)
# ---------------------------------------------------------
did_panel = df.groupby(["hospital_id", "year"]).agg(
    antibiotic_duration=("antibiotic_duration", "mean"),
    asp_adoption_year=("asp_adoption_year", "first"),
    asp_active=("asp_active", "first")
).reset_index()

twfe_mod = smf.ols(
    "antibiotic_duration ~ asp_active + C(hospital_id) + C(year)",
    data=did_panel
).fit()
twfe_att_estimate = float(twfe_mod.params["asp_active"])

# Event Study (TWFE leads and lags, omitting event_time = -1)
event_times = [-4, -3, -2, 0, 1, 2]
for e in event_times:
    col = f"lead_{abs(e)}" if e < 0 else f"lag_{e}"
    did_panel[col] = (
        (did_panel["asp_adoption_year"] < 9999) &
        (did_panel["year"] - did_panel["asp_adoption_year"] == e)
    ).astype(int)

formula_es = "antibiotic_duration ~ lead_4 + lead_3 + lead_2 + lag_0 + lag_1 + lag_2 + C(hospital_id) + C(year)"
es_fit = smf.ols(formula_es, data=did_panel).fit()
ci_es = es_fit.conf_int()

es_rows = []
for e in event_times:
    col = f"lead_{abs(e)}" if e < 0 else f"lag_{e}"
    es_rows.append({
        "event_time": e,
        "coef": float(es_fit.params[col]),
        "ci_low": float(ci_es.loc[col, 0]),
        "ci_high": float(ci_es.loc[col, 1])
    })
event_study = pd.DataFrame(es_rows)

pre_trend_coeff = float(es_fit.params["lead_2"])
pre_trend_pvalue = float(es_fit.pvalues["lead_2"])
pre_trend_ci = [float(ci_es.loc["lead_2", 0]), float(ci_es.loc["lead_2", 1])]

# Robust estimator: Stacked DiD using never-treated hospitals as controls
stacks = []
for g in [2020, 2021]:
    sub = did_panel[did_panel["asp_adoption_year"].isin([g, 9999])].copy()
    sub["stack"] = g
    sub["treat"] = (sub["asp_adoption_year"] == g).astype(int)
    sub["post"] = (sub["year"] >= g).astype(int)
    sub["treat_post"] = sub["treat"] * sub["post"]
    sub["hosp_stack"] = sub["hospital_id"].astype(str) + f"_{g}"
    sub["year_stack"] = sub["year"].astype(str) + f"_{g}"
    stacks.append(sub)

stacked_panel = pd.concat(stacks, ignore_index=True)
stacked_fit = smf.ols(
    "antibiotic_duration ~ treat_post + C(hosp_stack) + C(year_stack)",
    data=stacked_panel
).fit(cov_type="cluster", cov_kwds={"groups": stacked_panel["hospital_id"]})
did_att_estimate = float(stacked_fit.params["treat_post"])

# ---------------------------------------------------------
# 3. TREATMENT HETEROGENEITY (CATE)
# ---------------------------------------------------------
feature_cols = [
    "age", "sex", "comorbidity_score", "immune_compromised",
    "infection_type", "severity_score", "hospital_id", "year",
    "ward", "hospital_size"
]
X = pd.get_dummies(df[feature_cols], drop_first=True)
y = df["antibiotic_duration"].values
w = df["asp_active"].values

m0 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
m1 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)

m0.fit(X[w == 0], y[w == 0])
m1.fit(X[w == 1], y[w == 1])

cate_estimates = m1.predict(X) - m0.predict(X)

elderly_mask = (df["age"] >= 65).values
young_mask = (df["age"] < 40).values

np.random.seed(42)
n_boot = 1000
boot_diffs = np.empty(n_boot)
n_patients = len(df)

for b in range(n_boot):
    idx = np.random.choice(n_patients, size=n_patients, replace=True)
    boot_diffs[b] = np.mean(cate_estimates[idx][elderly_mask[idx]]) - np.mean(cate_estimates[idx][young_mask[idx]])

cate_elderly_minus_young_ci = [
    float(np.percentile(boot_diffs, 2.5)),
    float(np.percentile(boot_diffs, 97.5))
]

# ---------------------------------------------------------
# 4. HIERARCHICAL MODELING
# ---------------------------------------------------------
mixed_md = smf.mixedlm("antibiotic_duration ~ asp_active", data=df, groups=df["hospital_id"])
mixed_fit = mixed_md.fit(reml=False)

naive_fit = smf.ols("antibiotic_duration ~ asp_active", data=df).fit()

mixed_asp_coeff = float(mixed_fit.params["asp_active"])
mixed_model_aic = float(mixed_fit.aic)
naive_ols_aic = float(naive_fit.aic)

random_effect_variance = float(mixed_fit.cov_re.iloc[0, 0])
residual_variance = float(mixed_fit.scale)
icc_value = float(random_effect_variance / (random_effect_variance + residual_variance))