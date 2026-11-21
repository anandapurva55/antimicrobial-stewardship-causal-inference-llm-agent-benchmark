import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test
from sklearn.ensemble import GradientBoostingRegressor

# ---------------------------------------------------------
# 1. SURVIVAL ANALYSIS
# ---------------------------------------------------------
kmf = KaplanMeierFitter()
kmf.fit(
    durations=df["time_to_improvement"],
    event_observed=df["event_observed"],
)

cox_df = df[
    [
        "time_to_improvement",
        "event_observed",
        "asp_active",
        "hospital_id",
    ]
].copy()

cph = CoxPHFitter()
cph.fit(
    cox_df,
    duration_col="time_to_improvement",
    event_col="event_observed",
    cluster_col="hospital_id",
)

cox_hr_asp = float(cph.hazard_ratios_["asp_active"])
cox_se_asp = float(cph.standard_errors_["asp_active"])

ph_test = proportional_hazard_test(cph, cox_df, time_transform="rank")
ph_test_pvalues = {}
if hasattr(ph_test, "summary") and ph_test.summary is not None:
    p_col = (
        "p"
        if "p" in ph_test.summary.columns
        else [c for c in ph_test.summary.columns if "p" in c.lower()][0]
    )
    for cov, p_val in ph_test.summary[p_col].items():
        ph_test_pvalues[str(cov)] = float(p_val)
elif hasattr(ph_test, "p_value"):
    if isinstance(ph_test.p_value, (list, np.ndarray, pd.Series)):
        for name, p_val in zip(cph.params_.index, ph_test.p_value):
            ph_test_pvalues[str(name)] = float(p_val)
    else:
        ph_test_pvalues["asp_active"] = float(ph_test.p_value)

# ---------------------------------------------------------
# 2. CAUSAL IMPACT (DiD)
# ---------------------------------------------------------
did_panel = (
    df.groupby(["hospital_id", "year"], as_index=False)
    .agg(
        antibiotic_duration=("antibiotic_duration", "mean"),
        asp_active=("asp_active", "first"),
        asp_adoption_year=("asp_adoption_year", "first"),
    )
    .sort_values(["hospital_id", "year"])
    .reset_index(drop=True)
)

twfe_model = smf.ols(
    "antibiotic_duration ~ asp_active + C(hospital_id) + C(year)",
    data=did_panel,
).fit()
twfe_att_estimate = float(twfe_model.params["asp_active"])

treated_mask = did_panel["asp_adoption_year"] != 9999
rel_years = (
    did_panel.loc[treated_mask, "year"]
    - did_panel.loc[treated_mask, "asp_adoption_year"]
)
distinct_event_times = sorted([int(e) for e in rel_years.unique() if e != -1])

event_cols = []
for e in distinct_event_times:
    col_name = f"event_time_{e}".replace("-", "neg")
    did_panel[col_name] = (
        treated_mask
        & (
            did_panel["year"] - did_panel["asp_adoption_year"]
            == e
        )
    ).astype(int)
    event_cols.append((e, col_name))

es_formula = (
    "antibiotic_duration ~ "
    + " + ".join([c for _, c in event_cols])
    + " + C(hospital_id) + C(year)"
)
es_model = smf.ols(es_formula, data=did_panel).fit()
es_ci = es_model.conf_int()

es_records = []
for e, col_name in event_cols:
    es_records.append(
        {
            "event_time": e,
            "coef": float(es_model.params[col_name]),
            "ci_low": float(es_ci.loc[col_name, 0]),
            "ci_high": float(es_ci.loc[col_name, 1]),
        }
    )
event_study = pd.DataFrame(es_records).sort_values("event_time").reset_index(drop=True)

lead_col = "event_time_neg2"
if lead_col in es_model.params:
    pre_trend_coeff = float(es_model.params[lead_col])
    pre_trend_pvalue = float(es_model.pvalues[lead_col])
    pre_trend_ci = [
        float(es_ci.loc[lead_col, 0]),
        float(es_ci.loc[lead_col, 1]),
    ]
else:
    pre_leads = [
        (e, col)
        for e, col in event_cols
        if e < -1
    ]
    target_e, target_col = pre_leads[-1]
    pre_trend_coeff = float(es_model.params[target_col])
    pre_trend_pvalue = float(es_model.pvalues[target_col])
    pre_trend_ci = [
        float(es_ci.loc[target_col, 0]),
        float(es_ci.loc[target_col, 1]),
    ]

# Robust DiD estimator (Callaway-Sant'Anna 2x2 aggregation)
cs_atts = []
for cohort in [2020, 2021]:
    c_df = did_panel[did_panel["asp_adoption_year"] == cohort]
    nt_df = did_panel[did_panel["asp_adoption_year"] == 9999]
    base_year = cohort - 1
    y_c_base = c_df[c_df["year"] == base_year]["antibiotic_duration"].mean()
    y_nt_base = nt_df[nt_df["year"] == base_year]["antibiotic_duration"].mean()
    for t in sorted(did_panel["year"].unique()):
        if t >= cohort:
            y_c_t = c_df[c_df["year"] == t]["antibiotic_duration"].mean()
            y_nt_t = nt_df[nt_df["year"] == t]["antibiotic_duration"].mean()
            att_gt = (y_c_t - y_c_base) - (y_nt_t - y_nt_base)
            cs_atts.append(att_gt)

did_att_estimate = float(np.mean(cs_atts))

# ---------------------------------------------------------
# 3. TREATMENT HETEROGENEITY (T-Learner)
# ---------------------------------------------------------
feature_cols = [
    "age",
    "sex",
    "comorbidity_score",
    "immune_compromised",
    "infection_type",
    "severity_score",
    "hospital_size",
    "region",
    "ward",
    "hospital_id",
    "year",
]
X_raw = df[feature_cols].copy()
X_raw["hospital_id"] = X_raw["hospital_id"].astype(str)
X_raw["year"] = X_raw["year"].astype(str)
X = pd.get_dummies(X_raw, drop_first=True)

y = df["antibiotic_duration"].values
w = df["asp_active"].values

m0 = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
m1 = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)

m0.fit(X[w == 0], y[w == 0])
m1.fit(X[w == 1], y[w == 1])

cate_estimates = m1.predict(X) - m0.predict(X)

elderly_mask = (df["age"] >= 65).values
young_mask = (df["age"] < 40).values

rng = np.random.default_rng(42)
n_rows = len(df)
boot_diffs = []
for _ in range(1000):
    b_idx = rng.choice(n_rows, size=n_rows, replace=True)
    b_eld_mean = cate_estimates[b_idx][elderly_mask[b_idx]].mean()
    b_yng_mean = cate_estimates[b_idx][young_mask[b_idx]].mean()
    boot_diffs.append(b_eld_mean - b_yng_mean)

cate_elderly_minus_young_ci = [
    float(np.percentile(boot_diffs, 2.5)),
    float(np.percentile(boot_diffs, 97.5)),
]

# ---------------------------------------------------------
# 4. HIERARCHICAL MODELING
# ---------------------------------------------------------
mixed_mod = smf.mixedlm(
    "antibiotic_duration ~ asp_active",
    data=df,
    groups=df["hospital_id"],
)
mixed_res = mixed_mod.fit(reml=False)

random_effect_variance = float(mixed_res.cov_re.iloc[0, 0])
residual_variance = float(mixed_res.scale)
icc_value = float(
    random_effect_variance
    / (random_effect_variance + residual_variance)
)
mixed_asp_coeff = float(mixed_res.params["asp_active"])

try:
    mixed_model_aic = float(mixed_res.aic)
except (AttributeError, TypeError):
    k_params = len(mixed_res.params) + 1
    mixed_model_aic = float(-2.0 * mixed_res.llf + 2.0 * k_params)

naive_ols = smf.ols("antibiotic_duration ~ asp_active", data=df).fit()
naive_ols_aic = float(naive_ols.aic)