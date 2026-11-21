import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import proportional_hazard_test, logrank_test
from linearmodels.panel import PanelOLS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    _USE_HGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    _USE_HGB = False


# =============================================================================
# 1. SURVIVAL ANALYSIS: Kaplan-Meier & Stratified Cox PH (clustered by hospital)
# =============================================================================
kmf_treated = KaplanMeierFitter()
kmf_control = KaplanMeierFitter()

mask_treated = df["asp_active"] == 1
mask_control = df["asp_active"] == 0

kmf_treated.fit(
    df.loc[mask_treated, "time_to_improvement"],
    event_observed=df.loc[mask_treated, "event_observed"],
    label="ASP Active",
)
kmf_control.fit(
    df.loc[mask_control, "time_to_improvement"],
    event_observed=df.loc[mask_control, "event_observed"],
    label="No ASP",
)

lr_res = logrank_test(
    df.loc[mask_treated, "time_to_improvement"],
    df.loc[mask_control, "time_to_improvement"],
    event_observed_A=df.loc[mask_treated, "event_observed"],
    event_observed_B=df.loc[mask_control, "event_observed"],
)

cox_num_cols = [
    "time_to_improvement",
    "event_observed",
    "asp_active",
    "age",
    "comorbidity_score",
    "immune_compromised",
    "severity_score",
]
cox_cat_dummies = pd.get_dummies(
    df[["sex", "ward", "infection_type", "year"]].astype(
        {"sex": str, "ward": str, "infection_type": str, "year": str}
    ),
    drop_first=True,
    dtype=float,
)
cox_df = pd.concat([df[cox_num_cols].astype(float), cox_cat_dummies], axis=1)
cox_df["hospital_id"] = df["hospital_id"].values
cox_df["hospital_cluster"] = df["hospital_id"].values

cph = CoxPHFitter()
cph.fit(
    cox_df,
    duration_col="time_to_improvement",
    event_col="event_observed",
    strata=["hospital_id"],
    cluster_col="hospital_cluster",
    robust=True,
)

cox_hr_asp = float(np.exp(cph.params_["asp_active"]))
cox_se_asp = float(cph.standard_errors_["asp_active"])

try:
    ph_res = proportional_hazard_test(cph, cox_df, time_transform="rank")
except Exception:
    ph_res = proportional_hazard_test(
        cph, cox_df.drop(columns=["hospital_cluster"]), time_transform="rank"
    )

ph_test_pvalues = {
    str(cov): float(pval) for cov, pval in ph_res.summary["p"].items()
}

print("=== 1. SURVIVAL ANALYSIS ===")
print(f"KM median time-to-improvement (ASP=1): {kmf_treated.median_survival_time_:.2f}")
print(f"KM median time-to-improvement (ASP=0): {kmf_control.median_survival_time_:.2f}")
print(f"Log-rank p-value: {lr_res.p_value:.4g}")
print(f"Cox PH HR (asp_active): {cox_hr_asp:.4f} (log-HR SE clustered: {cox_se_asp:.4f})")
print(f"PH test p-value (asp_active): {ph_test_pvalues.get('asp_active', np.nan):.4f}")


# =============================================================================
# 2. CAUSAL IMPACT (DiD): Panel, TWFE, Sun-Abraham / Callaway-Sant'Anna, Event Study
# =============================================================================
did_panel = (
    df.groupby(["hospital_id", "year"], as_index=False)
    .agg(
        antibiotic_duration=("antibiotic_duration", "mean"),
        asp_active=("asp_active", "mean"),
        asp_adoption_year=("asp_adoption_year", "first"),
        n_patients=("patient_id", "count"),
        mean_age=("age", "mean"),
        mean_severity=("severity_score", "mean"),
        mean_comorbidity=("comorbidity_score", "mean"),
    )
    .sort_values(["hospital_id", "year"])
    .reset_index(drop=True)
)

# Plain TWFE on the hospital-year panel
did_panel_idx = did_panel.set_index(["hospital_id", "year"])
twfe_mod = PanelOLS(
    dependent=did_panel_idx["antibiotic_duration"],
    exog=did_panel_idx[["asp_active"]],
    entity_effects=True,
    time_effects=True,
)
twfe_res = twfe_mod.fit(cov_type="clustered", cluster_entity=True)
twfe_att_estimate = float(twfe_res.params["asp_active"])

# Staggered-adoption-robust estimator: Sun-Abraham (2021) / Callaway-Sant'Anna (2021)
# With balanced panel, no time-varying covariates, and never-treated controls,
# saturated cohort-by-event-time TWFE coefficients equal exact CS ATT(g, t) with base g-1.
max_year = int(did_panel["year"].max())
treated_cohorts = sorted(
    did_panel.loc[did_panel["asp_adoption_year"] <= max_year, "asp_adoption_year"]
    .unique()
    .astype(int)
)

sa_df = did_panel.copy()
sa_df["event_time"] = np.where(
    sa_df["asp_adoption_year"] <= max_year,
    sa_df["year"] - sa_df["asp_adoption_year"],
    np.nan,
)

inter_specs = []
inter_cols_dict = {}
for g in treated_cohorts:
    cohort_event_times = sorted(
        sa_df.loc[sa_df["asp_adoption_year"] == g, "event_time"].dropna().unique().astype(int)
    )
    for e in cohort_event_times:
        if e == -1:
            continue
        col_name = f"g{g}_e{e}"
        inter_cols_dict[col_name] = (
            (sa_df["asp_adoption_year"] == g) & (sa_df["event_time"] == e)
        ).astype(float)
        inter_specs.append((g, e, col_name))

hosp_dummies = pd.get_dummies(
    sa_df["hospital_id"].astype(str), prefix="hosp", drop_first=True, dtype=float
)
year_dummies = pd.get_dummies(
    sa_df["year"].astype(str), prefix="yr", drop_first=True, dtype=float
)
inter_df = pd.DataFrame(inter_cols_dict, index=sa_df.index)

X_sa = pd.concat(
    [pd.Series(1.0, index=sa_df.index, name="const"), hosp_dummies, year_dummies, inter_df],
    axis=1,
)
y_sa = sa_df["antibiotic_duration"].astype(float)

sa_res = sm.OLS(y_sa, X_sa).fit(
    cov_type="cluster",
    cov_kwds={"groups": sa_df["hospital_id"], "use_correction": True},
)

param_index = list(sa_res.params.index)
params_vec = sa_res.params.values
cov_mat = sa_res.cov_params().values

cohort_hospital_counts = (
    sa_df.groupby("asp_adoption_year")["hospital_id"].nunique().to_dict()
)
n_clusters = int(sa_df["hospital_id"].nunique())
df_t = n_clusters - 1
t_crit = float(stats.t.ppf(0.975, df=df_t))

all_event_times = sorted({e for _, e, _ in inter_specs})
es_rows = []
es_details = {}

for e in all_event_times:
    matching = [(g, col) for g, e_val, col in inter_specs if e_val == e]
    total_w = sum(cohort_hospital_counts[g] for g, _ in matching)
    w_vec = np.zeros(len(param_index), dtype=float)
    for g, col in matching:
        w_vec[param_index.index(col)] = cohort_hospital_counts[g] / total_w

    coef = float(w_vec @ params_vec)
    se = float(np.sqrt(max(w_vec @ cov_mat @ w_vec, 0.0)))
    ci_low = float(coef - t_crit * se)
    ci_high = float(coef + t_crit * se)
    pval = float(2.0 * stats.t.sf(np.abs(coef / se), df=df_t)) if se > 0 else 1.0

    es_rows.append(
        {
            "event_time": int(e),
            "coef": coef,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    )
    es_details[int(e)] = {
        "coef": coef,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "pvalue": pval,
    }

event_study = pd.DataFrame(es_rows, columns=["event_time", "coef", "ci_low", "ci_high"])

# Primary pre-trend test: lead immediately preceding reference period (event_time = -2)
lead_key = -2 if -2 in es_details else max(e for e in es_details if e < 0)
pre_trend_coeff = float(es_details[lead_key]["coef"])
pre_trend_pvalue = float(es_details[lead_key]["pvalue"])
pre_trend_ci = [
    float(es_details[lead_key]["ci_low"]),
    float(es_details[lead_key]["ci_high"]),
]

# Overall staggered-robust ATT: simple weighted average of post-treatment ATT(g, t)
post_specs = [(g, col) for g, e_val, col in inter_specs if e_val >= 0]
total_post_w = sum(cohort_hospital_counts[g] for g, _ in post_specs)
w_att = np.zeros(len(param_index), dtype=float)
for g, col in post_specs:
    w_att[param_index.index(col)] = cohort_hospital_counts[g] / total_post_w

did_att_estimate = float(w_att @ params_vec)
did_att_se = float(np.sqrt(max(w_att @ cov_mat @ w_att, 0.0)))

print("\n=== 2. CAUSAL IMPACT (DiD) ===")
print(f"Panel rows: {len(did_panel)}")
print(f"Pre-trend lead (e={lead_key}): coef={pre_trend_coeff:.4f}, p={pre_trend_pvalue:.4f}, 95% CI={pre_trend_ci}")
print(f"Plain TWFE ATT: {twfe_att_estimate:.4f} days")
print(f"Staggered-robust (CS/SA) ATT: {did_att_estimate:.4f} days (SE={did_att_se:.4f})")
print(event_study.to_string(index=False))


# =============================================================================
# 3. TREATMENT HETEROGENEITY: X-Learner (Hospital & Year Adjusted) + Bootstrap CI
# =============================================================================
patient_num_cols = ["age", "comorbidity_score", "immune_compromised", "severity_score"]
patient_cat_dummies = pd.get_dummies(
    df[["sex", "ward", "infection_type"]].astype(str),
    drop_first=True,
    dtype=float,
)
X_df = pd.concat([df[patient_num_cols].astype(float), patient_cat_dummies], axis=1)
X_all = X_df.values.astype(float)

# Fixed-effects design matrix for partialling out hospital & year effects on untreated units
fe_hosp = pd.get_dummies(
    df["hospital_id"].astype(str), prefix="h", drop_first=True, dtype=float
)
fe_year = pd.get_dummies(
    df["year"].astype(str), prefix="y", drop_first=True, dtype=float
)
n_fe_cols = fe_hosp.shape[1] + fe_year.shape[1]
Z_all = np.column_stack(
    [np.ones(len(df), dtype=float), fe_hosp.values, fe_year.values, X_all]
)
fe_col_slice = slice(1, 1 + n_fe_cols)

y_all = df["antibiotic_duration"].values.astype(float)
w_all = df["asp_active"].values.astype(int)
age_all = df["age"].values.astype(float)


def _make_gbr(seed):
    if _USE_HGB:
        return HistGradientBoostingRegressor(
            max_iter=80,
            max_depth=3,
            learning_rate=0.05,
            min_samples_leaf=25,
            l2_regularization=1.0,
            random_state=seed,
        )
    return GradientBoostingRegressor(
        n_estimators=60,
        max_depth=3,
        learning_rate=0.05,
        min_samples_leaf=25,
        subsample=0.8,
        random_state=seed,
    )


def _adjust_outcome_hosp_year(Z_fit, y_fit, w_fit, Z_pred):
    untreated = w_fit == 0
    beta, _, _, _ = np.linalg.lstsq(Z_fit[untreated], y_fit[untreated], rcond=None)
    fe_pred = Z_pred[:, fe_col_slice] @ beta[fe_col_slice]
    fe_pred = fe_pred - np.mean(fe_pred)
    return fe_pred


def _fit_predict_xlearner(X_fit, w_fit, y_adj_fit, X_pred, seed):
    mask0 = w_fit == 0
    mask1 = w_fit == 1

    # Stage 1: Outcome response models
    mu0 = _make_gbr(seed)
    mu1 = _make_gbr(seed + 1)
    mu0.fit(X_fit[mask0], y_adj_fit[mask0])
    mu1.fit(X_fit[mask1], y_adj_fit[mask1])

    # Stage 2: Imputed individual treatment effects
    d1 = y_adj_fit[mask1] - mu0.predict(X_fit[mask1])
    d0 = mu1.predict(X_fit[mask0]) - y_adj_fit[mask0]

    tau1 = _make_gbr(seed + 2)
    tau0 = _make_gbr(seed + 3)
    tau1.fit(X_fit[mask1], d1)
    tau0.fit(X_fit[mask0], d0)

    # Stage 3: Propensity-weighted combination
    prop_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    prop_model.fit(X_fit, w_fit)
    e_hat = np.clip(prop_model.predict_proba(X_pred)[:, 1], 0.05, 0.95)

    cate_pred = e_hat * tau0.predict(X_pred) + (1.0 - e_hat) * tau1.predict(X_pred)
    return np.asarray(cate_pred, dtype=float)


fe_all = _adjust_outcome_hosp_year(Z_all, y_all, w_all, Z_all)
y_adj_all = y_all - fe_all
cate_estimates = _fit_predict_xlearner(X_all, w_all, y_adj_all, X_all, seed=42)

elderly_mask = age_all >= 65
young_mask = age_all < 40
cate_diff_point = float(
    cate_estimates[elderly_mask].mean() - cate_estimates[young_mask].mean()
)

# Stratified bootstrap by (hospital_id, year) preserving panel design
cell_indices = [
    grp.index.values for _, grp in df.groupby(["hospital_id", "year"], sort=True)
]
n_boot = 100 if _USE_HGB else 50
rng = np.random.default_rng(42)
boot_diffs = np.empty(n_boot, dtype=float)

for b in range(n_boot):
    boot_idx = np.concatenate(
        [rng.choice(idx, size=len(idx), replace=True) for idx in cell_indices]
    )
    Z_b = Z_all[boot_idx]
    X_b = X_all[boot_idx]
    y_b = y_all[boot_idx]
    w_b = w_all[boot_idx]
    age_b = age_all[boot_idx]

    fe_b = _adjust_outcome_hosp_year(Z_b, y_b, w_b, Z_b)
    y_adj_b = y_b - fe_b
    cate_b = _fit_predict_xlearner(X_b, w_b, y_adj_b, X_b, seed=1000 + b * 10)
    boot_diffs[b] = float(
        cate_b[age_b >= 65].mean() - cate_b[age_b < 40].mean()
    )

cate_elderly_minus_young_ci = [
    float(np.percentile(boot_diffs, 2.5)),
    float(np.percentile(boot_diffs, 97.5)),
]

print("\n=== 3. TREATMENT HETEROGENEITY (X-LEARNER) ===")
print(f"Mean CATE (overall): {cate_estimates.mean():.4f} days")
print(f"Mean CATE (age>=65): {cate_estimates[elderly_mask].mean():.4f} days")
print(f"Mean CATE (age<40):  {cate_estimates[young_mask].mean():.4f} days")
print(
    f"Elderly - Young CATE diff: {cate_diff_point:.4f}, "
    f"Bootstrap 95% CI: {cate_elderly_minus_young_ci}"
)


# =============================================================================
# 4. HIERARCHICAL MODELING: Mixed-Effects Linear Model vs Naive OLS
# =============================================================================
mixed_formula = (
    "antibiotic_duration ~ asp_active + age + C(sex) + comorbidity_score + "
    "immune_compromised + severity_score + C(ward) + C(infection_type) + C(year)"
)

mixed_mod = smf.mixedlm(mixed_formula, data=df, groups=df["hospital_id"])
mixed_res = mixed_mod.fit(reml=False, method=["lbfgs", "bfgs", "powell"])

random_effect_variance = float(np.asarray(mixed_res.cov_re).ravel()[0])
residual_variance = float(mixed_res.scale)
icc_value = float(
    random_effect_variance / (random_effect_variance + residual_variance)
)
mixed_asp_coeff = float(mixed_res.fe_params["asp_active"])
mixed_model_aic = (
    float(mixed_res.aic)
    if np.isfinite(mixed_res.aic)
    else float(-2.0 * mixed_res.llf + 2.0 * (len(mixed_res.fe_params) + 2))
)

ols_res = smf.ols(mixed_formula, data=df).fit()
naive_ols_aic = float(ols_res.aic)

print("\n=== 4. HIERARCHICAL MODELING ===")
print(f"Hospital random-intercept variance: {random_effect_variance:.4f}")
print(f"Residual variance: {residual_variance:.4f}")
print(f"ICC: {icc_value:.4f}")
print(f"Mixed model asp_active coeff: {mixed_asp_coeff:.4f}")
print(f"Mixed model AIC (ML): {mixed_model_aic:.2f} vs Naive OLS AIC: {naive_ols_aic:.2f}")