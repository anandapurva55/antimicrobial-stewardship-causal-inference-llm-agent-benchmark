import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, proportional_hazard_test
from linearmodels.panel import PanelOLS
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ==============================================================================
# 1. SURVIVAL ANALYSIS: KAPLAN-MEIER, COX PH (CLUSTERED BY HOSPITAL), PH TEST
# ==============================================================================

kmf_treated = KaplanMeierFitter()
kmf_control = KaplanMeierFitter()

mask_asp_1 = df["asp_active"] == 1
mask_asp_0 = df["asp_active"] == 0

kmf_treated.fit(
    df.loc[mask_asp_1, "time_to_improvement"],
    event_observed=df.loc[mask_asp_1, "event_observed"],
    label="ASP Active",
)
kmf_control.fit(
    df.loc[mask_asp_0, "time_to_improvement"],
    event_observed=df.loc[mask_asp_0, "event_observed"],
    label="No ASP",
)

lr_res = logrank_test(
    df.loc[mask_asp_1, "time_to_improvement"],
    df.loc[mask_asp_0, "time_to_improvement"],
    event_observed_A=df.loc[mask_asp_1, "event_observed"],
    event_observed_B=df.loc[mask_asp_0, "event_observed"],
)

cox_num_cols = [
    "asp_active",
    "age",
    "comorbidity_score",
    "immune_compromised",
    "severity_score",
]
cox_cat_df = pd.get_dummies(
    df[["sex", "ward", "infection_type", "hospital_size", "region", "year"]].astype(str),
    drop_first=True,
    dtype=float,
)
cox_X_raw = pd.concat([df[cox_num_cols].astype(float), cox_cat_df], axis=1)

# Drop any rank-deficient covariate columns while preserving asp_active
kept_cols = []
current_mat = None
current_rank = 0
for col in cox_X_raw.columns:
    col_vec = cox_X_raw[col].values[:, None]
    cand_mat = col_vec if current_mat is None else np.hstack([current_mat, col_vec])
    cand_rank = np.linalg.matrix_rank(cand_mat)
    if cand_rank > current_rank:
        kept_cols.append(col)
        current_mat = cand_mat
        current_rank = cand_rank

cox_df = cox_X_raw[kept_cols].copy()
cox_df["time_to_improvement"] = df["time_to_improvement"].astype(float).values
cox_df["event_observed"] = df["event_observed"].astype(int).values
cox_df["hospital_id"] = df["hospital_id"].values

cph = CoxPHFitter()
cph.fit(
    cox_df,
    duration_col="time_to_improvement",
    event_col="event_observed",
    cluster_col="hospital_id",
    robust=True,
)

cox_hr_asp = float(cph.summary.loc["asp_active", "exp(coef)"])
cox_se_asp = float(cph.summary.loc["asp_active", "se(coef)"])

ph_test_res = proportional_hazard_test(cph, cox_df, time_transform="rank")
ph_test_pvalues = {str(k): float(v) for k, v in ph_test_res.summary["p"].items()}


# ==============================================================================
# 2. CAUSAL IMPACT (DiD): PANEL, TWFE, STAGGERED-ROBUST ATT, EVENT STUDY
# ==============================================================================

did_panel = (
    df.groupby(["hospital_id", "year"], as_index=False)
    .agg(
        antibiotic_duration=("antibiotic_duration", "mean"),
        asp_active=("asp_active", "mean"),
        asp_adoption_year=("asp_adoption_year", "first"),
        n_patients=("patient_id", "count"),
    )
    .sort_values(["hospital_id", "year"])
    .reset_index(drop=True)
)

panel_indexed = did_panel.set_index(["hospital_id", "year"])
twfe_model = PanelOLS(
    panel_indexed["antibiotic_duration"],
    panel_indexed[["asp_active"]],
    entity_effects=True,
    time_effects=True,
)
twfe_res = twfe_model.fit(cov_type="clustered", cluster_entity=True)
twfe_att_estimate = float(twfe_res.params["asp_active"])

# Callaway-Sant'Anna / Sun-Abraham estimator with never-treated controls
Y_wide = did_panel.pivot(index="hospital_id", columns="year", values="antibiotic_duration")
hosp_cohort = did_panel.groupby("hospital_id")["asp_adoption_year"].first()

all_years = sorted(did_panel["year"].unique().tolist())
max_year = max(all_years)
treated_cohorts = sorted([int(g) for g in hosp_cohort.unique() if g <= max_year])
never_treated_ids = hosp_cohort[hosp_cohort > max_year].index.tolist()
cohort_ids = {g: hosp_cohort[hosp_cohort == g].index.tolist() for g in treated_cohorts}

n_hospitals = len(hosp_cohort)
did_dof = n_hospitals - 1
did_t_crit = float(stats.t.ppf(0.975, df=did_dof))


def cs_linear_combination(weights_dict):
    """Exact Callaway-Sant'Anna / Sun-Abraham linear combination with never-treated controls."""
    est = 0.0
    var_total = 0.0
    ctrl_vals = np.zeros(len(never_treated_ids), dtype=float)
    for g in treated_cohorts:
        g_items = [(t, w) for (gg, t), w in weights_dict.items() if gg == g and w != 0.0]
        if not g_items:
            continue
        g_ids = cohort_ids[g]
        g_vals = np.zeros(len(g_ids), dtype=float)
        for t, w in g_items:
            g_vals += w * (Y_wide.loc[g_ids, t].values - Y_wide.loc[g_ids, g - 1].values)
            ctrl_vals += w * (
                Y_wide.loc[never_treated_ids, t].values
                - Y_wide.loc[never_treated_ids, g - 1].values
            )
        est += float(np.mean(g_vals))
        if len(g_vals) > 1:
            var_total += float(np.var(g_vals, ddof=1) / len(g_vals))
    est -= float(np.mean(ctrl_vals))
    if len(ctrl_vals) > 1:
        var_total += float(np.var(ctrl_vals, ddof=1) / len(ctrl_vals))
    se = float(np.sqrt(max(var_total, 0.0)))
    t_stat = est / se if se > 0 else 0.0
    pval = float(2.0 * stats.t.sf(np.abs(t_stat), df=did_dof))
    ci_low = float(est - did_t_crit * se)
    ci_high = float(est + did_t_crit * se)
    return float(est), se, pval, ci_low, ci_high


# Overall staggered-robust ATT weighted across post-treatment hospital-year cells
post_cells = [
    (g, t, len(cohort_ids[g]))
    for g in treated_cohorts
    for t in all_years
    if t >= g
]
total_post_weight = sum(n_g for _, _, n_g in post_cells)
overall_att_weights = {(g, t): n_g / total_post_weight for g, t, n_g in post_cells}
did_att_estimate, did_att_se, did_att_pval, did_att_ci_low, did_att_ci_high = cs_linear_combination(
    overall_att_weights
)

# Dynamic Event Study (reference period event_time = -1 omitted)
all_event_times = sorted(
    {int(t - g) for g in treated_cohorts for t in all_years if int(t - g) != -1}
)
event_study_rows = []
event_study_pvals = {}

for e in all_event_times:
    active_g = [g for g in treated_cohorts if (g + e) in all_years]
    denom = sum(len(cohort_ids[g]) for g in active_g)
    e_weights = {(g, g + e): len(cohort_ids[g]) / denom for g in active_g}
    coef_e, se_e, pval_e, ci_lo_e, ci_hi_e = cs_linear_combination(e_weights)
    event_study_rows.append(
        {
            "event_time": int(e),
            "coef": float(coef_e),
            "ci_low": float(ci_lo_e),
            "ci_high": float(ci_hi_e),
        }
    )
    event_study_pvals[int(e)] = float(pval_e)

event_study = pd.DataFrame(event_study_rows, columns=["event_time", "coef", "ci_low", "ci_high"])

# Pre-trend test using primary pre-treatment lead (event_time = -2)
pre_lead_row = event_study.loc[event_study["event_time"] == -2].iloc[0]
pre_trend_coeff = float(pre_lead_row["coef"])
pre_trend_pvalue = float(event_study_pvals[-2])
pre_trend_ci = [float(pre_lead_row["ci_low"]), float(pre_lead_row["ci_high"])]


# ==============================================================================
# 3. TREATMENT HETEROGENEITY: HOSPITAL/YEAR-ADJUSTED X-LEARNER & BOOTSTRAP CI
# ==============================================================================

fe_dummies = pd.get_dummies(
    df[["hospital_id", "year"]].astype(str), drop_first=True, dtype=float
).values
n_fe_cols = fe_dummies.shape[1]

pat_num = df[["age", "comorbidity_score", "immune_compromised", "severity_score"]].astype(float)
pat_cat = pd.get_dummies(
    df[["sex", "ward", "infection_type"]].astype(str), drop_first=True, dtype=float
)
X_pat = pd.concat([pat_num, pat_cat], axis=1).values.astype(float)
T_vec = df["asp_active"].values.astype(int)
Y_vec = df["antibiotic_duration"].values.astype(float)
age_vec = df["age"].values.astype(float)


def fit_predict_xlearner(X_tr, T_tr, Y_tr, fe_tr, X_pred, seed=42):
    """Hospital-and-year fixed-effect adjusted X-Learner with gradient boosting."""
    # Remove additive hospital and year fixed effects without absorbing heterogeneous treatment effects
    Z = np.column_stack([np.ones(len(Y_tr)), fe_tr, T_tr, X_tr, T_tr[:, None] * X_tr])
    beta, _, _, _ = np.linalg.lstsq(Z, Y_tr, rcond=None)
    fe_shift = fe_tr @ beta[1 : 1 + fe_tr.shape[1]]
    Y_adj = Y_tr - (fe_shift - np.mean(fe_shift))

    mask1 = T_tr == 1
    mask0 = T_tr == 0

    gbr_params = dict(
        max_iter=60,
        max_depth=3,
        learning_rate=0.07,
        min_samples_leaf=30,
        random_state=seed,
    )

    mu0 = HistGradientBoostingRegressor(**gbr_params)
    mu1 = HistGradientBoostingRegressor(**gbr_params)
    mu0.fit(X_tr[mask0], Y_adj[mask0])
    mu1.fit(X_tr[mask1], Y_adj[mask1])

    d1 = Y_adj[mask1] - mu0.predict(X_tr[mask1])
    d0 = mu1.predict(X_tr[mask0]) - Y_adj[mask0]

    tau1 = HistGradientBoostingRegressor(**gbr_params)
    tau0 = HistGradientBoostingRegressor(**gbr_params)
    tau1.fit(X_tr[mask1], d1)
    tau0.fit(X_tr[mask0], d0)

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_pred_scaled = scaler.transform(X_pred)
    ps_model = LogisticRegression(max_iter=500, random_state=seed)
    ps_model.fit(X_tr_scaled, T_tr)
    e_hat = np.clip(ps_model.predict_proba(X_pred_scaled)[:, 1], 0.05, 0.95)

    cate_hat = e_hat * tau0.predict(X_pred) + (1.0 - e_hat) * tau1.predict(X_pred)
    return cate_hat.astype(float)


cate_estimates = fit_predict_xlearner(X_pat, T_vec, Y_vec, fe_dummies, X_pat, seed=42)

# Stratified bootstrap (by hospital-year cell) for elderly (age>=65) minus young (age<40) CATE
rng = np.random.default_rng(42)
cell_groups = [idx.values for idx in df.groupby(["hospital_id", "year"]).groups.values()]
n_boot = 60
boot_diffs = np.empty(n_boot, dtype=float)

for b in range(n_boot):
    boot_idx = np.concatenate(
        [rng.choice(grp, size=len(grp), replace=True) for grp in cell_groups]
    )
    cate_b = fit_predict_xlearner(
        X_pat[boot_idx],
        T_vec[boot_idx],
        Y_vec[boot_idx],
        fe_dummies[boot_idx],
        X_pat[boot_idx],
        seed=100 + b,
    )
    age_b = age_vec[boot_idx]
    boot_diffs[b] = float(np.mean(cate_b[age_b >= 65]) - np.mean(cate_b[age_b < 40]))

ci_lo_cate, ci_hi_cate = np.percentile(boot_diffs, [2.5, 97.5])
cate_elderly_minus_young_ci = [float(ci_lo_cate), float(ci_hi_cate)]


# ==============================================================================
# 4. HIERARCHICAL MODELING: MIXED-EFFECTS RANDOM INTERCEPT, ICC, AIC VS OLS
# ==============================================================================

mixed_formula = (
    "antibiotic_duration ~ asp_active + C(year) + age + C(sex) "
    "+ comorbidity_score + immune_compromised + severity_score "
    "+ C(ward) + C(infection_type)"
)

mixed_model = smf.mixedlm(mixed_formula, data=df, groups=df["hospital_id"])
mixed_res = mixed_model.fit(reml=False, method=["lbfgs", "bfgs", "powell"])

ols_res = smf.ols(mixed_formula, data=df).fit()

random_effect_variance = float(mixed_res.cov_re.iloc[0, 0])
residual_variance = float(mixed_res.scale)
icc_value = float(random_effect_variance / (random_effect_variance + residual_variance))
mixed_asp_coeff = float(mixed_res.fe_params["asp_active"])
mixed_model_aic = (
    float(mixed_res.aic)
    if not np.isnan(mixed_res.aic)
    else float(-2.0 * mixed_res.llf + 2.0 * (len(mixed_res.fe_params) + 2))
)
naive_ols_aic = float(ols_res.aic)


# ==============================================================================
# SUMMARY PRINT
# ==============================================================================

print("=== 1. Survival Analysis ===")
print(f"KM Median Time-to-Improvement (ASP=1): {kmf_treated.median_survival_time_:.2f}")
print(f"KM Median Time-to-Improvement (ASP=0): {kmf_control.median_survival_time_:.2f}")
print(f"Log-rank p-value: {lr_res.p_value:.4g}")
print(f"Cox HR (asp_active): {cox_hr_asp:.4f} (clustered log-HR SE: {cox_se_asp:.4f})")
print(f"PH test p-value (asp_active): {ph_test_pvalues.get('asp_active', np.nan):.4f}")

print("\n=== 2. Causal Impact (DiD) ===")
print(f"Panel shape: {did_panel.shape}")
print(f"Pre-trend lead (e=-2): {pre_trend_coeff:.4f}, p={pre_trend_pvalue:.4f}, 95% CI={pre_trend_ci}")
print(f"TWFE ATT: {twfe_att_estimate:.4f} days")
print(f"Staggered-robust (CS/SA) ATT: {did_att_estimate:.4f} days (95% CI: [{did_att_ci_low:.4f}, {did_att_ci_high:.4f}])")
print(event_study.to_string(index=False))

print("\n=== 3. Treatment Heterogeneity (CATE) ===")
elderly_mean_cate = float(np.mean(cate_estimates[age_vec >= 65]))
young_mean_cate = float(np.mean(cate_estimates[age_vec < 40]))
print(f"Overall mean CATE: {np.mean(cate_estimates):.4f} days")
print(f"Elderly (65+) mean CATE: {elderly_mean_cate:.4f}; Young (<40) mean CATE: {young_mean_cate:.4f}")
print(f"Elderly minus Young diff: {elderly_mean_cate - young_mean_cate:.4f}, 95% Bootstrap CI: {cate_elderly_minus_young_ci}")

print("\n=== 4. Hierarchical Modeling ===")
print(f"Mixed Model ASP coeff: {mixed_asp_coeff:.4f} days")
print(f"Hospital random-intercept variance: {random_effect_variance:.4f}")
print(f"Residual variance: {residual_variance:.4f}")
print(f"ICC: {icc_value:.4f}")
print(f"Mixed Model AIC (ML): {mixed_model_aic:.2f} vs Naive OLS AIC: {naive_ols_aic:.2f}")