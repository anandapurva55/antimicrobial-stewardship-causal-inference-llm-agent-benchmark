import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from linearmodels.panel import PanelOLS
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

# =============================================================================
# 1. SURVIVAL ANALYSIS: Kaplan-Meier, Clustered Cox PH, Schoenfeld PH Test
# =============================================================================
kmf_asp = KaplanMeierFitter()
kmf_ctrl = KaplanMeierFitter()

mask_treated = df["asp_active"] == 1
kmf_asp.fit(
    durations=df.loc[mask_treated, "time_to_improvement"],
    event_observed=df.loc[mask_treated, "event_observed"],
    label="ASP Active",
)
kmf_ctrl.fit(
    durations=df.loc[~mask_treated, "time_to_improvement"],
    event_observed=df.loc[~mask_treated, "event_observed"],
    label="No ASP",
)

cox_covariate_cols = [
    "asp_active",
    "age",
    "sex",
    "comorbidity_score",
    "immune_compromised",
    "severity_score",
    "ward",
    "infection_type",
    "hospital_size",
    "region",
]
cox_X = pd.get_dummies(df[cox_covariate_cols], drop_first=True, dtype=float)
cox_year_dummies = pd.get_dummies(
    df["year"].astype(str), prefix="year", drop_first=True, dtype=float
)
cox_df = pd.concat(
    [
        df[["time_to_improvement", "event_observed", "hospital_id"]].reset_index(drop=True),
        cox_X.reset_index(drop=True),
        cox_year_dummies.reset_index(drop=True),
    ],
    axis=1,
)

cph = CoxPHFitter()
cph.fit(
    cox_df,
    duration_col="time_to_improvement",
    event_col="event_observed",
    cluster_col="hospital_id",
)

cox_hr_asp = float(cph.hazard_ratios_["asp_active"])
cox_se_asp = float(cph.standard_errors_["asp_active"])

try:
    ph_test_res = proportional_hazard_test(cph, cox_df, time_transform="rank")
except KeyError:
    ph_test_res = proportional_hazard_test(
        cph, cox_df.drop(columns=["hospital_id"]), time_transform="rank"
    )

ph_test_pvalues = {str(k): float(v) for k, v in ph_test_res.summary["p"].items()}


# =============================================================================
# 2. CAUSAL IMPACT (DiD): Hospital-Year Panel, Pre-Trends, TWFE, Sun-Abraham
# =============================================================================
did_panel = (
    df.groupby(["hospital_id", "year"], as_index=False)
    .agg(
        antibiotic_duration=("antibiotic_duration", "mean"),
        asp_active=("asp_active", "first"),
        asp_adoption_year=("asp_adoption_year", "first"),
        age=("age", "mean"),
        comorbidity_score=("comorbidity_score", "mean"),
        immune_compromised=("immune_compromised", "mean"),
        severity_score=("severity_score", "mean"),
        n_patients=("patient_id", "count"),
    )
    .sort_values(["hospital_id", "year"])
    .reset_index(drop=True)
)

# Plain Two-Way Fixed Effects (TWFE) on hospital-year panel
panel_indexed = did_panel.set_index(["hospital_id", "year"])
twfe_mod = PanelOLS(
    dependent=panel_indexed["antibiotic_duration"],
    exog=panel_indexed[["asp_active"]],
    entity_effects=True,
    time_effects=True,
)
twfe_res = twfe_mod.fit(cov_type="clustered", cluster_entity=True)
twfe_att_estimate = float(twfe_res.params["asp_active"])

# Staggered-adoption robust DiD: Sun-Abraham interaction-weighted estimator
# (Equivalent to Callaway-Sant'Anna on balanced hospital-year panel using never-treated controls)
sa_df = did_panel.copy()
treated_cohorts = sorted(
    int(g) for g in sa_df["asp_adoption_year"].unique() if int(g) < 9000
)
cohort_sizes = {
    g: int(sa_df.loc[sa_df["asp_adoption_year"] == g, "hospital_id"].nunique())
    for g in treated_cohorts
}

col_map = {}
sa_cols = []
for g in treated_cohorts:
    rel_times = sorted(
        int(t - g) for t in sa_df.loc[sa_df["asp_adoption_year"] == g, "year"].unique()
    )
    for e in rel_times:
        if e == -1:
            continue
        col_name = f"sa_{g}_{'m' + str(abs(e)) if e < 0 else 'p' + str(e)}"
        sa_df[col_name] = (
            (sa_df["asp_adoption_year"] == g) & ((sa_df["year"] - g) == e)
        ).astype(float)
        col_map[(g, e)] = col_name
        sa_cols.append(col_name)

sa_formula = "antibiotic_duration ~ C(hospital_id) + C(year) + " + " + ".join(sa_cols)
sa_res = smf.ols(sa_formula, data=sa_df).fit(
    cov_type="cluster", cov_kwds={"groups": sa_df["hospital_id"]}
)
sa_param_names = list(sa_res.params.index)

all_event_times = sorted({e for (_, e) in col_map.keys()})
es_rows = []
lead_stats = {}

for e in all_event_times:
    active_cohorts = [g for g in treated_cohorts if (g, e) in col_map]
    total_w = float(sum(cohort_sizes[g] for g in active_cohorts))
    L_e = np.zeros(len(sa_param_names), dtype=float)
    for g in active_cohorts:
        idx = sa_param_names.index(col_map[(g, e)])
        L_e[idx] = cohort_sizes[g] / total_w

    tt_e = sa_res.t_test(L_e)
    coef_e = float(np.asarray(tt_e.effect).item())
    pval_e = float(np.asarray(tt_e.pvalue).item())
    ci_e = np.asarray(tt_e.conf_int()).reshape(-1)
    ci_lo_e, ci_hi_e = float(ci_e[0]), float(ci_e[1])

    es_rows.append(
        {
            "event_time": int(e),
            "coef": coef_e,
            "ci_low": ci_lo_e,
            "ci_high": ci_hi_e,
        }
    )
    if e < 0:
        lead_stats[e] = (coef_e, pval_e, [ci_lo_e, ci_hi_e])

event_study = pd.DataFrame(es_rows)[["event_time", "coef", "ci_low", "ci_high"]]

# Pre-trend validation using immediate pre-treatment lead (event_time = -2 vs -1 ref)
target_lead = -2 if -2 in lead_stats else max(lead_stats.keys())
pre_trend_coeff = float(lead_stats[target_lead][0])
pre_trend_pvalue = float(lead_stats[target_lead][1])
pre_trend_ci = [float(lead_stats[target_lead][2][0]), float(lead_stats[target_lead][2][1])]

# Overall staggered-robust ATT across all post-treatment (cohort, event_time >= 0) cells
post_cells = [(g, e) for (g, e) in col_map.keys() if e >= 0]
total_post_w = float(sum(cohort_sizes[g] for (g, _) in post_cells))
L_att = np.zeros(len(sa_param_names), dtype=float)
for g, e in post_cells:
    idx = sa_param_names.index(col_map[(g, e)])
    L_att[idx] = cohort_sizes[g] / total_post_w

did_att_estimate = float(np.asarray(sa_res.t_test(L_att).effect).item())


# =============================================================================
# 3. TREATMENT HETEROGENEITY: FE-Adjusted X-Learner + Bootstrap 95% CI
# =============================================================================
patient_het_cols = [
    "age",
    "age_group",
    "sex",
    "comorbidity_score",
    "immune_compromised",
    "severity_score",
    "ward",
    "infection_type",
    "hospital_size",
    "region",
]
X_het_df = pd.get_dummies(df[patient_het_cols], drop_first=True, dtype=float)
X_het = X_het_df.to_numpy(dtype=float)

# Construct design matrices to absorb hospital and year fixed effects on untreated (D=0)
# preventing staggered positivity violations while preserving individual CATE heterogeneity
H_mat = pd.get_dummies(
    df["hospital_id"].astype(str), prefix="h", drop_first=True, dtype=float
).to_numpy(dtype=float)
T_mat = pd.get_dummies(
    df["year"].astype(str), prefix="y", drop_first=True, dtype=float
).to_numpy(dtype=float)
X_pat_fe = pd.get_dummies(
    df[
        [
            "age",
            "age_group",
            "sex",
            "comorbidity_score",
            "immune_compromised",
            "severity_score",
            "ward",
            "infection_type",
        ]
    ],
    drop_first=True,
    dtype=float,
).to_numpy(dtype=float)

Y_all = df["antibiotic_duration"].to_numpy(dtype=float)
D_all = df["asp_active"].to_numpy(dtype=int)
age_all = df["age"].to_numpy(dtype=int)

n_h = H_mat.shape[1]
n_t = T_mat.shape[1]
Z_fe_all = np.column_stack([np.ones(len(df), dtype=float), H_mat, T_mat, X_pat_fe])


def fit_x_learner_cate(Z_fe, H_m, T_m, X_x, Y_vec, D_vec, seed):
    d0_mask = D_vec == 0
    d1_mask = D_vec == 1

    # Estimate hospital and year fixed effects cleanly from untreated observations
    beta_0, _, _, _ = np.linalg.lstsq(Z_fe[d0_mask], Y_vec[d0_mask], rcond=None)
    beta_h = beta_0[1 : 1 + n_h]
    beta_t = beta_0[1 + n_h : 1 + n_h + n_t]
    Y_adj = Y_vec - (H_m @ beta_h + T_m @ beta_t)

    gbr_params = dict(
        n_estimators=90,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=25,
        random_state=seed,
    )

    mu0 = GradientBoostingRegressor(**gbr_params)
    mu1 = GradientBoostingRegressor(**gbr_params)
    mu0.fit(X_x[d0_mask], Y_adj[d0_mask])
    mu1.fit(X_x[d1_mask], Y_adj[d1_mask])

    # Imputed treatment effects
    imp_tau1 = Y_adj[d1_mask] - mu0.predict(X_x[d1_mask])
    imp_tau0 = mu1.predict(X_x[d0_mask]) - Y_adj[d0_mask]

    tau1_mod = GradientBoostingRegressor(**gbr_params)
    tau0_mod = GradientBoostingRegressor(**gbr_params)
    tau1_mod.fit(X_x[d1_mask], imp_tau1)
    tau0_mod.fit(X_x[d0_mask], imp_tau0)

    prop_mod = GradientBoostingClassifier(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=25,
        random_state=seed,
    )
    prop_mod.fit(X_x, D_vec)
    g_hat = np.clip(prop_mod.predict_proba(X_x)[:, 1], 0.05, 0.95)

    cate = g_hat * tau0_mod.predict(X_x) + (1.0 - g_hat) * tau1_mod.predict(X_x)
    return np.asarray(cate, dtype=float)


cate_estimates = fit_x_learner_cate(Z_fe_all, H_mat, T_mat, X_het, Y_all, D_all, seed=42)

rng = np.random.default_rng(42)
n_obs = len(df)
n_boot = 100
boot_diffs = np.empty(n_boot, dtype=float)

for b in range(n_boot):
    b_idx = rng.integers(0, n_obs, size=n_obs)
    cate_b = fit_x_learner_cate(
        Z_fe_all[b_idx],
        H_mat[b_idx],
        T_mat[b_idx],
        X_het[b_idx],
        Y_all[b_idx],
        D_all[b_idx],
        seed=1000 + b,
    )
    age_b = age_all[b_idx]
    boot_diffs[b] = np.mean(cate_b[age_b >= 65]) - np.mean(cate_b[age_b < 40])

ci_bounds = np.percentile(boot_diffs, [2.5, 97.5])
cate_elderly_minus_young_ci = [float(ci_bounds[0]), float(ci_bounds[1])]


# =============================================================================
# 4. HIERARCHICAL MODELING: Mixed-Effects Random Intercept vs Naive OLS
# =============================================================================
fe_formula = (
    "antibiotic_duration ~ asp_active + age + C(sex) + comorbidity_score + "
    "immune_compromised + severity_score + C(ward) + C(infection_type) + C(year)"
)

mixed_mod = smf.mixedlm(fe_formula, data=df, groups=df["hospital_id"])
mixed_res = mixed_mod.fit(reml=False, method="lbfgs")
if not mixed_res.converged:
    mixed_res = mixed_mod.fit(reml=False, method="powell")

random_effect_variance = float(mixed_res.cov_re.iloc[0, 0])
residual_variance = float(mixed_res.scale)
icc_value = float(
    random_effect_variance / (random_effect_variance + residual_variance)
)
mixed_model_aic = float(mixed_res.aic)
mixed_asp_coeff = float(mixed_res.fe_params["asp_active"])

naive_ols_res = smf.ols(fe_formula, data=df).fit()
naive_ols_aic = float(naive_ols_res.aic)