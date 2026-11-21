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
kmf_control = KaplanMeierFitter()
kmf_treated = KaplanMeierFitter()

mask_ctrl = df["asp_active"] == 0
mask_trt = df["asp_active"] == 1

kmf_control.fit(
    durations=df.loc[mask_ctrl, "time_to_improvement"],
    event_observed=df.loc[mask_ctrl, "event_observed"],
    label="ASP_Inactive",
)
kmf_treated.fit(
    durations=df.loc[mask_trt, "time_to_improvement"],
    event_observed=df.loc[mask_trt, "event_observed"],
    label="ASP_Active",
)

cox_cat_cols = ["sex", "ward", "infection_type", "hospital_size", "region"]
cox_num_cols = [
    "asp_active",
    "age",
    "comorbidity_score",
    "immune_compromised",
    "severity_score",
    "year",
]
cox_dummies = pd.get_dummies(df[cox_cat_cols], drop_first=True, dtype=float)
cox_df = pd.concat(
    [
        df[["time_to_improvement", "event_observed", "hospital_id"] + cox_num_cols].copy(),
        cox_dummies,
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

ph_res = proportional_hazard_test(cph, cox_df, time_transform="rank")
ph_test_pvalues = {str(k): float(v) for k, v in ph_res.summary["p"].items()}

# =============================================================================
# 2. CAUSAL IMPACT (DiD): Panel, TWFE, Sun-Abraham / Callaway-Sant'Anna, Event Study
# =============================================================================
did_panel = (
    df.groupby(["hospital_id", "year"], as_index=False)
    .agg(
        antibiotic_duration=("antibiotic_duration", "mean"),
        asp_active=("asp_active", "mean"),
        asp_adoption_year=("asp_adoption_year", "first"),
    )
    .sort_values(["hospital_id", "year"])
    .reset_index(drop=True)
)
did_panel["asp_active"] = (did_panel["asp_active"] >= 0.5).astype(int)

twfe_panel = did_panel.set_index(["hospital_id", "year"])
twfe_res = PanelOLS.from_formula(
    "antibiotic_duration ~ asp_active + EntityEffects + TimeEffects",
    data=twfe_panel,
).fit(cov_type="clustered", cluster_entity=True)
twfe_att_estimate = float(twfe_res.params["asp_active"])

max_year = int(did_panel["year"].max())
treated_cohorts = sorted(
    [int(g) for g in did_panel["asp_adoption_year"].unique() if g <= max_year]
)

sa_df = did_panel.copy()
sa_col_map = {}
cohort_sizes = {
    g: int(sa_df.loc[sa_df["asp_adoption_year"] == g, "hospital_id"].nunique())
    for g in treated_cohorts
}

all_event_times = set()
for g in treated_cohorts:
    g_years = sorted(sa_df.loc[sa_df["asp_adoption_year"] == g, "year"].unique())
    for y_val in g_years:
        e = int(y_val - g)
        if e == -1:
            continue
        all_event_times.add(e)
        cname = f"sa_g{g}_m{abs(e)}" if e < 0 else f"sa_g{g}_p{e}"
        sa_df[cname] = (
            (sa_df["asp_adoption_year"] == g) & (sa_df["year"] == y_val)
        ).astype(float)
        sa_col_map[(g, e)] = cname

sa_cols = list(sa_col_map.values())
sa_formula = "antibiotic_duration ~ C(hospital_id) + C(year) + " + " + ".join(sa_cols)
sa_res = smf.ols(sa_formula, data=sa_df).fit(
    cov_type="cluster", cov_kwds={"groups": sa_df["hospital_id"]}
)

param_names = list(sa_res.params.index)
n_params = len(param_names)

es_rows = []
es_stats = {}
for e in sorted(all_event_times):
    avail_g = [g for g in treated_cohorts if (g, e) in sa_col_map]
    total_w = sum(cohort_sizes[g] for g in avail_g)
    contrast = np.zeros(n_params, dtype=float)
    for g in avail_g:
        idx_p = param_names.index(sa_col_map[(g, e)])
        contrast[idx_p] = cohort_sizes[g] / total_w
    tt = sa_res.t_test(contrast, use_t=True)
    coef_e = float(np.squeeze(tt.effect))
    pval_e = float(np.squeeze(tt.pvalue))
    ci_e = np.asarray(tt.conf_int(alpha=0.05)).reshape(-1)
    ci_lo, ci_hi = float(ci_e[0]), float(ci_e[1])
    es_rows.append(
        {
            "event_time": int(e),
            "coef": coef_e,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
        }
    )
    es_stats[int(e)] = (coef_e, pval_e, [ci_lo, ci_hi])

event_study = pd.DataFrame(es_rows).sort_values("event_time").reset_index(drop=True)

lead_key = -2 if -2 in es_stats else max(k for k in es_stats if k < 0)
pre_trend_coeff = float(es_stats[lead_key][0])
pre_trend_pvalue = float(es_stats[lead_key][1])
pre_trend_ci = [float(es_stats[lead_key][2][0]), float(es_stats[lead_key][2][1])]

post_cells = [(g, e) for (g, e) in sa_col_map if e >= 0]
total_post_weight = sum(cohort_sizes[g] for (g, _) in post_cells)
att_contrast = np.zeros(n_params, dtype=float)
for g, e in post_cells:
    idx_p = param_names.index(sa_col_map[(g, e)])
    att_contrast[idx_p] = cohort_sizes[g] / total_post_weight

att_tt = sa_res.t_test(att_contrast, use_t=True)
did_att_estimate = float(np.squeeze(att_tt.effect))

# =============================================================================
# 3. TREATMENT HETEROGENEITY: X-Learner Adjusting for Hospital & Year + Bootstrap CI
# =============================================================================
hosp_dummies = pd.get_dummies(df["hospital_id"], prefix="hosp", drop_first=True, dtype=float)
year_dummies = pd.get_dummies(df["year"], prefix="yr", drop_first=True, dtype=float)
X_hy = np.column_stack([hosp_dummies.values, year_dummies.values])
n_hy = X_hy.shape[1]

patient_cat = pd.get_dummies(
    df[["sex", "ward", "infection_type", "hospital_size", "region"]],
    drop_first=True,
    dtype=float,
)
patient_num = df[["age", "comorbidity_score", "immune_compromised", "severity_score"]].astype(float)
X_ml = np.column_stack([patient_num.values, patient_cat.values])

patient_fe_cat = pd.get_dummies(
    df[["sex", "ward", "infection_type"]], drop_first=True, dtype=float
)
X_fe_ctrl = np.column_stack(
    [np.ones(len(df), dtype=float), X_hy, patient_num.values, patient_fe_cat.values]
)

y_all = df["antibiotic_duration"].values.astype(float)
w_all = df["asp_active"].values.astype(int)
age_all = df["age"].values


def fit_x_learner(X_fe_mat, X_hy_mat, X_feat, y_vec, w_vec, seed):
    ctrl_mask = w_vec == 0
    trt_mask = w_vec == 1

    beta_ctrl, _, _, _ = np.linalg.lstsq(X_fe_mat[ctrl_mask], y_vec[ctrl_mask], rcond=None)
    hy_effect = X_hy_mat @ beta_ctrl[1 : 1 + n_hy]
    y_adj = y_vec - hy_effect

    prop_model = GradientBoostingClassifier(
        n_estimators=50, max_depth=2, learning_rate=0.08, random_state=seed
    )
    prop_model.fit(X_feat, w_vec)
    e_hat = np.clip(prop_model.predict_proba(X_feat)[:, 1], 0.05, 0.95)

    gb_params = dict(
        n_estimators=70,
        max_depth=3,
        learning_rate=0.07,
        subsample=0.8,
        random_state=seed,
    )
    m0 = GradientBoostingRegressor(**gb_params).fit(X_feat[ctrl_mask], y_adj[ctrl_mask])
    m1 = GradientBoostingRegressor(**gb_params).fit(X_feat[trt_mask], y_adj[trt_mask])

    d1 = y_adj[trt_mask] - m0.predict(X_feat[trt_mask])
    d0 = m1.predict(X_feat[ctrl_mask]) - y_adj[ctrl_mask]

    tau1 = GradientBoostingRegressor(**gb_params).fit(X_feat[trt_mask], d1)
    tau0 = GradientBoostingRegressor(**gb_params).fit(X_feat[ctrl_mask], d0)

    cate = e_hat * tau0.predict(X_feat) + (1.0 - e_hat) * tau1.predict(X_feat)
    return np.asarray(cate, dtype=float)


cate_estimates = fit_x_learner(X_fe_ctrl, X_hy, X_ml, y_all, w_all, seed=42)

rng = np.random.default_rng(42)
n_obs = len(df)
n_boot = 80
boot_diffs = np.empty(n_boot, dtype=float)

for b in range(n_boot):
    idx = rng.integers(0, n_obs, size=n_obs)
    cate_b = fit_x_learner(
        X_fe_ctrl[idx],
        X_hy[idx],
        X_ml[idx],
        y_all[idx],
        w_all[idx],
        seed=1000 + b,
    )
    age_b = age_all[idx]
    boot_diffs[b] = float(
        np.mean(cate_b[age_b >= 65]) - np.mean(cate_b[age_b < 40])
    )

cate_elderly_minus_young_ci = [
    float(np.percentile(boot_diffs, 2.5)),
    float(np.percentile(boot_diffs, 97.5)),
]

# =============================================================================
# 4. HIERARCHICAL MODELING: Hospital Random-Intercept MixedLM (ML) vs Naive OLS
# =============================================================================
fe_formula = (
    "antibiotic_duration ~ asp_active + C(year) + age + C(sex) + "
    "comorbidity_score + immune_compromised + severity_score + "
    "C(ward) + C(infection_type)"
)

mixed_mod = smf.mixedlm(fe_formula, data=df, groups=df["hospital_id"])
mixed_res = mixed_mod.fit(reml=False, method=["lbfgs", "powell"])

naive_ols_res = smf.ols(fe_formula, data=df).fit()

random_effect_variance = float(mixed_res.cov_re.iloc[0, 0])
residual_variance = float(mixed_res.scale)
icc_value = float(random_effect_variance / (random_effect_variance + residual_variance))

mixed_model_aic = float(mixed_res.aic)
if np.isnan(mixed_model_aic):
    mixed_model_aic = float(-2.0 * mixed_res.llf + 2.0 * (len(mixed_res.fe_params) + 2))

naive_ols_aic = float(naive_ols_res.aic)
mixed_asp_coeff = float(mixed_res.fe_params["asp_active"])