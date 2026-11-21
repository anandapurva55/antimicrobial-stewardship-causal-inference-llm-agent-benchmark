import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import proportional_hazard_test, logrank_test
from linearmodels.panel import PanelOLS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.ensemble import HistGradientBoostingRegressor as _GBR
    _GBR_PARAMS = {
        "max_iter": 80,
        "max_depth": 4,
        "min_samples_leaf": 20,
        "learning_rate": 0.1,
    }
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor as _GBR
    _GBR_PARAMS = {
        "n_estimators": 60,
        "max_depth": 3,
        "min_samples_leaf": 20,
        "learning_rate": 0.1,
    }

# ==============================================================================
# 1. SURVIVAL ANALYSIS: Kaplan-Meier & Cox Proportional Hazards (clustered SEs)
# ==============================================================================
km_treated = KaplanMeierFitter()
km_control = KaplanMeierFitter()
mask_asp = df["asp_active"] == 1
km_treated.fit(
    df.loc[mask_asp, "time_to_improvement"],
    event_observed=df.loc[mask_asp, "event_observed"],
    label="ASP Active",
)
km_control.fit(
    df.loc[~mask_asp, "time_to_improvement"],
    event_observed=df.loc[~mask_asp, "event_observed"],
    label="No ASP",
)
km_logrank = logrank_test(
    df.loc[mask_asp, "time_to_improvement"],
    df.loc[~mask_asp, "time_to_improvement"],
    event_observed_A=df.loc[mask_asp, "event_observed"],
    event_observed_B=df.loc[~mask_asp, "event_observed"],
)

cox_num_cols = [
    "asp_active",
    "age",
    "comorbidity_score",
    "immune_compromised",
    "severity_score",
]
cox_cat_cols = ["hospital_id", "year", "ward", "sex", "infection_type"]
cox_dummies = pd.get_dummies(
    df[cox_cat_cols].astype(str), drop_first=True, dtype=float
)
cox_df = pd.concat(
    [
        df[["time_to_improvement", "event_observed"] + cox_num_cols].astype(float),
        cox_dummies,
    ],
    axis=1,
)
cox_df["cluster_hospital_id"] = df["hospital_id"].values

cph = CoxPHFitter()
cph.fit(
    cox_df,
    duration_col="time_to_improvement",
    event_col="event_observed",
    cluster_col="cluster_hospital_id",
)

cox_hr_asp = float(cph.hazard_ratios_["asp_active"])
cox_se_asp = float(cph.standard_errors_["asp_active"])

ph_test_res = proportional_hazard_test(cph, cox_df, time_transform="rank")
ph_test_pvalues = {
    str(cov): float(pval) for cov, pval in ph_test_res.summary["p"].items()
}

# ==============================================================================
# 2. CAUSAL IMPACT (DiD): Panel, TWFE, Pre-Trends & Staggered Adoption (Sun-Abraham / CS)
# ==============================================================================
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

did_panel_idx = did_panel.set_index(["hospital_id", "year"])
twfe_mod = PanelOLS.from_formula(
    "antibiotic_duration ~ asp_active + EntityEffects + TimeEffects",
    data=did_panel_idx,
)
twfe_res = twfe_mod.fit(cov_type="clustered", cluster_entity=True)
twfe_att_estimate = float(twfe_res.params["asp_active"])

treated_cohorts = sorted(
    [int(g) for g in did_panel["asp_adoption_year"].unique() if g != 9999]
)
cohort_counts = {
    g: int(did_panel.loc[did_panel["asp_adoption_year"] == g, "hospital_id"].nunique())
    for g in treated_cohorts
}
n_clusters = int(did_panel["hospital_id"].nunique())
t_crit = float(stats.t.ppf(0.975, df=n_clusters - 1))

sa_df = did_panel.copy()
sa_terms = []
cohort_event_map = {}

for g in treated_cohorts:
    rel_times = sorted(
        [int(yr - g) for yr in sa_df["year"].unique() if int(yr - g) != -1]
    )
    for e in rel_times:
        tag = f"m{abs(e)}" if e < 0 else f"p{e}"
        col = f"D_{g}_{tag}"
        sa_df[col] = (
            (sa_df["asp_adoption_year"] == g) & ((sa_df["year"] - g) == e)
        ).astype(float)
        sa_terms.append(col)
        cohort_event_map.setdefault(e, []).append((g, col))

sa_idx = sa_df.set_index(["hospital_id", "year"])
sa_formula = (
    "antibiotic_duration ~ " + " + ".join(sa_terms) + " + EntityEffects + TimeEffects"
)
sa_res = PanelOLS.from_formula(sa_formula, data=sa_idx).fit(
    cov_type="clustered", cluster_entity=True
)

param_names = list(sa_res.params.index)
cov_mat = sa_res.cov.loc[param_names, param_names].values

event_rows = []
for e in sorted(cohort_event_map.keys()):
    pairs = cohort_event_map[e]
    total_n = sum(cohort_counts[g] for g, _ in pairs)
    w = np.zeros(len(param_names), dtype=float)
    for g, col in pairs:
        w[param_names.index(col)] = cohort_counts[g] / total_n
    coef_e = float(w @ sa_res.params.values)
    se_e = float(np.sqrt(max(w @ cov_mat @ w, 0.0)))
    ci_low_e = float(coef_e - t_crit * se_e)
    ci_high_e = float(coef_e + t_crit * se_e)
    event_rows.append(
        {
            "event_time": int(e),
            "coef": coef_e,
            "ci_low": ci_low_e,
            "ci_high": ci_high_e,
            "_se": se_e,
        }
    )

event_study_full = pd.DataFrame(event_rows).sort_values("event_time").reset_index(drop=True)
event_study = event_study_full[["event_time", "coef", "ci_low", "ci_high"]].copy()

pre_lead_row = event_study_full.loc[event_study_full["event_time"] == -2].iloc[0]
pre_trend_coeff = float(pre_lead_row["coef"])
pre_trend_se = float(pre_lead_row["_se"])
pre_trend_pvalue = float(
    2.0 * (1.0 - stats.t.cdf(abs(pre_trend_coeff / pre_trend_se), df=n_clusters - 1))
)
pre_trend_ci = [float(pre_lead_row["ci_low"]), float(pre_lead_row["ci_high"])]

post_pairs = [
    (g, col)
    for e, pairs in cohort_event_map.items()
    if e >= 0
    for g, col in pairs
]
total_post_weight = sum(cohort_counts[g] for g, _ in post_pairs)
w_att = np.zeros(len(param_names), dtype=float)
for g, col in post_pairs:
    w_att[param_names.index(col)] = cohort_counts[g] / total_post_weight
did_att_estimate = float(w_att @ sa_res.params.values)

# ==============================================================================
# 3. TREATMENT HETEROGENEITY: Hospital/Year-Adjusted X-Learner + Bootstrap CI
# ==============================================================================
cate_num_cols = [
    "age",
    "comorbidity_score",
    "immune_compromised",
    "severity_score",
]
cate_cat_cols = ["sex", "ward", "infection_type"]
X_cate_df = pd.concat(
    [
        df[cate_num_cols].astype(float),
        pd.get_dummies(df[cate_cat_cols], drop_first=True, dtype=float),
    ],
    axis=1,
)
X_cate = X_cate_df.values
Z_fe = pd.get_dummies(
    df[["hospital_id", "year"]].astype(str), drop_first=True, dtype=float
).values
y_primary = df["antibiotic_duration"].values.astype(float)
w_treat = df["asp_active"].values.astype(int)
age_vals = df["age"].values.astype(float)


def _fit_x_learner(X_tr, y_tr, w_tr, Z_tr, X_eval, seed):
    mask0 = w_tr == 0
    mask1 = w_tr == 1

    A_tr = np.column_stack([np.ones(len(y_tr)), Z_tr, X_tr])
    beta_fe, _, _, _ = np.linalg.lstsq(A_tr[mask0], y_tr[mask0], rcond=None)
    n_fe = Z_tr.shape[1]
    y_adj = y_tr - (Z_tr @ beta_fe[1 : 1 + n_fe])

    mu0 = _GBR(random_state=seed, **_GBR_PARAMS)
    mu1 = _GBR(random_state=seed + 1, **_GBR_PARAMS)
    mu0.fit(X_tr[mask0], y_adj[mask0])
    mu1.fit(X_tr[mask1], y_adj[mask1])

    d1 = y_adj[mask1] - mu0.predict(X_tr[mask1])
    d0 = mu1.predict(X_tr[mask0]) - y_adj[mask0]

    tau1 = _GBR(random_state=seed + 2, **_GBR_PARAMS)
    tau0 = _GBR(random_state=seed + 3, **_GBR_PARAMS)
    tau1.fit(X_tr[mask1], d1)
    tau0.fit(X_tr[mask0], d0)

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_eval_scaled = scaler.transform(X_eval)
    prop_clf = LogisticRegression(max_iter=500, random_state=seed)
    prop_clf.fit(X_tr_scaled, w_tr)
    p_hat = np.clip(prop_clf.predict_proba(X_eval_scaled)[:, 1], 0.05, 0.95)

    return p_hat * tau0.predict(X_eval) + (1.0 - p_hat) * tau1.predict(X_eval)


cate_estimates = np.asarray(
    _fit_x_learner(X_cate, y_primary, w_treat, Z_fe, X_cate, seed=42),
    dtype=float,
)

rng = np.random.default_rng(42)
n_obs = len(df)
n_boot = 100
boot_diffs = np.empty(n_boot, dtype=float)

for b in range(n_boot):
    idx_b = rng.integers(0, n_obs, size=n_obs)
    X_b = X_cate[idx_b]
    y_b = y_primary[idx_b]
    w_b = w_treat[idx_b]
    Z_b = Z_fe[idx_b]
    age_b = age_vals[idx_b]

    cate_b = _fit_x_learner(X_b, y_b, w_b, Z_b, X_b, seed=1000 + b * 10)
    boot_diffs[b] = float(
        np.mean(cate_b[age_b >= 65]) - np.mean(cate_b[age_b < 40])
    )

cate_elderly_minus_young_ci = [
    float(np.percentile(boot_diffs, 2.5)),
    float(np.percentile(boot_diffs, 97.5)),
]

# ==============================================================================
# 4. HIERARCHICAL MODELING: Hospital Random-Intercept Mixed Model vs Naive OLS
# ==============================================================================
fixed_formula = (
    "antibiotic_duration ~ asp_active + C(year) + C(hospital_size) + C(region) "
    "+ C(ward) + age + C(sex) + comorbidity_score + immune_compromised "
    "+ C(infection_type) + severity_score"
)

mixed_mod = smf.mixedlm(fixed_formula, data=df, groups=df["hospital_id"])
mixed_res = mixed_mod.fit(reml=False, method=["lbfgs", "powell"])

ols_res = smf.ols(fixed_formula, data=df).fit()

random_effect_variance = float(mixed_res.cov_re.iloc[0, 0])
residual_variance = float(mixed_res.scale)
icc_value = float(
    random_effect_variance / (random_effect_variance + residual_variance)
)
mixed_model_aic = float(mixed_res.aic)
if np.isnan(mixed_model_aic):
    mixed_model_aic = float(
        -2.0 * mixed_res.llf + 2.0 * (len(mixed_res.fe_params) + 2)
    )
naive_ols_aic = float(ols_res.aic)
mixed_asp_coeff = float(mixed_res.params["asp_active"])