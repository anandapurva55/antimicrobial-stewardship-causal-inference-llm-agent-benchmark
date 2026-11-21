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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

data = df.copy().reset_index(drop=True)

# =============================================================================
# 1. SURVIVAL ANALYSIS
# =============================================================================
kmf_control = KaplanMeierFitter()
kmf_treated = KaplanMeierFitter()

mask_ctrl = data["asp_active"] == 0
mask_trt = data["asp_active"] == 1

kmf_control.fit(
    durations=data.loc[mask_ctrl, "time_to_improvement"],
    event_observed=data.loc[mask_ctrl, "event_observed"],
    label="ASP Inactive",
)
kmf_treated.fit(
    durations=data.loc[mask_trt, "time_to_improvement"],
    event_observed=data.loc[mask_trt, "event_observed"],
    label="ASP Active",
)
_km_logrank = logrank_test(
    data.loc[mask_trt, "time_to_improvement"],
    data.loc[mask_ctrl, "time_to_improvement"],
    event_observed_A=data.loc[mask_trt, "event_observed"],
    event_observed_B=data.loc[mask_ctrl, "event_observed"],
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
cox_dummies = pd.get_dummies(data[cox_cat_cols], drop_first=True, dtype=float)
cox_df = pd.concat(
    [
        data[["time_to_improvement", "event_observed", "hospital_id"] + cox_num_cols],
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

cox_hr_asp = float(np.exp(cph.params_["asp_active"]))
cox_se_asp = float(cph.standard_errors_["asp_active"])

try:
    ph_res = proportional_hazard_test(cph, cox_df, time_transform="rank")
except Exception:
    ph_res = proportional_hazard_test(
        cph, cox_df.drop(columns=["hospital_id"]), time_transform="rank"
    )
ph_test_pvalues = {str(k): float(v) for k, v in ph_res.summary["p"].items()}

# =============================================================================
# 2. CAUSAL IMPACT (DiD & STAGGERED EVENT STUDY)
# =============================================================================
did_panel = (
    data.groupby(["hospital_id", "year"], as_index=False)
    .agg(
        antibiotic_duration=("antibiotic_duration", "mean"),
        asp_active=("asp_active", "max"),
        asp_adoption_year=("asp_adoption_year", "first"),
    )
    .sort_values(["hospital_id", "year"])
    .reset_index(drop=True)
)

panel_indexed = did_panel.set_index(["hospital_id", "year"])
twfe_res = PanelOLS.from_formula(
    "antibiotic_duration ~ 1 + asp_active + EntityEffects + TimeEffects",
    data=panel_indexed,
).fit(cov_type="clustered", cluster_entity=True)
twfe_att_estimate = float(twfe_res.params["asp_active"])

# Sun & Abraham (2021) interaction-weighted estimator / Callaway-Sant'Anna equivalent
# using never-treated hospitals (asp_adoption_year > max_year) as controls
sa_df = did_panel.copy()
max_year = int(sa_df["year"].max())
treated_cohorts = sorted(
    sa_df.loc[sa_df["asp_adoption_year"] <= max_year, "asp_adoption_year"].unique()
)

cohort_event_terms = {}
for g in treated_cohorts:
    g_int = int(g)
    rel_times = sorted((sa_df.loc[sa_df["asp_adoption_year"] == g, "year"] - g_int).unique())
    for e in rel_times:
        e_int = int(e)
        if e_int == -1:
            continue
        col_name = f"sa_g{g_int}_e{'m' + str(abs(e_int)) if e_int < 0 else str(e_int)}"
        sa_df[col_name] = (
            (sa_df["asp_adoption_year"] == g_int) & ((sa_df["year"] - g_int) == e_int)
        ).astype(float)
        n_cell = int(sa_df[col_name].sum())
        cohort_event_terms[(g_int, e_int)] = (col_name, n_cell)

sa_rhs = " + ".join([ v[0] for v in cohort_event_terms.values() ])
sa_formula = f"antibiotic_duration ~ {sa_rhs} + C(hospital_id) + C(year)"
sa_fit = smf.ols(sa_formula, data=sa_df).fit(
    cov_type="cluster", cov_kwds={"groups": sa_df["hospital_id"]}
)

param_names = list(sa_fit.params.index)
beta_vec = sa_fit.params.values
vcov_mat = sa_fit.cov_params().values
n_clusters = int(sa_df["hospital_id"].nunique())
df_resid_clust = n_clusters - 1
t_crit = float(stats.t.ppf(0.975, df=df_resid_clust))

all_event_times = sorted({e for (_, e) in cohort_event_terms.keys()})
es_rows = []
lead_stats = {}

for e in all_event_times:
    matching = [(g, col, cnt) for (g, ev), (col, cnt) in cohort_event_terms.items() if ev == e]
    total_cnt = sum(cnt for _, _, cnt in matching)
    w_vec = np.zeros(len(param_names), dtype=float)
    for _, col, cnt in matching:
        w_vec[param_names.index(col)] = cnt / total_cnt
    coef_e = float(w_vec @ beta_vec)
    se_e = float(np.sqrt(max(float(w_vec @ vcov_mat @ w_vec), 0.0)))
    ci_lo = float(coef_e - t_crit * se_e)
    ci_hi = float(coef_e + t_crit * se_e)
    t_stat = coef_e / se_e if se_e > 0 else 0.0
    pval_e = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df_resid_clust)))
    es_rows.append(
        {
            "event_time": int(e),
            "coef": coef_e,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
        }
    )
    lead_stats[int(e)] = (coef_e, pval_e, [ci_lo, ci_hi])

event_study = pd.DataFrame(es_rows)[["event_time", "coef", "ci_low", "ci_high"]]

# Validate parallel pre-trends via immediate pre-treatment lead (event_time = -2)
pre_trend_coeff, pre_trend_pvalue, pre_trend_ci = lead_stats[-2]

# Aggregate post-treatment cohort-event ATTs weighted by treated hospital-year counts
post_matching = [
    (col, cnt) for (_, ev), (col, cnt) in cohort_event_terms.items() if ev >= 0
]
total_post_cnt = sum(cnt for _, cnt in post_matching)
w_post = np.zeros(len(param_names), dtype=float)
for col, cnt in post_matching:
    w_post[param_names.index(col)] = cnt / total_post_cnt
did_att_estimate = float(w_post @ beta_vec)

# =============================================================================
# 3. TREATMENT HETEROGENEITY (X-LEARNER WITH HOSPITAL & YEAR ADJUSTMENT)
# =============================================================================
# Prebuild design matrix to orthogonalize hospital and year fixed effects on untreated rows
hosp_dummies = pd.get_dummies(data["hospital_id"], prefix="h", drop_first=True, dtype=float)
year_dummies = pd.get_dummies(data["year"], prefix="yr", drop_first=True, dtype=float)
pat_cov_df = pd.concat(
    [
        data[["age", "comorbidity_score", "immune_compromised", "severity_score"]].astype(float),
        pd.get_dummies(
            data[["sex", "age_group", "ward", "infection_type"]],
            drop_first=True,
            dtype=float,
        ),
    ],
    axis=1,
)

n_hosp_cols = hosp_dummies.shape[1]
n_year_cols = year_dummies.shape[1]
Z_fe_full = np.column_stack(
    [
        np.ones(len(data), dtype=float),
        hosp_dummies.values,
        year_dummies.values,
        pat_cov_df.values,
    ]
)
Z_hy_only = np.column_stack([hosp_dummies.values, year_dummies.values])

X_cate_df = pd.concat(
    [
        pat_cov_df,
        pd.get_dummies(data[["hospital_size", "region"]], drop_first=True, dtype=float),
    ],
    axis=1,
)
X_cate = X_cate_df.values
T_vec = data["asp_active"].values.astype(int)
Y_vec = data["antibiotic_duration"].values.astype(float)
age_vec = data["age"].values

scaler_ps = StandardScaler()
X_scaled = scaler_ps.fit_transform(X_cate)
ps_model = LogisticRegression(max_iter=1000, random_state=42)
ps_model.fit(X_scaled, T_vec)
e_hat_full = np.clip(ps_model.predict_proba(X_scaled)[:, 1], 0.05, 0.95)


def fit_x_learner(Z_full, Z_hy, X_mat, T_arr, Y_arr, e_prob, n_est=150, lr=0.05, seed=42):
    u_mask = T_arr == 0
    t_mask = T_arr == 1

    fe_coef, _, _, _ = np.linalg.lstsq(Z_full[u_mask], Y_arr[u_mask], rcond=None)
    hy_coef = fe_coef[1 : 1 + n_hosp_cols + n_year_cols]
    y_adj = Y_arr - (Z_hy @ hy_coef)

    mu0 = GradientBoostingRegressor(
        n_estimators=n_est, max_depth=3, learning_rate=lr, subsample=0.8, random_state=seed
    )
    mu1 = GradientBoostingRegressor(
        n_estimators=n_est, max_depth=3, learning_rate=lr, subsample=0.8, random_state=seed + 1
    )
    mu0.fit(X_mat[u_mask], y_adj[u_mask])
    mu1.fit(X_mat[t_mask], y_adj[t_mask])

    d1 = y_adj[t_mask] - mu0.predict(X_mat[t_mask])
    d0 = mu1.predict(X_mat[u_mask]) - y_adj[u_mask]

    tau1 = GradientBoostingRegressor(
        n_estimators=n_est, max_depth=3, learning_rate=lr, subsample=0.8, random_state=seed + 2
    )
    tau0 = GradientBoostingRegressor(
        n_estimators=n_est, max_depth=3, learning_rate=lr, subsample=0.8, random_state=seed + 3
    )
    tau1.fit(X_mat[t_mask], d1)
    tau0.fit(X_mat[u_mask], d0)

    cate_pred = e_prob * tau0.predict(X_mat) + (1.0 - e_prob) * tau1.predict(X_mat)
    return np.asarray(cate_pred, dtype=float)


cate_estimates = fit_x_learner(
    Z_fe_full, Z_hy_only, X_cate, T_vec, Y_vec, e_hat_full, n_est=150, lr=0.05, seed=42
)

rng = np.random.default_rng(42)
n_rows = len(data)
n_boot = 100
boot_diffs = np.empty(n_boot, dtype=float)

for b in range(n_boot):
    b_idx = rng.integers(0, n_rows, size=n_rows)
    cate_b = fit_x_learner(
        Z_fe_full[b_idx],
        Z_hy_only[b_idx],
        X_cate[b_idx],
        T_vec[b_idx],
        Y_vec[b_idx],
        e_hat_full[b_idx],
        n_est=100,
        lr=0.08,
        seed=1000 + b * 4,
    )
    age_b = age_vec[b_idx]
    boot_diffs[b] = cate_b[age_b >= 65].mean() - cate_b[age_b < 40].mean()

cate_elderly_minus_young_ci = [
    float(np.percentile(boot_diffs, 2.5)),
    float(np.percentile(boot_diffs, 97.5)),
]

# =============================================================================
# 4. HIERARCHICAL MODELING
# =============================================================================
fe_formula = (
    "antibiotic_duration ~ asp_active + age + C(sex) + comorbidity_score + "
    "immune_compromised + severity_score + C(ward) + C(infection_type) + "
    "C(hospital_size) + C(region) + C(year)"
)

mixed_fit = smf.mixedlm(fe_formula, data=data, groups=data["hospital_id"]).fit(reml=False)
naive_ols_fit = smf.ols(fe_formula, data=data).fit()

random_effect_variance = float(mixed_fit.cov_re.iloc[0, 0])
residual_variance = float(mixed_fit.scale)
icc_value = float(random_effect_variance / (random_effect_variance + residual_variance))
mixed_model_aic = float(mixed_fit.aic)
naive_ols_aic = float(naive_ols_fit.aic)
mixed_asp_coeff = float(mixed_fit.fe_params["asp_active"])