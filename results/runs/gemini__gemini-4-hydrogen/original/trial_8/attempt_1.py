import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from linearmodels.panel import PanelOLS
from sklearn.ensemble import HistGradientBoostingRegressor

# ==============================================================================
# 1. SURVIVAL ANALYSIS
# ==============================================================================
kmf_treated = KaplanMeierFitter()
kmf_control = KaplanMeierFitter()
treated_mask = df["asp_active"] == 1
kmf_treated.fit(
    df.loc[treated_mask, "time_to_improvement"],
    event_observed=df.loc[treated_mask, "event_observed"],
    label="ASP Active",
)
kmf_control.fit(
    df.loc[~treated_mask, "time_to_improvement"],
    event_observed=df.loc[~treated_mask, "event_observed"],
    label="Control",
)

cox_cols = [
    "time_to_improvement",
    "event_observed",
    "hospital_id",
    "asp_active",
    "age",
    "comorbidity_score",
    "immune_compromised",
    "severity_score",
    "sex",
    "ward",
    "infection_type",
    "year",
]
cox_df = pd.get_dummies(
    df[cox_cols].copy(),
    columns=["sex", "ward", "infection_type", "year"],
    drop_first=True,
    dtype=float,
)

cph = CoxPHFitter()
try:
    cph.fit(
        cox_df,
        duration_col="time_to_improvement",
        event_col="event_observed",
        cluster_col="hospital_id",
    )
except Exception:
    cph = CoxPHFitter(penalizer=0.001)
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

# ==============================================================================
# 2. CAUSAL IMPACT (DiD): PANEL, PARALLEL PRE-TRENDS, TWFE & STAGGERED DiD
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
did_panel["asp_active"] = (did_panel["asp_active"] >= 0.5).astype(int)

# Sun-Abraham (2021) interaction-weighted event study using never-treated controls
treated_cohorts = sorted(
    [int(g) for g in did_panel["asp_adoption_year"].unique() if g < 9999]
)
hospital_cohorts = did_panel.groupby("hospital_id")["asp_adoption_year"].first()
cohort_sizes = {int(g): int((hospital_cohorts == g).sum()) for g in treated_cohorts}

sa_df = did_panel.copy()
sa_cols = []
col_meta = {}

for g in treated_cohorts:
    years_for_g = sorted(sa_df.loc[sa_df["asp_adoption_year"] == g, "year"].unique())
    for y in years_for_g:
        e = int(y) - g
        if e == -1:
            continue
        col_name = f"sa_g{g}_e{e}"
        sa_df[col_name] = (
            (sa_df["asp_adoption_year"] == g) & (sa_df["year"] == y)
        ).astype(float)
        sa_cols.append(col_name)
        col_meta[col_name] = (g, e)

sa_indexed = sa_df.set_index(["hospital_id", "year"])
sa_mod = PanelOLS(
    sa_indexed["antibiotic_duration"],
    sa_indexed[sa_cols],
    entity_effects=True,
    time_effects=True,
)
sa_res = sa_mod.fit(cov_type="clustered", cluster_entity=True)

n_clusters = int(did_panel["hospital_id"].nunique())
df_clust = max(n_clusters - 1, 1)
t_crit = float(stats.t.ppf(0.975, df=df_clust))

params_vec = sa_res.params[sa_cols].values
cov_mat = sa_res.cov.loc[sa_cols, sa_cols].values

all_event_times = sorted(set(e for (_, e) in col_meta.values()))
es_rows = []
pre_trend_stats = {}

for e in all_event_times:
    w = np.zeros(len(sa_cols), dtype=float)
    total_w = 0.0
    for idx_c, col_name in enumerate(sa_cols):
        g_c, e_c = col_meta[col_name]
        if e_c == e:
            w[idx_c] = float(cohort_sizes[g_c])
            total_w += float(cohort_sizes[g_c])
    w = w / total_w
    coef_e = float(np.dot(w, params_vec))
    se_e = float(np.sqrt(max(float(w @ cov_mat @ w), 0.0)))
    ci_low_e = float(coef_e - t_crit * se_e)
    ci_high_e = float(coef_e + t_crit * se_e)
    p_val_e = float(2.0 * (1.0 - stats.t.cdf(abs(coef_e / se_e), df=df_clust))) if se_e > 0 else 1.0
    es_rows.append(
        {
            "event_time": int(e),
            "coef": coef_e,
            "ci_low": ci_low_e,
            "ci_high": ci_high_e,
        }
    )
    if e < 0:
        pre_trend_stats[e] = (coef_e, p_val_e, [ci_low_e, ci_high_e])

event_study = pd.DataFrame(es_rows, columns=["event_time", "coef", "ci_low", "ci_high"])

# Report primary pre-treatment lead (event_time = -2, identifiable across all treated cohorts)
lead_key = -2 if -2 in pre_trend_stats else max(pre_trend_stats.keys())
pre_trend_coeff, pre_trend_pvalue, pre_trend_ci = pre_trend_stats[lead_key]
pre_trend_coeff = float(pre_trend_coeff)
pre_trend_pvalue = float(pre_trend_pvalue)
pre_trend_ci = [float(pre_trend_ci[0]), float(pre_trend_ci[1])]

# Plain TWFE on hospital-year panel
panel_indexed = did_panel.set_index(["hospital_id", "year"])
twfe_mod = PanelOLS(
    panel_indexed["antibiotic_duration"],
    panel_indexed[["asp_active"]],
    entity_effects=True,
    time_effects=True,
)
twfe_res = twfe_mod.fit(cov_type="clustered", cluster_entity=True)
twfe_att_estimate = float(twfe_res.params["asp_active"])

# Staggered-adoption-robust ATT (weighted average of post-treatment cohort-time ATTs, never-treated controls)
w_post = np.zeros(len(sa_cols), dtype=float)
total_post_w = 0.0
for idx_c, col_name in enumerate(sa_cols):
    g_c, e_c = col_meta[col_name]
    if e_c >= 0:
        w_post[idx_c] = float(cohort_sizes[g_c])
        total_post_w += float(cohort_sizes[g_c])
w_post = w_post / total_post_w
did_att_estimate = float(np.dot(w_post, params_vec))

# ==============================================================================
# 3. TREATMENT HETEROGENEITY (T-LEARNER WITH GRADIENT BOOSTING)
# ==============================================================================
cate_features = [
    "age",
    "comorbidity_score",
    "immune_compromised",
    "severity_score",
    "sex",
    "ward",
    "infection_type",
    "hospital_size",
    "region",
    "hospital_id",
    "year",
]
X_cate = pd.get_dummies(
    df[cate_features],
    columns=["sex", "ward", "infection_type", "hospital_size", "region", "hospital_id", "year"],
    drop_first=False,
    dtype=float,
).values
Y_cate = df["antibiotic_duration"].values.astype(float)
W_cate = df["asp_active"].values.astype(int)


def _fit_predict_t_learner(X_tr, W_tr, Y_tr, X_ev, seed=42):
    m0 = HistGradientBoostingRegressor(
        max_iter=80, max_depth=5, learning_rate=0.08, random_state=seed
    )
    m1 = HistGradientBoostingRegressor(
        max_iter=80, max_depth=5, learning_rate=0.08, random_state=seed
    )
    m0.fit(X_tr[W_tr == 0], Y_tr[W_tr == 0])
    m1.fit(X_tr[W_tr == 1], Y_tr[W_tr == 1])
    return m1.predict(X_ev) - m0.predict(X_ev)


cate_estimates = np.asarray(
    _fit_predict_t_learner(X_cate, W_cate, Y_cate, X_cate, seed=42),
    dtype=float,
)

rng = np.random.default_rng(42)
elderly_mask = df["age"].values >= 65
young_mask = df["age"].values < 40
n_obs = len(df)
n_boot = 50
boot_diffs = []

for b in range(n_boot):
    idx = rng.integers(0, n_obs, size=n_obs)
    W_b = W_cate[idx]
    e_b = elderly_mask[idx]
    y_b = young_mask[idx]
    if (W_b == 0).sum() < 10 or (W_b == 1).sum() < 10 or e_b.sum() == 0 or y_b.sum() == 0:
        continue
    cate_b = _fit_predict_t_learner(X_cate[idx], W_b, Y_cate[idx], X_cate[idx], seed=100 + b)
    boot_diffs.append(float(np.mean(cate_b[e_b]) - np.mean(cate_b[y_b])))

cate_elderly_minus_young_ci = [
    float(np.percentile(boot_diffs, 2.5)),
    float(np.percentile(boot_diffs, 97.5)),
]

# ==============================================================================
# 4. HIERARCHICAL MODELING (MIXED-EFFECTS RANDOM INTERCEPT VS NAIVE OLS)
# ==============================================================================
mixed_formula = (
    "antibiotic_duration ~ asp_active + age + comorbidity_score + "
    "immune_compromised + severity_score + C(sex) + C(ward) + C(infection_type) + C(year)"
)
mixed_mod = smf.mixedlm(mixed_formula, data=df, groups=df["hospital_id"])
mixed_fit = mixed_mod.fit(reml=False)

ols_fit = smf.ols(mixed_formula, data=df).fit()

random_effect_variance = float(mixed_fit.cov_re.iloc[0, 0])
residual_variance = float(mixed_fit.scale)
icc_value = float(random_effect_variance / (random_effect_variance + residual_variance))

mixed_asp_coeff = float(mixed_fit.fe_params["asp_active"])
mixed_model_aic = (
    float(mixed_fit.aic)
    if np.isfinite(getattr(mixed_fit, "aic", np.nan))
    else float(-2.0 * mixed_fit.llf + 2.0 * (mixed_fit.k_fe + 2))
)
naive_ols_aic = float(ols_fit.aic)