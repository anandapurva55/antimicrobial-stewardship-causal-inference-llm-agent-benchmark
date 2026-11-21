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

# =============================================================================
# 1. SURVIVAL ANALYSIS: Kaplan-Meier, Cox PH (clustered by hospital_id), PH Test
# =============================================================================
kmf_asp = KaplanMeierFitter()
kmf_no_asp = KaplanMeierFitter()

mask_asp_1 = df["asp_active"] == 1
mask_asp_0 = df["asp_active"] == 0

kmf_asp.fit(
    df.loc[mask_asp_1, "time_to_improvement"],
    event_observed=df.loc[mask_asp_1, "event_observed"],
    label="ASP Active",
)
kmf_no_asp.fit(
    df.loc[mask_asp_0, "time_to_improvement"],
    event_observed=df.loc[mask_asp_0, "event_observed"],
    label="No ASP",
)

lr_test_res = logrank_test(
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
cox_cat_cols = ["sex", "ward", "infection_type", "year", "hospital_id"]
cox_dummies = pd.get_dummies(
    df[cox_cat_cols].astype(str),
    prefix=["sex", "ward", "inf", "yr", "hosp"],
    drop_first=True,
    dtype=float,
)

cox_df = pd.concat(
    [
        df[["time_to_improvement", "event_observed", "hospital_id"]].reset_index(drop=True),
        df[cox_num_cols].astype(float).reset_index(drop=True),
        cox_dummies.reset_index(drop=True),
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
# 2. CAUSAL IMPACT (DiD): Hospital-Year Panel, Pre-Trends, TWFE, Sun-Abraham/CS
# =============================================================================
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
did_panel["asp_active"] = did_panel["asp_active"].round().astype(int)
did_panel["asp_adoption_year"] = did_panel["asp_adoption_year"].astype(int)

panel_idx = did_panel.set_index(["hospital_id", "year"])

# Plain Two-Way Fixed Effects (TWFE)
twfe_mod = PanelOLS(
    panel_idx["antibiotic_duration"],
    panel_idx[["asp_active"]],
    entity_effects=True,
    time_effects=True,
)
twfe_res = twfe_mod.fit(cov_type="clustered", cluster_entity=True)
twfe_att_estimate = float(twfe_res.params["asp_active"])

# Staggered-Adoption Robust Estimator (Sun-Abraham / Callaway-Sant'Anna with never-treated controls)
n_hospitals = int(did_panel["hospital_id"].nunique())
treated_cohorts = sorted(
    int(g) for g in did_panel["asp_adoption_year"].unique() if g < 9000
)
cohort_counts = (
    did_panel.groupby("asp_adoption_year")["hospital_id"].nunique().to_dict()
)
years_list = sorted(int(y) for y in did_panel["year"].unique())

sa_df = panel_idx.copy()
sa_cols = []
col_meta = []
for g in treated_cohorts:
    for y in years_list:
        e = y - g
        if e == -1:
            continue
        cname = f"coh_{g}_e_{e}".replace("-", "m")
        sa_df[cname] = (
            (sa_df["asp_adoption_year"] == g)
            & (sa_df.index.get_level_values("year") == y)
        ).astype(float)
        sa_cols.append(cname)
        col_meta.append((cname, g, e))

sa_mod = PanelOLS(
    sa_df["antibiotic_duration"],
    sa_df[sa_cols],
    entity_effects=True,
    time_effects=True,
)
sa_res = sa_mod.fit(cov_type="clustered", cluster_entity=True)
sa_beta = sa_res.params[sa_cols].values
sa_cov = sa_res.cov.loc[sa_cols, sa_cols].values
t_crit = float(stats.t.ppf(0.975, df=n_hospitals - 1))

event_times = sorted(set(e for _, _, e in col_meta))
es_rows = []
es_details = {}
for e in event_times:
    w = np.zeros(len(sa_cols), dtype=float)
    for idx_c, (_, g, e_c) in enumerate(col_meta):
        if e_c == e:
            w[idx_c] = float(cohort_counts[g])
    w /= w.sum()
    coef_e = float(w @ sa_beta)
    se_e = float(np.sqrt(max(float(w @ sa_cov @ w), 1e-18)))
    ci_lo = float(coef_e - t_crit * se_e)
    ci_hi = float(coef_e + t_crit * se_e)
    pval_e = float(2.0 * (1.0 - stats.t.cdf(abs(coef_e / se_e), df=n_hospitals - 1)))
    es_rows.append(
        {
            "event_time": int(e),
            "coef": coef_e,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
        }
    )
    es_details[int(e)] = (coef_e, pval_e, [ci_lo, ci_hi])

event_study = pd.DataFrame(es_rows, columns=["event_time", "coef", "ci_low", "ci_high"])

pre_trend_coeff, pre_trend_pvalue, pre_trend_ci = es_details[-2]

w_att = np.zeros(len(sa_cols), dtype=float)
for idx_c, (_, g, e_c) in enumerate(col_meta):
    if e_c >= 0:
        w_att[idx_c] = float(cohort_counts[g])
w_att /= w_att.sum()
did_att_estimate = float(w_att @ sa_beta)

# =============================================================================
# 3. TREATMENT HETEROGENEITY: Hospital & Year Adjusted X-Learner + Bootstrap CI
# =============================================================================
pat_num_cols = ["age", "comorbidity_score", "immune_compromised", "severity_score"]
pat_cat_cols = ["sex", "ward", "infection_type"]
X_pat = pd.concat(
    [
        df[pat_num_cols].astype(float).reset_index(drop=True),
        pd.get_dummies(
            df[pat_cat_cols].astype(str), drop_first=True, dtype=float
        ).reset_index(drop=True),
    ],
    axis=1,
).values

H_mat = pd.get_dummies(df["hospital_id"].astype(str), drop_first=True, dtype=float).values
Y_mat = pd.get_dummies(df["year"].astype(str), drop_first=True, dtype=float).values

cp_list = []
for g in treated_cohorts:
    for y in years_list:
        if y >= g:
            cp_list.append(
                ((df["asp_adoption_year"].values == g) & (df["year"].values == y)).astype(float)
            )
CP_mat = np.column_stack(cp_list)

T_arr = df["asp_active"].values.astype(int)
y_arr = df["antibiotic_duration"].values.astype(float)
age_arr = df["age"].values.astype(float)


def _fit_predict_x_learner(X_in, H_in, Y_in, CP_in, T_in, y_in, seed=42):
    n_obs_local = len(y_in)
    ones = np.ones((n_obs_local, 1), dtype=float)
    A = np.column_stack([ones, H_in, Y_in, CP_in, X_in, T_in[:, None] * X_in])
    beta_fe, _, _, _ = np.linalg.lstsq(A, y_in, rcond=None)
    n_h = H_in.shape[1]
    n_y = Y_in.shape[1]
    fe_hat = H_in @ beta_fe[1 : 1 + n_h] + Y_in @ beta_fe[1 + n_h : 1 + n_h + n_y]
    y_adj = y_in - fe_hat

    mask1 = T_in == 1
    mask0 = T_in == 0

    m0 = HistGradientBoostingRegressor(
        max_iter=80,
        max_depth=4,
        min_samples_leaf=30,
        learning_rate=0.05,
        l2_regularization=1.0,
        random_state=seed,
    )
    m1 = HistGradientBoostingRegressor(
        max_iter=80,
        max_depth=4,
        min_samples_leaf=30,
        learning_rate=0.05,
        l2_regularization=1.0,
        random_state=seed + 1,
    )
    m0.fit(X_in[mask0], y_adj[mask0])
    m1.fit(X_in[mask1], y_adj[mask1])

    d1 = y_adj[mask1] - m0.predict(X_in[mask1])
    d0 = m1.predict(X_in[mask0]) - y_adj[mask0]

    tau1 = HistGradientBoostingRegressor(
        max_iter=60,
        max_depth=3,
        min_samples_leaf=50,
        learning_rate=0.05,
        l2_regularization=2.0,
        random_state=seed + 2,
    )
    tau0 = HistGradientBoostingRegressor(
        max_iter=60,
        max_depth=3,
        min_samples_leaf=50,
        learning_rate=0.05,
        l2_regularization=2.0,
        random_state=seed + 3,
    )
    tau1.fit(X_in[mask1], d1)
    tau0.fit(X_in[mask0], d0)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_in)
    prop = LogisticRegression(max_iter=500, random_state=seed)
    prop.fit(X_std, T_in)
    g_hat = prop.predict_proba(X_std)[:, 1]

    cate = g_hat * tau0.predict(X_in) + (1.0 - g_hat) * tau1.predict(X_in)
    return np.asarray(cate, dtype=float)


cate_estimates = _fit_predict_x_learner(X_pat, H_mat, Y_mat, CP_mat, T_arr, y_arr, seed=42)

rng = np.random.default_rng(42)
n_boot = 100
n_obs = len(df)
boot_diffs = []
for b in range(n_boot):
    idx = rng.integers(0, n_obs, size=n_obs)
    cate_b = _fit_predict_x_learner(
        X_pat[idx],
        H_mat[idx],
        Y_mat[idx],
        CP_mat[idx],
        T_arr[idx],
        y_arr[idx],
        seed=100 + b * 10,
    )
    age_b = age_arr[idx]
    diff_b = float(cate_b[age_b >= 65].mean() - cate_b[age_b < 40].mean())
    boot_diffs.append(diff_b)

cate_elderly_minus_young_ci = [
    float(np.percentile(boot_diffs, 2.5)),
    float(np.percentile(boot_diffs, 97.5)),
]

# =============================================================================
# 4. HIERARCHICAL MODELING: Mixed-Effects Model (ML), ICC, AIC vs Naive OLS
# =============================================================================
fe_formula = (
    "antibiotic_duration ~ asp_active + C(year) + age + C(sex) "
    "+ comorbidity_score + immune_compromised + severity_score "
    "+ C(ward) + C(infection_type)"
)

mixed_mod = smf.mixedlm(fe_formula, data=df, groups=df["hospital_id"])
mixed_res = mixed_mod.fit(reml=False, method="lbfgs")
if not mixed_res.converged:
    mixed_res = mixed_mod.fit(reml=False, method="powell")

random_effect_variance = float(mixed_res.cov_re.iloc[0, 0])
residual_variance = float(mixed_res.scale)
icc_value = float(random_effect_variance / (random_effect_variance + residual_variance))
mixed_asp_coeff = float(mixed_res.fe_params["asp_active"])
mixed_model_aic = float(mixed_res.aic)
if np.isnan(mixed_model_aic):
    mixed_model_aic = float(-2.0 * mixed_res.llf + 2.0 * (len(mixed_res.fe_params) + 1))

naive_ols_res = smf.ols(fe_formula, data=df).fit()
naive_ols_aic = float(naive_ols_res.aic)