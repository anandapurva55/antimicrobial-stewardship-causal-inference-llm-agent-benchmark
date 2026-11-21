import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import proportional_hazard_test, logrank_test
from linearmodels.panel import PanelOLS
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. SURVIVAL ANALYSIS: KAPLAN-MEIER & COX PH (CLUSTERED BY HOSPITAL_ID)
# ==============================================================================
km_treated = KaplanMeierFitter()
km_control = KaplanMeierFitter()

mask_treated = df["asp_active"] == 1
mask_control = df["asp_active"] == 0

km_control.fit(
    durations=df.loc[mask_control, "time_to_improvement"],
    event_observed=df.loc[mask_control, "event_observed"],
    label="ASP Inactive",
)
km_treated.fit(
    durations=df.loc[mask_treated, "time_to_improvement"],
    event_observed=df.loc[mask_treated, "event_observed"],
    label="ASP Active",
)

lr_test = logrank_test(
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
cox_cat_cols = ["sex", "ward", "infection_type", "year"]

cox_base = df[cox_num_cols + ["hospital_id"]].copy()
cox_dummies = pd.get_dummies(
    df[cox_cat_cols].astype(str),
    drop_first=True,
    dtype=float,
)
cox_df = pd.concat([cox_base, cox_dummies], axis=1).dropna().copy()
cox_df["cluster_hosp"] = cox_df["hospital_id"].values

cph = CoxPHFitter()
try:
    cph.fit(
        cox_df,
        duration_col="time_to_improvement",
        event_col="event_observed",
        strata=["hospital_id"],
        cluster_col="cluster_hosp",
    )
    ph_res = proportional_hazard_test(cph, cox_df, time_transform="rank")
except Exception:
    cox_hosp_dummies = pd.get_dummies(
        cox_df["hospital_id"].astype(str),
        prefix="hosp",
        drop_first=True,
        dtype=float,
    )
    cox_df_fallback = pd.concat(
        [cox_df.drop(columns=["hospital_id"]), cox_hosp_dummies], axis=1
    )
    cph = CoxPHFitter()
    cph.fit(
        cox_df_fallback,
        duration_col="time_to_improvement",
        event_col="event_observed",
        cluster_col="cluster_hosp",
    )
    ph_res = proportional_hazard_test(cph, cox_df_fallback, time_transform="rank")

cox_hr_asp = float(cph.hazard_ratios_["asp_active"])
cox_se_asp = float(cph.standard_errors_["asp_active"])

ph_test_pvalues = {}
for idx_key, pval in ph_res.summary["p"].items():
    cov_name = idx_key[0] if isinstance(idx_key, tuple) else str(idx_key)
    ph_test_pvalues[cov_name] = float(pval)

# ==============================================================================
# 2. CAUSAL IMPACT (DiD): PANEL, PARALLEL PRE-TRENDS, TWFE, STAGGERED DiD
# ==============================================================================
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
did_panel["asp_active"] = did_panel["asp_active"].round().astype(int)

# Plain TWFE on the hospital-year panel
panel_indexed = did_panel.set_index(["hospital_id", "year"])
twfe_mod = PanelOLS(
    panel_indexed["antibiotic_duration"],
    panel_indexed[["asp_active"]],
    entity_effects=True,
    time_effects=True,
)
twfe_res = twfe_mod.fit(cov_type="clustered", cluster_entity=True)
twfe_att_estimate = float(twfe_res.params["asp_active"])

# Staggered-adoption-robust estimator (Sun-Abraham / Callaway-Sant'Anna with never-treated controls)
treated_cohorts = sorted(
    [int(g) for g in did_panel["asp_adoption_year"].unique() if g < 9999]
)
all_years = sorted([int(y) for y in did_panel["year"].unique()])
all_hospitals = sorted([int(h) for h in did_panel["hospital_id"].unique()])

hosp_cohort = did_panel.groupby("hospital_id")["asp_adoption_year"].first()
cohort_counts = {g: int((hosp_cohort == g).sum()) for g in treated_cohorts}

X_sa = pd.DataFrame(index=did_panel.index)
X_sa["const"] = 1.0
for h in all_hospitals[1:]:
    X_sa[f"hosp_{h}"] = (did_panel["hospital_id"] == h).astype(float)
for y in all_years[1:]:
    X_sa[f"year_{y}"] = (did_panel["year"] == y).astype(float)

cell_cols = {}
for g in treated_cohorts:
    for y in all_years:
        e = y - g
        if e == -1:
            continue
        col_name = f"g{g}_e{e}"
        X_sa[col_name] = (
            (did_panel["asp_adoption_year"] == g) & (did_panel["year"] == y)
        ).astype(float)
        cell_cols[(g, e)] = col_name

sa_res = sm.OLS(did_panel["antibiotic_duration"].astype(float), X_sa).fit(
    cov_type="cluster",
    cov_kwds={"groups": did_panel["hospital_id"].values},
    use_t=True,
)

sa_params = sa_res.params.values
sa_vcov = sa_res.cov_params().values
col_to_idx = {c: i for i, c in enumerate(X_sa.columns)}
df_clust = len(all_hospitals) - 1
t_crit = float(stats.t.ppf(0.975, df=df_clust))

event_times = sorted(set(e for (_, e) in cell_cols.keys()))
es_records = []
es_pvalues = {}

for e in event_times:
    cohorts_for_e = [g for (g, e_) in cell_cols.keys() if e_ == e]
    tot_n = sum(cohort_counts[g] for g in cohorts_for_e)
    w = np.zeros(len(X_sa.columns), dtype=float)
    for g in cohorts_for_e:
        w[col_to_idx[cell_cols[(g, e)]]] = cohort_counts[g] / tot_n

    coef_e = float(np.dot(w, sa_params))
    var_e = float(w @ sa_vcov @ w)
    se_e = float(np.sqrt(max(var_e, 0.0)))
    ci_low_e = coef_e - t_crit * se_e
    ci_high_e = coef_e + t_crit * se_e
    t_stat_e = coef_e / se_e if se_e > 0 else 0.0
    pval_e = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat_e), df=df_clust)))

    es_records.append(
        {
            "event_time": int(e),
            "coef": coef_e,
            "ci_low": float(ci_low_e),
            "ci_high": float(ci_high_e),
        }
    )
    es_pvalues[int(e)] = pval_e

event_study = pd.DataFrame(es_records)[["event_time", "coef", "ci_low", "ci_high"]]

# Pre-trend test from event study (primary lead event_time = -2, identifiable for all cohorts)
lead_e = -2 if -2 in es_pvalues else min(e for e in event_times if e < -1)
lead_row = event_study.loc[event_study["event_time"] == lead_e].iloc[0]
pre_trend_coeff = float(lead_row["coef"])
pre_trend_pvalue = float(es_pvalues[lead_e])
pre_trend_ci = [float(lead_row["ci_low"]), float(lead_row["ci_high"])]

# Overall staggered ATT (Callaway-Sant'Anna simple post-treatment ATT using never-treated controls)
post_cells = [(g, e) for (g, e) in cell_cols.keys() if e >= 0]
tot_post_weight = sum(cohort_counts[g] for (g, _) in post_cells)
w_att = np.zeros(len(X_sa.columns), dtype=float)
for g, e in post_cells:
    w_att[col_to_idx[cell_cols[(g, e)]]] = cohort_counts[g] / tot_post_weight

did_att_estimate = float(np.dot(w_att, sa_params))

# ==============================================================================
# 3. TREATMENT HETEROGENEITY: HOSPITAL & YEAR ADJUSTED X-LEARNER + BOOTSTRAP CI
# ==============================================================================
patient_num_cols = ["age", "comorbidity_score", "immune_compromised", "severity_score"]
patient_cat_cols = ["sex", "ward", "infection_type"]

X_pat_df = pd.concat(
    [
        df[patient_num_cols].astype(float).reset_index(drop=True),
        pd.get_dummies(
            df[patient_cat_cols].astype(str).reset_index(drop=True),
            drop_first=True,
            dtype=float,
        ),
    ],
    axis=1,
)
X_pat = X_pat_df.values

fe_dummies_df = pd.concat(
    [
        pd.get_dummies(
            df["hospital_id"].astype(str).reset_index(drop=True),
            prefix="hosp",
            drop_first=True,
            dtype=float,
        ),
        pd.get_dummies(
            df["year"].astype(str).reset_index(drop=True),
            prefix="yr",
            drop_first=True,
            dtype=float,
        ),
    ],
    axis=1,
)
n_fe_cols = fe_dummies_df.shape[1]
Z_fe = np.column_stack([np.ones(len(df), dtype=float), fe_dummies_df.values, X_pat])
fe_slice = slice(1, 1 + n_fe_cols)

W_arr = df["asp_active"].values.astype(int)
Y_arr = df["antibiotic_duration"].values.astype(float)
age_arr = df["age"].values.astype(float)


def fit_predict_xlearner(X_tr, Z_tr, W_tr, Y_tr, X_ev, seed=42):
    untreated = W_tr == 0
    treated = W_tr == 1

    # Adjust for hospital and year fixed effects estimated on untreated observations
    beta_fe, _, _, _ = np.linalg.lstsq(Z_tr[untreated], Y_tr[untreated], rcond=None)
    fe_hat = Z_tr[:, fe_slice] @ beta_fe[fe_slice]
    Y_adj = Y_tr - fe_hat

    # Stage 1: outcome models mu_0 and mu_1
    m0 = HistGradientBoostingRegressor(
        max_iter=60,
        max_depth=4,
        min_samples_leaf=40,
        learning_rate=0.06,
        l2_regularization=1.0,
        random_state=seed,
    )
    m1 = HistGradientBoostingRegressor(
        max_iter=60,
        max_depth=4,
        min_samples_leaf=40,
        learning_rate=0.06,
        l2_regularization=1.0,
        random_state=seed + 1,
    )
    m0.fit(X_tr[untreated], Y_adj[untreated])
    m1.fit(X_tr[treated], Y_adj[treated])

    # Stage 2: imputed treatment effects and CATE models tau_0 and tau_1
    d1 = Y_adj[treated] - m0.predict(X_tr[treated])
    d0 = m1.predict(X_tr[untreated]) - Y_adj[untreated]

    tau1 = HistGradientBoostingRegressor(
        max_iter=60,
        max_depth=4,
        min_samples_leaf=40,
        learning_rate=0.06,
        l2_regularization=1.0,
        random_state=seed + 2,
    )
    tau0 = HistGradientBoostingRegressor(
        max_iter=60,
        max_depth=4,
        min_samples_leaf=40,
        learning_rate=0.06,
        l2_regularization=1.0,
        random_state=seed + 3,
    )
    tau1.fit(X_tr[treated], d1)
    tau0.fit(X_tr[untreated], d0)

    # Stage 3: propensity-weighted X-learner CATE
    ps_mod = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    ps_mod.fit(X_tr, W_tr)
    e_hat = np.clip(ps_mod.predict_proba(X_ev)[:, 1], 0.05, 0.95)

    return e_hat * tau0.predict(X_ev) + (1.0 - e_hat) * tau1.predict(X_ev)


cate_estimates = np.asarray(
    fit_predict_xlearner(X_pat, Z_fe, W_arr, Y_arr, X_pat, seed=42),
    dtype=float,
)

# Bootstrap 95% CI for mean CATE(age >= 65) - mean CATE(age < 40)
n_boot = 100
n_obs = len(df)
rng = np.random.default_rng(42)
boot_diffs = np.empty(n_boot, dtype=float)

for b in range(n_boot):
    b_idx = rng.integers(0, n_obs, size=n_obs)
    tau_b = fit_predict_xlearner(
        X_pat[b_idx],
        Z_fe[b_idx],
        W_arr[b_idx],
        Y_arr[b_idx],
        X_pat[b_idx],
        seed=1000 + b * 10,
    )
    age_b = age_arr[b_idx]
    boot_diffs[b] = float(np.mean(tau_b[age_b >= 65]) - np.mean(tau_b[age_b < 40]))

cate_elderly_minus_young_ci = [
    float(np.percentile(boot_diffs, 2.5)),
    float(np.percentile(boot_diffs, 97.5)),
]

# ==============================================================================
# 4. HIERARCHICAL MODELING: MIXED-EFFECTS LINEAR MODEL & ICC VS NAIVE OLS
# ==============================================================================
mixed_formula = (
    "antibiotic_duration ~ asp_active + age + C(sex) + comorbidity_score "
    "+ immune_compromised + severity_score + C(ward) + C(infection_type) + C(year)"
)

mixed_mod = smf.mixedlm(mixed_formula, data=df, groups=df["hospital_id"])
mixed_res = mixed_mod.fit(reml=False, method="lbfgs")
if not mixed_res.converged:
    mixed_res = mixed_mod.fit(reml=False, method=["lbfgs", "powell", "nm"])

ols_res = smf.ols(mixed_formula, data=df).fit()

random_effect_variance = float(mixed_res.cov_re.iloc[0, 0])
residual_variance = float(mixed_res.scale)
icc_value = float(random_effect_variance / (random_effect_variance + residual_variance))
mixed_model_aic = float(mixed_res.aic)
naive_ols_aic = float(ols_res.aic)
mixed_asp_coeff = float(mixed_res.fe_params["asp_active"])