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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# 1. SURVIVAL ANALYSIS (Kaplan-Meier + Cox PH with hospital-clustered SEs)
# ==============================================================================
kmf_control = KaplanMeierFitter()
kmf_treated = KaplanMeierFitter()
mask_treated_surv = df["asp_active"] == 1
kmf_control.fit(
    df.loc[~mask_treated_surv, "time_to_improvement"],
    event_observed=df.loc[~mask_treated_surv, "event_observed"],
    label="No ASP",
)
kmf_treated.fit(
    df.loc[mask_treated_surv, "time_to_improvement"],
    event_observed=df.loc[mask_treated_surv, "event_observed"],
    label="ASP Active",
)

cox_num_cols = [
    "asp_active",
    "age",
    "comorbidity_score",
    "immune_compromised",
    "severity_score",
]
cox_cat_cols = ["sex", "ward", "infection_type", "hospital_size", "region", "year"]
cox_dummies = pd.get_dummies(
    df[cox_cat_cols].astype(str), drop_first=True, dtype=float
)
cox_df = pd.concat(
    [
        df[["time_to_improvement", "event_observed", "hospital_id"] + cox_num_cols].astype(float),
        cox_dummies,
    ],
    axis=1,
)

cph = CoxPHFitter(penalizer=0.0)
cph.fit(
    cox_df,
    duration_col="time_to_improvement",
    event_col="event_observed",
    cluster_col="hospital_id",
)

cox_hr_asp = float(cph.hazard_ratios_["asp_active"])
cox_se_asp = float(cph.standard_errors_["asp_active"])

ph_test_res = proportional_hazard_test(cph, cox_df, time_transform="rank")
ph_test_pvalues = {
    str(cov): float(pval)
    for cov, pval in ph_test_res.summary["p"].items()
}

# ==============================================================================
# 2. CAUSAL IMPACT (DiD: Hospital-Year Panel, Pre-Trends, TWFE, Sun-Abraham)
# ==============================================================================
did_panel = (
    df.groupby(["hospital_id", "year"], as_index=False)
    .agg(
        antibiotic_duration=("antibiotic_duration", "mean"),
        asp_active=("asp_active", "mean"),
        asp_adoption_year=("asp_adoption_year", "first"),
        age=("age", "mean"),
        comorbidity_score=("comorbidity_score", "mean"),
        severity_score=("severity_score", "mean"),
        immune_compromised=("immune_compromised", "mean"),
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
).fit(cov_type="clustered", cluster_entity=True)
twfe_att_estimate = float(twfe_model.params["asp_active"])

# Staggered-adoption-robust event study & ATT (Sun-Abraham / Callaway-Sant'Anna
# with never-treated hospitals as controls and event_time = -1 as reference)
treated_cohorts = sorted(
    [int(g) for g in did_panel["asp_adoption_year"].unique() if g != 9999]
)
all_years = sorted([int(y) for y in did_panel["year"].unique()])
cohort_sizes = (
    did_panel.groupby("hospital_id")["asp_adoption_year"]
    .first()
    .value_counts()
    .to_dict()
)

sa_design = pd.DataFrame(index=did_panel.index)
sa_terms = []
for g in treated_cohorts:
    for t in all_years:
        if t == g - 1:
            continue
        col_name = f"sa_g{g}_t{t}"
        sa_design[col_name] = (
            (did_panel["asp_adoption_year"] == g) & (did_panel["year"] == t)
        ).astype(float)
        sa_terms.append((col_name, g, t, t - g))

fe_hosp = pd.get_dummies(
    did_panel["hospital_id"], prefix="hosp", drop_first=True, dtype=float
)
fe_year = pd.get_dummies(
    did_panel["year"], prefix="yr", drop_first=True, dtype=float
)
X_sa = sm.add_constant(pd.concat([sa_design, fe_hosp, fe_year], axis=1))
sa_ols = sm.OLS(did_panel["antibiotic_duration"].values, X_sa.values).fit(
    cov_type="cluster",
    cov_kwds={"groups": did_panel["hospital_id"].values, "use_t": True},
)

param_names = list(X_sa.columns)
params_vec = sa_ols.params
cov_mat = sa_ols.cov_params()
n_clusters = int(did_panel["hospital_id"].nunique())
t_crit = float(stats.t.ppf(0.975, df=n_clusters - 1))


def _weighted_linear_combo(term_list):
    weights = np.zeros(len(param_names), dtype=float)
    raw_w = [float(cohort_sizes[g]) for (_, g, _, _) in term_list]
    total_w = sum(raw_w)
    for (col_name, _, _, _), w_val in zip(term_list, raw_w):
        idx = param_names.index(col_name)
        weights[idx] = w_val / total_w
    est = float(weights @ params_vec)
    se = float(np.sqrt(max(float(weights @ cov_mat @ weights), 0.0)))
    ci_lo = float(est - t_crit * se)
    ci_hi = float(est + t_crit * se)
    t_stat = est / se if se > 0 else 0.0
    pval = float(2.0 * stats.t.sf(abs(t_stat), df=n_clusters - 1))
    return est, se, ci_lo, ci_hi, pval


event_times_avail = sorted(list({e for (_, _, _, e) in sa_terms}))
es_rows = []
for e in event_times_avail:
    terms_e = [item for item in sa_terms if item[3] == e]
    est_e, _, lo_e, hi_e, _ = _weighted_linear_combo(terms_e)
    es_rows.append(
        {
            "event_time": int(e),
            "coef": est_e,
            "ci_low": lo_e,
            "ci_high": hi_e,
        }
    )
event_study = pd.DataFrame(es_rows).sort_values("event_time").reset_index(drop=True)

pre_terms = [item for item in sa_terms if item[3] < -1]
pre_trend_coeff, _, pre_lo, pre_hi, pre_trend_pvalue = _weighted_linear_combo(pre_terms)
pre_trend_ci = [pre_lo, pre_hi]

post_terms = [item for item in sa_terms if item[3] >= 0]
did_att_estimate, _, _, _, _ = _weighted_linear_combo(post_terms)

# ==============================================================================
# 3. TREATMENT HETEROGENEITY (X-Learner with Random Forests + Bootstrap CI)
# ==============================================================================
cate_num_cols = ["age", "comorbidity_score", "immune_compromised", "severity_score"]
cate_cat_cols = ["sex", "ward", "infection_type", "hospital_size", "region"]
X_pat_df = pd.concat(
    [
        df[cate_num_cols].astype(float),
        pd.get_dummies(df[cate_cat_cols].astype(str), drop_first=True, dtype=float),
    ],
    axis=1,
)
X_pat_cols = list(X_pat_df.columns)
X_pat_all = X_pat_df.values
H_all = pd.get_dummies(df["hospital_id"], prefix="h", drop_first=True, dtype=float).values
T_all = pd.get_dummies(df["year"], prefix="t", drop_first=True, dtype=float).values
W_all = df["asp_active"].values.astype(int)
Y_all = df["antibiotic_duration"].values.astype(float)
age_all = df["age"].values.astype(float)


def _fit_xlearner(X_pat, H_mat, T_mat, W, Y, n_trees=80, seed=42):
    # Partial out hospital and year fixed effects controlling for patient mix and treatment
    age_c = X_pat[:, 0] - np.mean(X_pat[:, 0])
    com_c = X_pat[:, 1] - np.mean(X_pat[:, 1])
    sev_c = X_pat[:, 3] - np.mean(X_pat[:, 3])
    Z = np.column_stack(
        [
            np.ones(len(Y)),
            H_mat,
            T_mat,
            X_pat,
            W,
            W * age_c,
            W * com_c,
            W * sev_c,
        ]
    )
    beta, _, _, _ = np.linalg.lstsq(Z, Y, rcond=None)
    k_h = H_mat.shape[1]
    k_t = T_mat.shape[1]
    fe_hat = H_mat @ beta[1 : 1 + k_h] + T_mat @ beta[1 + k_h : 1 + k_h + k_t]
    Y_adj = Y - fe_hat

    mask1 = W == 1
    mask0 = W == 0

    mu0 = RandomForestRegressor(
        n_estimators=n_trees,
        max_depth=8,
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=seed,
    )
    mu1 = RandomForestRegressor(
        n_estimators=n_trees,
        max_depth=8,
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=seed + 1,
    )
    mu0.fit(X_pat[mask0], Y_adj[mask0])
    mu1.fit(X_pat[mask1], Y_adj[mask1])

    d1 = Y_adj[mask1] - mu0.predict(X_pat[mask1])
    d0 = mu1.predict(X_pat[mask0]) - Y_adj[mask0]

    tau1 = RandomForestRegressor(
        n_estimators=n_trees,
        max_depth=8,
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=seed + 2,
    )
    tau0 = RandomForestRegressor(
        n_estimators=n_trees,
        max_depth=8,
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=seed + 3,
    )
    tau1.fit(X_pat[mask1], d1)
    tau0.fit(X_pat[mask0], d0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pat)
    prop_model = LogisticRegression(max_iter=1000, random_state=seed)
    prop_model.fit(X_scaled, W)
    e_hat = np.clip(prop_model.predict_proba(X_scaled)[:, 1], 0.05, 0.95)

    cate = e_hat * tau0.predict(X_pat) + (1.0 - e_hat) * tau1.predict(X_pat)
    return cate


cate_estimates = _fit_xlearner(X_pat_all, H_all, T_all, W_all, Y_all, n_trees=100, seed=42)

# Stratified bootstrap by (hospital_id, year) to preserve staggered adoption structure
rng = np.random.RandomState(42)
cell_groups = df.groupby(["hospital_id", "year"]).indices
cell_idx_arrays = list(cell_groups.values())
n_boot = 50
boot_diffs = np.empty(n_boot, dtype=float)

for b in range(n_boot):
    sampled_parts = [
        rng.choice(idx_arr, size=len(idx_arr), replace=True)
        for idx_arr in cell_idx_arrays
    ]
    b_idx = np.concatenate(sampled_parts)
    cate_b = _fit_xlearner(
        X_pat_all[b_idx],
        H_all[b_idx],
        T_all[b_idx],
        W_all[b_idx],
        Y_all[b_idx],
        n_trees=40,
        seed=1000 + b * 10,
    )
    age_b = age_all[b_idx]
    boot_diffs[b] = float(
        np.mean(cate_b[age_b >= 65]) - np.mean(cate_b[age_b < 40])
    )

cate_elderly_minus_young_ci = [
    float(np.percentile(boot_diffs, 2.5)),
    float(np.percentile(boot_diffs, 97.5)),
]

# ==============================================================================
# 4. HIERARCHICAL MODELING (Random-Intercept Mixed Model vs Naive OLS)
# ==============================================================================
df_mixed = df.copy()
for col in ["age", "comorbidity_score", "severity_score"]:
    df_mixed[f"{col}_c"] = df_mixed[col] - df_mixed[col].mean()

mixed_formula = (
    "antibiotic_duration ~ asp_active + C(year) + age_c + C(sex) + "
    "comorbidity_score_c + immune_compromised + C(infection_type) + "
    "severity_score_c + C(ward) + C(hospital_size) + C(region)"
)

mixed_mod = smf.mixedlm(mixed_formula, data=df_mixed, groups=df_mixed["hospital_id"])
mixed_res = mixed_mod.fit(reml=False, method="lbfgs")
if not mixed_res.converged:
    mixed_res = mixed_mod.fit(reml=False, method="powell")

ols_res = smf.ols(mixed_formula, data=df_mixed).fit()

random_effect_variance = float(mixed_res.cov_re.iloc[0, 0])
residual_variance = float(mixed_res.scale)
icc_value = float(
    random_effect_variance / (random_effect_variance + residual_variance)
)
mixed_model_aic = float(mixed_res.aic)
naive_ols_aic = float(ols_res.aic)
mixed_asp_coeff = float(mixed_res.fe_params["asp_active"])