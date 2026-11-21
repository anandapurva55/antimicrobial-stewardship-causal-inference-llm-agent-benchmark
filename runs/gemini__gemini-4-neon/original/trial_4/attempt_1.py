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

# =============================================================================
# 1. SURVIVAL ANALYSIS: Kaplan-Meier & Cox PH (Clustered by hospital_id)
# =============================================================================
kmf_treated = KaplanMeierFitter()
kmf_control = KaplanMeierFitter()

mask_asp_1 = df['asp_active'] == 1
mask_asp_0 = df['asp_active'] == 0

kmf_control.fit(
    durations=df.loc[mask_asp_0, 'time_to_improvement'],
    event_observed=df.loc[mask_asp_0, 'event_observed'],
    label='ASP Inactive',
)
kmf_treated.fit(
    durations=df.loc[mask_asp_1, 'time_to_improvement'],
    event_observed=df.loc[mask_asp_1, 'event_observed'],
    label='ASP Active',
)

lr_test = logrank_test(
    df.loc[mask_asp_1, 'time_to_improvement'],
    df.loc[mask_asp_0, 'time_to_improvement'],
    event_observed_A=df.loc[mask_asp_1, 'event_observed'],
    event_observed_B=df.loc[mask_asp_0, 'event_observed'],
)

cox_base_cols = [
    'asp_active',
    'age',
    'comorbidity_score',
    'immune_compromised',
    'severity_score',
]
cox_cat_cols = ['sex', 'ward', 'infection_type', 'year']

cox_X = pd.get_dummies(
    df[cox_base_cols + cox_cat_cols],
    columns=cox_cat_cols,
    drop_first=True,
    dtype=float,
)
cox_df = pd.concat(
    [
        df[['time_to_improvement', 'event_observed']].reset_index(drop=True),
        cox_X.reset_index(drop=True),
    ],
    axis=1,
)
cox_df['strata_hospital'] = df['hospital_id'].values
cox_df['cluster_hospital'] = df['hospital_id'].values

try:
    cph = CoxPHFitter()
    cph.fit(
        cox_df,
        duration_col='time_to_improvement',
        event_col='event_observed',
        strata=['strata_hospital'],
        cluster_col='cluster_hospital',
    )
    ph_res = proportional_hazard_test(cph, cox_df, time_transform='rank')
    ph_pvals_raw = ph_res.summary['p'].to_dict()
    if any(pd.isna(v) for v in ph_pvals_raw.values()):
        raise ValueError("NaN in PH test p-values with stratified Cox")
except Exception:
    cox_X_fb = pd.get_dummies(
        df[cox_base_cols + cox_cat_cols + ['hospital_size', 'region']],
        columns=cox_cat_cols + ['hospital_size', 'region'],
        drop_first=True,
        dtype=float,
    )
    cox_df_fb = pd.concat(
        [
            df[['time_to_improvement', 'event_observed']].reset_index(drop=True),
            cox_X_fb.reset_index(drop=True),
        ],
        axis=1,
    )
    cox_df_fb['cluster_hospital'] = df['hospital_id'].values
    cph = CoxPHFitter()
    cph.fit(
        cox_df_fb,
        duration_col='time_to_improvement',
        event_col='event_observed',
        cluster_col='cluster_hospital',
    )
    ph_res = proportional_hazard_test(cph, cox_df_fb, time_transform='rank')
    ph_pvals_raw = ph_res.summary['p'].to_dict()

cox_hr_asp = float(cph.hazard_ratios_['asp_active'])
cox_se_asp = float(cph.standard_errors_['asp_active'])
ph_test_pvalues = {str(k): float(v) for k, v in ph_pvals_raw.items()}

# =============================================================================
# 2. CAUSAL IMPACT (DiD): Hospital-Year Panel, TWFE, Sun-Abraham / CS Event Study
# =============================================================================
did_panel = (
    df.groupby(['hospital_id', 'year'], as_index=False)
    .agg(
        antibiotic_duration=('antibiotic_duration', 'mean'),
        asp_active=('asp_active', 'first'),
        asp_adoption_year=('asp_adoption_year', 'first'),
        n_patients=('patient_id', 'count'),
    )
    .sort_values(['hospital_id', 'year'])
    .reset_index(drop=True)
)

n_hospitals = int(did_panel['hospital_id'].nunique())
t_crit_did = float(stats.t.ppf(0.975, df=n_hospitals - 1))

# Plain TWFE on hospital-year panel
panel_idx = did_panel.set_index(['hospital_id', 'year'])
twfe_mod = PanelOLS(
    panel_idx['antibiotic_duration'],
    panel_idx[['asp_active']],
    entity_effects=True,
    time_effects=True,
)
twfe_res = twfe_mod.fit(cov_type='clustered', cluster_entity=True)
twfe_att_estimate = float(twfe_res.params['asp_active'])

# Staggered-adoption-robust estimator (Sun-Abraham / Callaway-Sant'Anna
# interaction-weighted estimator using never-treated hospitals as controls)
treated_cohorts = sorted(
    int(g) for g in did_panel['asp_adoption_year'].unique() if g < 9000
)
cohort_sizes = (
    did_panel.groupby('asp_adoption_year')['hospital_id'].nunique().to_dict()
)

sa_df = did_panel.copy()
sa_cols = []
cell_meta = []

for g in treated_cohorts:
    sub_years = sorted(sa_df.loc[sa_df['asp_adoption_year'] == g, 'year'].unique())
    for yr in sub_years:
        e = int(yr - g)
        if e == -1:
            continue
        col_name = f"d_{g}_{e}"
        sa_df[col_name] = (
            (sa_df['asp_adoption_year'] == g) & (sa_df['year'] == yr)
        ).astype(float)
        sa_cols.append(col_name)
        cell_meta.append((col_name, g, e))

sa_panel_idx = sa_df.set_index(['hospital_id', 'year'])
sa_mod = PanelOLS(
    sa_panel_idx['antibiotic_duration'],
    sa_panel_idx[sa_cols],
    entity_effects=True,
    time_effects=True,
)
sa_res = sa_mod.fit(cov_type='clustered', cluster_entity=True)

sa_params = sa_res.params.loc[sa_cols].values
sa_cov = sa_res.cov.loc[sa_cols, sa_cols].values

# Aggregate to event-study estimates weighted by cohort hospital counts
unique_event_times = sorted(set(e for _, _, e in cell_meta))
es_rows = []
es_pvalues = {}

for e in unique_event_times:
    w = np.zeros(len(sa_cols), dtype=float)
    for idx_c, (_, g, e_c) in enumerate(cell_meta):
        if e_c == e:
            w[idx_c] = float(cohort_sizes[g])
    w = w / w.sum()
    coef = float(w @ sa_params)
    se = float(np.sqrt(max(float(w @ sa_cov @ w), 1e-18)))
    ci_low = float(coef - t_crit_did * se)
    ci_high = float(coef + t_crit_did * se)
    pval = float(2.0 * (1.0 - stats.t.cdf(abs(coef / se), df=n_hospitals - 1)))
    es_rows.append(
        {
            'event_time': int(e),
            'coef': coef,
            'ci_low': ci_low,
            'ci_high': ci_high,
        }
    )
    es_pvalues[int(e)] = pval

event_study = pd.DataFrame(es_rows)[['event_time', 'coef', 'ci_low', 'ci_high']]

# Pre-trend test (primary lead event_time = -2, shared by all treated cohorts)
pre_lead_time = -2 if -2 in es_pvalues else max(e for e in unique_event_times if e < -1)
pre_row = event_study.loc[event_study['event_time'] == pre_lead_time].iloc[0]
pre_trend_coeff = float(pre_row['coef'])
pre_trend_pvalue = float(es_pvalues[pre_lead_time])
pre_trend_ci = [float(pre_row['ci_low']), float(pre_row['ci_high'])]

# Overall staggered-robust ATT (Callaway-Sant'Anna simple post-treatment ATT)
w_att = np.zeros(len(sa_cols), dtype=float)
for idx_c, (_, g, e_c) in enumerate(cell_meta):
    if e_c >= 0:
        w_att[idx_c] = float(cohort_sizes[g])
w_att = w_att / w_att.sum()
did_att_estimate = float(w_att @ sa_params)
did_att_se = float(np.sqrt(max(float(w_att @ sa_cov @ w_att), 1e-18)))

# =============================================================================
# 3. TREATMENT HETEROGENEITY: Hospital/Year-Adjusted X-Learner & Bootstrap CI
# =============================================================================
pat_num_cols = ['age', 'comorbidity_score', 'immune_compromised', 'severity_score']
pat_cat_cols = ['sex', 'ward', 'infection_type']

X_pat_df = pd.get_dummies(
    df[pat_num_cols + pat_cat_cols],
    columns=pat_cat_cols,
    drop_first=True,
    dtype=float,
)
X_pat = X_pat_df.values.astype(float)

M_fe_df = pd.get_dummies(
    df[['hospital_id', 'year']],
    columns=['hospital_id', 'year'],
    drop_first=True,
    dtype=float,
)
M_fe = M_fe_df.values.astype(float)
n_obs = len(df)
M_full = np.column_stack([np.ones(n_obs, dtype=float), M_fe, X_pat])
n_fe_cols = M_fe.shape[1]

Y_all = df['antibiotic_duration'].values.astype(float)
T_all = df['asp_active'].values.astype(int)
age_all = df['age'].values


def fit_predict_xlearner(X_tr, M_full_tr, M_fe_tr, Y_tr, T_tr, X_eval, seed=42):
    mask0 = T_tr == 0
    mask1 = T_tr == 1

    # Estimate hospital and year fixed effects on untreated observations
    # (all 15 hospitals and all 6 years are observed in T == 0)
    beta_fe_full = np.linalg.lstsq(M_full_tr[mask0], Y_tr[mask0], rcond=None)[0]
    fe_beta = beta_fe_full[1 : 1 + n_fe_cols]
    fe_hat = M_fe_tr @ fe_beta
    fe_hat = fe_hat - np.mean(fe_hat)
    Y_adj = Y_tr - fe_hat

    hgb_kwargs = dict(
        max_iter=80,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=30,
        l2_regularization=1.0,
    )

    # Stage 1: Outcome models for control and treated
    m0 = HistGradientBoostingRegressor(**hgb_kwargs, random_state=seed)
    m1 = HistGradientBoostingRegressor(**hgb_kwargs, random_state=seed + 1)
    m0.fit(X_tr[mask0], Y_adj[mask0])
    m1.fit(X_tr[mask1], Y_adj[mask1])

    # Stage 2: Imputed individual treatment effects
    d1 = Y_adj[mask1] - m0.predict(X_tr[mask1])
    d0 = m1.predict(X_tr[mask0]) - Y_adj[mask0]

    tau1 = HistGradientBoostingRegressor(**hgb_kwargs, random_state=seed + 2)
    tau0 = HistGradientBoostingRegressor(**hgb_kwargs, random_state=seed + 3)
    tau1.fit(X_tr[mask1], d1)
    tau0.fit(X_tr[mask0], d0)

    # Stage 3: Propensity-weighted CATE combination
    prop_mod = LogisticRegression(max_iter=500, C=1.0)
    prop_mod.fit(X_tr, T_tr)
    e_hat = np.clip(prop_mod.predict_proba(X_eval)[:, 1], 0.05, 0.95)

    cate = e_hat * tau0.predict(X_eval) + (1.0 - e_hat) * tau1.predict(X_eval)
    return cate


cate_estimates = np.asarray(
    fit_predict_xlearner(X_pat, M_full, M_fe, Y_all, T_all, X_pat, seed=42),
    dtype=float,
)

rng = np.random.default_rng(42)
n_boot = 100
boot_diffs = np.empty(n_boot, dtype=float)

for b in range(n_boot):
    idx_b = rng.integers(0, n_obs, size=n_obs)
    cate_b = fit_predict_xlearner(
        X_pat[idx_b],
        M_full[idx_b],
        M_fe[idx_b],
        Y_all[idx_b],
        T_all[idx_b],
        X_pat[idx_b],
        seed=1000 + b,
    )
    age_b = age_all[idx_b]
    boot_diffs[b] = float(
        np.mean(cate_b[age_b >= 65]) - np.mean(cate_b[age_b < 40])
    )

cate_elderly_minus_young_ci = [
    float(np.percentile(boot_diffs, 2.5)),
    float(np.percentile(boot_diffs, 97.5)),
]

# =============================================================================
# 4. HIERARCHICAL MODELING: Mixed-Effects Model (ML) vs Naive OLS & ICC
# =============================================================================
mixed_formula = (
    "antibiotic_duration ~ asp_active + age + C(sex) + comorbidity_score "
    "+ immune_compromised + severity_score + C(ward) + C(infection_type) + C(year)"
)

mixed_mod = smf.mixedlm(mixed_formula, data=df, groups=df['hospital_id'])
mixed_res = mixed_mod.fit(reml=False, method=['lbfgs', 'powell'])

ols_res = smf.ols(mixed_formula, data=df).fit()

random_effect_variance = float(mixed_res.cov_re.iloc[0, 0])
residual_variance = float(mixed_res.scale)
icc_value = float(
    random_effect_variance / (random_effect_variance + residual_variance)
)
mixed_asp_coeff = float(mixed_res.params['asp_active'])

mixed_model_aic = float(mixed_res.aic)
if np.isnan(mixed_model_aic):
    mixed_model_aic = float(
        -2.0 * mixed_res.llf + 2.0 * (len(mixed_res.fe_params) + 2)
    )
naive_ols_aic = float(ols_res.aic)

print(f"KM Median Time (ASP=0 vs ASP=1): {kmf_control.median_survival_time_:.2f} vs {kmf_treated.median_survival_time_:.2f}, Log-rank p={lr_test.p_value:.4g}")
print(f"Cox PH HR (asp_active): {cox_hr_asp:.4f}, Clustered SE: {cox_se_asp:.4f}, PH p-val (asp_active): {ph_test_pvalues.get('asp_active', np.nan):.4f}")
print(f"DiD Pre-trend (e={pre_lead_time}): {pre_trend_coeff:.4f}, p={pre_trend_pvalue:.4f}, 95% CI={pre_trend_ci}")
print(f"TWFE ATT: {twfe_att_estimate:.4f} days | Staggered-Robust ATT: {did_att_estimate:.4f} (SE={did_att_se:.4f}) days")
print(f"Mean CATE: {cate_estimates.mean():.4f} days | Elderly - Young 95% CI: {cate_elderly_minus_young_ci}")
print(f"MixedLM ASP Coeff: {mixed_asp_coeff:.4f} | RE Var: {random_effect_variance:.4f} | ICC: {icc_value:.4f}")
print(f"MixedLM AIC (ML): {mixed_model_aic:.2f} vs Naive OLS AIC: {naive_ols_aic:.2f}")