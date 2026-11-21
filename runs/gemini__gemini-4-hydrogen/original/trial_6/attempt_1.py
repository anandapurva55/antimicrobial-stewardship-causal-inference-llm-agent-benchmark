import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from linearmodels.panel import PanelOLS
from sklearn.ensemble import GradientBoostingRegressor

# ==============================================================================
# 1. SURVIVAL ANALYSIS: Kaplan-Meier & Cox PH (hospital-clustered SEs) + PH Test
# ==============================================================================
kmf_control = KaplanMeierFitter()
mask_ctrl = df['asp_active'] == 0
kmf_control.fit(
    durations=df.loc[mask_ctrl, 'time_to_improvement'],
    event_observed=df.loc[mask_ctrl, 'event_observed'],
    label='No ASP'
)

kmf_treated = KaplanMeierFitter()
mask_trt = df['asp_active'] == 1
kmf_treated.fit(
    durations=df.loc[mask_trt, 'time_to_improvement'],
    event_observed=df.loc[mask_trt, 'event_observed'],
    label='ASP Active'
)

cox_cat_cols = ['sex', 'ward', 'infection_type', 'hospital_size', 'region']
cox_num_cols = ['asp_active', 'age', 'comorbidity_score', 'immune_compromised', 'severity_score']

df_cox_dummies = pd.get_dummies(
    df[cox_cat_cols].assign(year=df['year'].astype(str)),
    drop_first=True,
    dtype=float
)
df_cox = pd.concat(
    [
        df[['time_to_improvement', 'event_observed', 'hospital_id'] + cox_num_cols].astype(float),
        df_cox_dummies
    ],
    axis=1
)

cph = CoxPHFitter()
try:
    cph.fit(
        df_cox,
        duration_col='time_to_improvement',
        event_col='event_observed',
        cluster_col='hospital_id',
        robust=True
    )
except Exception:
    cph = CoxPHFitter(penalizer=1e-4)
    cph.fit(
        df_cox,
        duration_col='time_to_improvement',
        event_col='event_observed',
        cluster_col='hospital_id',
        robust=True
    )

cox_hr_asp = float(cph.hazard_ratios_['asp_active'])
cox_se_asp = float(cph.standard_errors_['asp_active'])

ph_res = proportional_hazard_test(cph, df_cox, time_transform='rank')
ph_test_pvalues = {
    str(k[0] if isinstance(k, tuple) else k): float(v)
    for k, v in ph_res.summary['p'].items()
}

# ==============================================================================
# 2. CAUSAL IMPACT (DiD): Hospital-Year Panel, TWFE, Staggered DiD & Event Study
# ==============================================================================
did_panel = (
    df.groupby(['hospital_id', 'year'], as_index=False)
    .agg(
        antibiotic_duration=('antibiotic_duration', 'mean'),
        asp_active=('asp_active', 'max'),
        asp_adoption_year=('asp_adoption_year', 'first')
    )
    .sort_values(['hospital_id', 'year'])
    .reset_index(drop=True)
)

panel_indexed = did_panel.set_index(['hospital_id', 'year'])
twfe_mod = PanelOLS.from_formula(
    'antibiotic_duration ~ asp_active + EntityEffects + TimeEffects',
    data=panel_indexed
)
twfe_res = twfe_mod.fit(cov_type='clustered', cluster_entity=True)
twfe_att_estimate = float(twfe_res.params['asp_active'])

# Sun-Abraham / Callaway-Sant'Anna equivalent saturated cohort x event-time DiD
# using never-treated hospitals (asp_adoption_year == 9999) as clean controls
sa_df = did_panel.copy()
treated_cohorts = sorted([int(g) for g in sa_df['asp_adoption_year'].unique() if g < 9999])
cohort_sizes = {
    g: int(sa_df.loc[sa_df['asp_adoption_year'] == g, 'hospital_id'].nunique())
    for g in treated_cohorts
}

cohort_event_cols = []
event_time_map = {}

for g in treated_cohorts:
    years_for_g = sorted(sa_df['year'].unique())
    for y in years_for_g:
        e = int(y - g)
        if e == -1:
            continue
        e_label = f'em{abs(e)}' if e < 0 else f'e{e}'
        col_name = f'd_g{g}_{e_label}'
        sa_df[col_name] = ((sa_df['asp_adoption_year'] == g) & (sa_df['year'] == y)).astype(float)
        cohort_event_cols.append(col_name)
        event_time_map.setdefault(e, []).append((g, col_name))

sa_formula = 'antibiotic_duration ~ ' + ' + '.join(cohort_event_cols) + ' + C(hospital_id) + C(year)'
sa_res = smf.ols(sa_formula, data=sa_df).fit(
    cov_type='cluster',
    cov_kwds={'groups': sa_df['hospital_id']}
)

param_names = list(sa_res.params.index)
event_study_rows = []
pre_trend_stats = {}

for e in sorted(event_time_map.keys()):
    pairs = event_time_map[e]
    total_w = sum(cohort_sizes[g] for g, _ in pairs)
    contrast = np.zeros(len(param_names), dtype=float)
    for g, col_name in pairs:
        contrast[param_names.index(col_name)] = cohort_sizes[g] / total_w

    tt = sa_res.t_test(contrast)
    coef_e = float(np.asarray(tt.effect).item())
    pval_e = float(np.asarray(tt.pvalue).item())
    ci_e = np.asarray(tt.conf_int(alpha=0.05)).reshape(-1)
    lo_e, hi_e = float(ci_e[0]), float(ci_e[1])

    event_study_rows.append({
        'event_time': int(e),
        'coef': coef_e,
        'ci_low': lo_e,
        'ci_high': hi_e
    })
    if e < -1:
        pre_trend_stats[e] = (coef_e, pval_e, [lo_e, hi_e])

event_study = pd.DataFrame(event_study_rows).sort_values('event_time').reset_index(drop=True)

# Pre-treatment lead immediately prior to reference period (-1)
lead_key = -2 if -2 in pre_trend_stats else max(pre_trend_stats.keys())
pre_trend_coeff, pre_trend_pvalue, pre_trend_ci = pre_trend_stats[lead_key]

# Overall staggered ATT across all post-treatment (e >= 0) hospital-year cells
post_pairs = [
    (g, col_name)
    for e, pairs in event_time_map.items()
    if e >= 0
    for g, col_name in pairs
]
total_post_w = sum(cohort_sizes[g] for g, _ in post_pairs)
att_contrast = np.zeros(len(param_names), dtype=float)
for g, col_name in post_pairs:
    att_contrast[param_names.index(col_name)] = cohort_sizes[g] / total_post_w

att_tt = sa_res.t_test(att_contrast)
did_att_estimate = float(np.asarray(att_tt.effect).item())

# ==============================================================================
# 3. TREATMENT HETEROGENEITY: T-Learner (GBDT) Adjusted for Hospital & Year
# ==============================================================================
cate_patient_num = ['age', 'comorbidity_score', 'immune_compromised', 'severity_score']
cate_patient_cat = ['sex', 'ward', 'infection_type']

X_patient_df = pd.concat(
    [
        df[cate_patient_num].astype(float),
        pd.get_dummies(df[cate_patient_cat], drop_first=True, dtype=float)
    ],
    axis=1
)
X_hy_df = pd.get_dummies(
    df[['hospital_id', 'year']].astype(str),
    drop_first=True,
    dtype=float
)

X_patient_arr = X_patient_df.values
X_hy_arr = X_hy_df.values
w_arr = df['asp_active'].values.astype(int)
y_arr = df['antibiotic_duration'].values.astype(float)
age_arr = df['age'].values

X_fe_full = np.column_stack([
    np.ones(len(df), dtype=float),
    w_arr.astype(float),
    X_hy_arr,
    X_patient_arr
])
hy_slice = slice(2, 2 + X_hy_arr.shape[1])

def fit_t_learner_cate(indices, n_estimators=150, learning_rate=0.05, seed=42):
    X_fe_sub = X_fe_full[indices]
    y_sub = y_arr[indices]
    w_sub = w_arr[indices]
    X_pat_sub = X_patient_arr[indices]

    beta_fe, *_ = np.linalg.lstsq(X_fe_sub, y_sub, rcond=None)
    y_adj = y_sub - X_fe_sub[:, hy_slice] @ beta_fe[hy_slice]

    m0 = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=3,
        learning_rate=learning_rate,
        subsample=0.8,
        random_state=seed
    )
    m1 = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=3,
        learning_rate=learning_rate,
        subsample=0.8,
        random_state=seed
    )

    m0.fit(X_pat_sub[w_sub == 0], y_adj[w_sub == 0])
    m1.fit(X_pat_sub[w_sub == 1], y_adj[w_sub == 1])
    return m1.predict(X_pat_sub) - m0.predict(X_pat_sub)

all_idx = np.arange(len(df))
cate_estimates = np.asarray(
    fit_t_learner_cate(all_idx, n_estimators=150, learning_rate=0.05, seed=42),
    dtype=float
).ravel()

rng = np.random.default_rng(42)
n_boot = 100
boot_diffs = np.empty(n_boot, dtype=float)
n_obs = len(df)

for b in range(n_boot):
    b_idx = rng.integers(0, n_obs, size=n_obs)
    cate_b = fit_t_learner_cate(b_idx, n_estimators=80, learning_rate=0.08, seed=100 + b)
    age_b = age_arr[b_idx]
    boot_diffs[b] = float(np.mean(cate_b[age_b >= 65]) - np.mean(cate_b[age_b < 40]))

ci_boot = np.percentile(boot_diffs, [2.5, 97.5])
cate_elderly_minus_young_ci = [float(ci_boot[0]), float(ci_boot[1])]

# ==============================================================================
# 4. HIERARCHICAL MODELING: Mixed-Effects Linear Model vs Naive OLS
# ==============================================================================
mixed_formula = (
    'antibiotic_duration ~ asp_active + age + C(sex) + comorbidity_score + '
    'immune_compromised + severity_score + C(infection_type) + C(ward) + C(year)'
)

mixed_mod = smf.mixedlm(mixed_formula, data=df, groups=df['hospital_id'])
mixed_res = mixed_mod.fit(reml=False)
if not mixed_res.converged:
    mixed_res = mixed_mod.fit(reml=False, method='lbfgs')

random_effect_variance = float(mixed_res.cov_re.iloc[0, 0])
residual_variance = float(mixed_res.scale)
icc_value = float(random_effect_variance / (random_effect_variance + residual_variance))
mixed_model_aic = float(mixed_res.aic)
mixed_asp_coeff = float(mixed_res.fe_params['asp_active'])

naive_ols_res = smf.ols(mixed_formula, data=df).fit()
naive_ols_aic = float(naive_ols_res.aic)