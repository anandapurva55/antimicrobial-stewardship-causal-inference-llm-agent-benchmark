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
# 1. SURVIVAL ANALYSIS: Kaplan-Meier, Cox PH (clustered by hospital_id), PH Test
# ==============================================================================
kmf_control = KaplanMeierFitter()
kmf_treated = KaplanMeierFitter()
mask_ctrl_surv = df['asp_active'] == 0
mask_trt_surv = df['asp_active'] == 1
kmf_control.fit(
    df.loc[mask_ctrl_surv, 'time_to_improvement'],
    event_observed=df.loc[mask_ctrl_surv, 'event_observed'],
    label='ASP Inactive'
)
kmf_treated.fit(
    df.loc[mask_trt_surv, 'time_to_improvement'],
    event_observed=df.loc[mask_trt_surv, 'event_observed'],
    label='ASP Active'
)

cox_covariates = [
    'asp_active', 'age', 'comorbidity_score', 'immune_compromised',
    'severity_score', 'year', 'sex', 'infection_type', 'ward',
    'hospital_size', 'region'
]
cox_df = pd.get_dummies(
    df[['time_to_improvement', 'event_observed', 'hospital_id'] + cox_covariates],
    columns=['year', 'sex', 'infection_type', 'ward', 'hospital_size', 'region'],
    drop_first=True,
    dtype=float
)

cph = CoxPHFitter()
cph.fit(
    cox_df,
    duration_col='time_to_improvement',
    event_col='event_observed',
    cluster_col='hospital_id',
    robust=True
)

cox_hr_asp = float(cph.hazard_ratios_['asp_active'])
cox_se_asp = float(cph.standard_errors_['asp_active'])

ph_res = proportional_hazard_test(cph, cox_df, time_transform='rank')
ph_test_pvalues = {
    str(cov): float(pval)
    for cov, pval in ph_res.summary['p'].items()
}

# ==============================================================================
# 2. CAUSAL IMPACT (DiD): Hospital-Year Panel, Pre-Trends, TWFE, Sun-Abraham
# ==============================================================================
did_panel = (
    df.groupby(['hospital_id', 'year'], as_index=False)
    .agg(
        antibiotic_duration=('antibiotic_duration', 'mean'),
        asp_active=('asp_active', 'mean'),
        asp_adoption_year=('asp_adoption_year', 'first'),
        n_patients=('patient_id', 'count')
    )
    .sort_values(['hospital_id', 'year'])
    .reset_index(drop=True)
)

panel_idx = did_panel.set_index(['hospital_id', 'year'])
twfe_mod = PanelOLS(
    panel_idx['antibiotic_duration'],
    panel_idx[['asp_active']],
    entity_effects=True,
    time_effects=True
)
twfe_res = twfe_mod.fit(cov_type='clustered', cluster_entity=True)
twfe_att_estimate = float(twfe_res.params['asp_active'])

# Sun-Abraham (2021) interaction-weighted event study & ATT with never-treated controls
treated_cohorts = sorted([int(g) for g in did_panel['asp_adoption_year'].unique() if g != 9999])
cohort_counts = (
    did_panel[did_panel['asp_adoption_year'] != 9999]
    .groupby('asp_adoption_year')['hospital_id']
    .nunique()
    .to_dict()
)

es_df = did_panel.copy()
es_df['rel_time'] = es_df['year'] - es_df['asp_adoption_year']

interaction_cols = []
cell_meta = []
for g in treated_cohorts:
    g_rel_times = sorted(es_df.loc[es_df['asp_adoption_year'] == g, 'rel_time'].unique())
    for e in g_rel_times:
        if e == -1:
            continue
        col_name = f'cohort_{g}_rel_{e}'
        es_df[col_name] = ((es_df['asp_adoption_year'] == g) & (es_df['rel_time'] == e)).astype(float)
        interaction_cols.append(col_name)
        cell_meta.append((col_name, int(g), int(e)))

hosp_dummies = pd.get_dummies(es_df['hospital_id'], prefix='hosp', drop_first=True, dtype=float)
year_dummies = pd.get_dummies(es_df['year'], prefix='yr', drop_first=True, dtype=float)
X_es = pd.concat([es_df[interaction_cols], hosp_dummies, year_dummies], axis=1)
X_es = sm.add_constant(X_es, has_constant='add')
y_es = es_df['antibiotic_duration'].astype(float)

es_ols = sm.OLS(y_es, X_es).fit(
    cov_type='cluster',
    cov_kwds={'groups': es_df['hospital_id']},
    use_t=True
)

n_clusters = int(es_df['hospital_id'].nunique())
t_crit = float(stats.t.ppf(0.975, df=n_clusters - 1))
param_names = list(es_ols.params.index)
beta_hat = es_ols.params.to_numpy()
V_cluster = es_ols.cov_params().to_numpy()

all_event_times = sorted(set(e for _, _, e in cell_meta))
event_study_rows = []
pre_trend_stats = {}

for e in all_event_times:
    cells_e = [(col, g) for col, g, rel in cell_meta if rel == e]
    total_n_e = sum(cohort_counts[g] for _, g in cells_e)
    w_vec = np.zeros(len(param_names), dtype=float)
    for col, g in cells_e:
        w_vec[param_names.index(col)] = cohort_counts[g] / total_n_e
    coef_e = float(w_vec @ beta_hat)
    se_e = float(np.sqrt(max(0.0, w_vec @ V_cluster @ w_vec)))
    ci_lo_e = float(coef_e - t_crit * se_e)
    ci_hi_e = float(coef_e + t_crit * se_e)
    pval_e = float(2.0 * (1.0 - stats.t.cdf(abs(coef_e / se_e), df=n_clusters - 1))) if se_e > 0 else 1.0
    event_study_rows.append({
        'event_time': int(e),
        'coef': coef_e,
        'ci_low': ci_lo_e,
        'ci_high': ci_hi_e
    })
    if e < -1:
        pre_trend_stats[e] = (coef_e, pval_e, [ci_lo_e, ci_hi_e])

event_study = pd.DataFrame(event_study_rows, columns=['event_time', 'coef', 'ci_low', 'ci_high'])

# Pre-treatment lead coefficient at event_time == -2 (identifiable across all treated cohorts)
pre_lead_key = -2 if -2 in pre_trend_stats else max(pre_trend_stats.keys())
pre_trend_coeff, pre_trend_pvalue, pre_trend_ci = pre_trend_stats[pre_lead_key]
pre_trend_coeff = float(pre_trend_coeff)
pre_trend_pvalue = float(pre_trend_pvalue)
pre_trend_ci = [float(pre_trend_ci[0]), float(pre_trend_ci[1])]

# Overall staggered ATT across all post-treatment (g, e >= 0) cells weighted by cohort hospital count
post_cells = [(col, g) for col, g, rel in cell_meta if rel >= 0]
total_post_weight = sum(cohort_counts[g] for _, g in post_cells)
w_att = np.zeros(len(param_names), dtype=float)
for col, g in post_cells:
    w_att[param_names.index(col)] = cohort_counts[g] / total_post_weight
did_att_estimate = float(w_att @ beta_hat)

# ==============================================================================
# 3. TREATMENT HETEROGENEITY: Hospital/Year-Adjusted T-Learner + Bootstrap CI
# ==============================================================================
fe_hosp_dum = pd.get_dummies(df['hospital_id'], prefix='h', drop_first=True, dtype=float)
fe_year_dum = pd.get_dummies(df['year'], prefix='y', drop_first=True, dtype=float)
X_fe = np.column_stack([
    np.ones(len(df), dtype=float),
    df['asp_active'].to_numpy(dtype=float),
    fe_hosp_dum.to_numpy(dtype=float),
    fe_year_dum.to_numpy(dtype=float)
])
y_raw = df['antibiotic_duration'].to_numpy(dtype=float)

beta_fe, _, _, _ = np.linalg.lstsq(X_fe, y_raw, rcond=None)
X_fe_no_trt = X_fe.copy()
X_fe_no_trt[:, 1] = 0.0
y_fe_adj = y_raw - (X_fe_no_trt @ beta_fe)

cate_feature_cols = [
    'age', 'comorbidity_score', 'immune_compromised', 'severity_score',
    'sex', 'infection_type', 'ward', 'hospital_size', 'region'
]
X_cate = pd.get_dummies(
    df[cate_feature_cols],
    columns=['sex', 'infection_type', 'ward', 'hospital_size', 'region'],
    drop_first=True,
    dtype=float
).to_numpy(dtype=float)
w_trt = df['asp_active'].to_numpy(dtype=int)
ages_arr = df['age'].to_numpy(dtype=float)

m0_main = HistGradientBoostingRegressor(max_iter=80, max_depth=4, learning_rate=0.05, random_state=42)
m1_main = HistGradientBoostingRegressor(max_iter=80, max_depth=4, learning_rate=0.05, random_state=42)
m0_main.fit(X_cate[w_trt == 0], y_fe_adj[w_trt == 0])
m1_main.fit(X_cate[w_trt == 1], y_fe_adj[w_trt == 1])
cate_estimates = np.asarray(m1_main.predict(X_cate) - m0_main.predict(X_cate), dtype=float)

rng = np.random.default_rng(42)
n_obs = len(df)
n_boot = 100
boot_diffs = []

for b in range(n_boot):
    idx = rng.integers(0, n_obs, size=n_obs)
    X_fe_b = X_fe[idx]
    y_b = y_raw[idx]
    beta_b, _, _, _ = np.linalg.lstsq(X_fe_b, y_b, rcond=None)
    X_fe_b_notrt = X_fe_b.copy()
    X_fe_b_notrt[:, 1] = 0.0
    y_adj_b = y_b - (X_fe_b_notrt @ beta_b)

    X_cate_b = X_cate[idx]
    w_b = w_trt[idx]
    ages_b = ages_arr[idx]

    m0_b = HistGradientBoostingRegressor(max_iter=60, max_depth=4, learning_rate=0.05, random_state=100 + b)
    m1_b = HistGradientBoostingRegressor(max_iter=60, max_depth=4, learning_rate=0.05, random_state=200 + b)
    m0_b.fit(X_cate_b[w_b == 0], y_adj_b[w_b == 0])
    m1_b.fit(X_cate_b[w_b == 1], y_adj_b[w_b == 1])

    cate_b = m1_b.predict(X_cate_b) - m0_b.predict(X_cate_b)
    diff_b = float(cate_b[ages_b >= 65].mean() - cate_b[ages_b < 40].mean())
    boot_diffs.append(diff_b)

cate_elderly_minus_young_ci = [
    float(np.percentile(boot_diffs, 2.5)),
    float(np.percentile(boot_diffs, 97.5))
]

# ==============================================================================
# 4. HIERARCHICAL MODELING: MixedLM (ML Fit), ICC, AIC Comparison vs Naive OLS
# ==============================================================================
mixed_formula = (
    'antibiotic_duration ~ asp_active + C(year) + age + C(sex) + '
    'comorbidity_score + immune_compromised + severity_score + '
    'C(infection_type) + C(ward) + C(hospital_size) + C(region)'
)
mixed_mod = smf.mixedlm(mixed_formula, data=df, groups=df['hospital_id'])
mixed_res = mixed_mod.fit(reml=False)
if not mixed_res.converged:
    mixed_res = mixed_mod.fit(reml=False, method='bfgs')

ols_res = smf.ols(mixed_formula, data=df).fit()

random_effect_variance = float(np.asarray(mixed_res.cov_re)[0, 0])
residual_variance = float(mixed_res.scale)
icc_value = float(random_effect_variance / (random_effect_variance + residual_variance))
mixed_asp_coeff = float(mixed_res.fe_params['asp_active'])
mixed_model_aic = (
    float(mixed_res.aic)
    if not np.isnan(mixed_res.aic)
    else float(-2.0 * mixed_res.llf + 2.0 * (len(mixed_res.fe_params) + 2))
)
naive_ols_aic = float(ols_res.aic)