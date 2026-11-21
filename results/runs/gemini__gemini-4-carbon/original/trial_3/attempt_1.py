import time
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, proportional_hazard_test
from linearmodels.panel import PanelOLS
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge
import statsmodels.api as sm
import statsmodels.formula.api as smf

df_work = df.copy()

# ==============================================================================
# 1. SURVIVAL ANALYSIS
# ==============================================================================
kmf_control = KaplanMeierFitter()
kmf_treated = KaplanMeierFitter()

mask_treated = df_work['asp_active'] == 1
kmf_control.fit(
    durations=df_work.loc[~mask_treated, 'time_to_improvement'],
    event_observed=df_work.loc[~mask_treated, 'event_observed'],
    label='ASP Inactive',
)
kmf_treated.fit(
    durations=df_work.loc[mask_treated, 'time_to_improvement'],
    event_observed=df_work.loc[mask_treated, 'event_observed'],
    label='ASP Active',
)
km_logrank = logrank_test(
    df_work.loc[~mask_treated, 'time_to_improvement'],
    df_work.loc[mask_treated, 'time_to_improvement'],
    event_observed_A=df_work.loc[~mask_treated, 'event_observed'],
    event_observed_B=df_work.loc[mask_treated, 'event_observed'],
)

cox_num_cols = [
    'asp_active',
    'age',
    'comorbidity_score',
    'immune_compromised',
    'severity_score',
    'year',
]
cox_cat_cols = ['sex', 'ward', 'infection_type', 'hospital_size', 'region']

cox_df = pd.get_dummies(
    df_work[cox_num_cols + cox_cat_cols],
    columns=cox_cat_cols,
    drop_first=True,
    dtype=float,
)
cox_df['time_to_improvement'] = df_work['time_to_improvement'].astype(float)
cox_df['event_observed'] = df_work['event_observed'].astype(int)
cox_df['hospital_id'] = df_work['hospital_id']

cph = CoxPHFitter()
try:
  cph.fit(
      cox_df,
      duration_col='time_to_improvement',
      event_col='event_observed',
      cluster_col='hospital_id',
  )
except Exception:
  cph = CoxPHFitter(penalizer=1e-4)
  cph.fit(
      cox_df,
      duration_col='time_to_improvement',
      event_col='event_observed',
      cluster_col='hospital_id',
  )

cox_hr_asp = float(cph.hazard_ratios_['asp_active'])
cox_se_asp = float(cph.standard_errors_['asp_active'])

try:
  ph_res = proportional_hazard_test(cph, cox_df, time_transform='rank')
except Exception:
  ph_res = proportional_hazard_test(
      cph, cox_df.drop(columns=['hospital_id']), time_transform='rank'
  )

ph_test_pvalues = {str(k): float(v) for k, v in ph_res.summary['p'].items()}

# ==============================================================================
# 2. CAUSAL IMPACT (STAGGERED DiD & EVENT STUDY)
# ==============================================================================
did_panel = (
    df_work.groupby(['hospital_id', 'year'], as_index=False)
    .agg(
        antibiotic_duration=('antibiotic_duration', 'mean'),
        asp_active=('asp_active', 'max'),
        asp_adoption_year=('asp_adoption_year', 'first'),
        hospital_size=('hospital_size', 'first'),
        region=('region', 'first'),
        mean_age=('age', 'mean'),
        mean_comorbidity=('comorbidity_score', 'mean'),
        mean_severity=('severity_score', 'mean'),
        n_patients=('patient_id', 'count'),
    )
    .sort_values(['hospital_id', 'year'])
    .reset_index(drop=True)
)

max_study_year = int(did_panel['year'].max())
treated_cohorts = sorted(
    int(g)
    for g in did_panel['asp_adoption_year'].unique()
    if g <= max_study_year
)
cohort_sizes = (
    did_panel.groupby('asp_adoption_year')['hospital_id'].nunique().to_dict()
)

# Sun & Abraham (2021) / Callaway & Sant'Anna (2021) saturated cohort-time specification
# Never-treated hospitals (asp_adoption_year == 9999) serve as clean controls; event_time = -1 is omitted
sa_df = did_panel.copy()
cohort_event_cols = {}

for g in treated_cohorts:
  years_g = sorted(sa_df.loc[sa_df['asp_adoption_year'] == g, 'year'].unique())
  for yr in years_g:
    e = int(yr - g)
    if e == -1:
      continue
    col_name = f'D_{g}_m{abs(e)}' if e < 0 else f'D_{g}_p{e}'
    sa_df[col_name] = (
        (sa_df['asp_adoption_year'] == g) & (sa_df['year'] == yr)
    ).astype(float)
    cohort_event_cols[(g, e)] = col_name

rhs_terms = (
    ['C(hospital_id)', 'C(year)'] + list(cohort_event_cols.values())
)
sa_formula = 'antibiotic_duration ~ ' + ' + '.join(rhs_terms)
sa_res = smf.ols(sa_formula, data=sa_df).fit(
    cov_type='cluster', cov_kwds={'groups': sa_df['hospital_id']}
)
param_names = list(sa_res.params.index)

all_event_times = sorted({e for (_, e) in cohort_event_cols.keys()})
es_rows = []
pre_trend_stats = {}

for e in all_event_times:
  cohorts_at_e = [g for g in treated_cohorts if (g, e) in cohort_event_cols]
  total_n = float(sum(cohort_sizes[g] for g in cohorts_at_e))
  L = np.zeros((1, len(param_names)), dtype=float)
  for g in cohorts_at_e:
    col_name = cohort_event_cols[(g, e)]
    L[0, param_names.index(col_name)] = cohort_sizes[g] / total_n
  tt = sa_res.t_test(L)
  coef_e = float(np.asarray(tt.effect).item())
  pval_e = float(np.asarray(tt.pvalue).item())
  ci_arr = np.asarray(tt.conf_int(alpha=0.05)).reshape(-1)
  ci_lo, ci_hi = float(ci_arr[0]), float(ci_arr[1])
  es_rows.append(
      {'event_time': int(e), 'coef': coef_e, 'ci_low': ci_lo, 'ci_high': ci_hi}
  )
  pre_trend_stats[int(e)] = (coef_e, pval_e, [ci_lo, ci_hi])

event_study = pd.DataFrame(es_rows).sort_values('event_time').reset_index(
    drop=True
)

# Pre-trends validation using the immediate pre-reference lead (event_time = -2)
lead_key = -2 if -2 in pre_trend_stats else min(
    e for e in pre_trend_stats if e < 0
)
pre_trend_coeff = float(pre_trend_stats[lead_key][0])
pre_trend_pvalue = float(pre_trend_stats[lead_key][1])
pre_trend_ci = [
    float(pre_trend_stats[lead_key][2][0]),
    float(pre_trend_stats[lead_key][2][1]),
]

# Plain TWFE on the hospital-year panel
twfe_mod = PanelOLS.from_formula(
    'antibiotic_duration ~ asp_active + EntityEffects + TimeEffects',
    data=did_panel.set_index(['hospital_id', 'year']),
)
twfe_res = twfe_mod.fit(cov_type='clustered', cluster_entity=True)
twfe_att_estimate = float(twfe_res.params['asp_active'])

# Staggered-adoption-robust overall ATT (Callaway-Sant'Anna / Sun-Abraham aggregation over e >= 0)
post_cells = [
    (g, e) for (g, e) in cohort_event_cols.keys() if e >= 0
]
total_post_weight = float(sum(cohort_sizes[g] for (g, _) in post_cells))
L_att = np.zeros((1, len(param_names)), dtype=float)
for g, e in post_cells:
  col_name = cohort_event_cols[(g, e)]
  L_att[0, param_names.index(col_name)] = cohort_sizes[g] / total_post_weight

att_test = sa_res.t_test(L_att)
did_att_estimate = float(np.asarray(att_test.effect).item())

# ==============================================================================
# 3. TREATMENT HETEROGENEITY (X-LEARNER WITH HOSPITAL & YEAR ADJUSTMENT)
# ==============================================================================
X_fe_df = pd.get_dummies(
    df_work[['hospital_id', 'year']].astype(str), drop_first=True, dtype=float
)
ml_num_cols = [
    'age',
    'comorbidity_score',
    'immune_compromised',
    'severity_score',
]
ml_cat_cols = [
    'sex',
    'age_group',
    'ward',
    'infection_type',
    'hospital_size',
    'region',
]
X_ml_df = pd.get_dummies(
    df_work[ml_num_cols + ml_cat_cols],
    columns=ml_cat_cols,
    drop_first=True,
    dtype=float,
)

X_fe_mat = X_fe_df.to_numpy(dtype=float)
X_ml_mat = X_ml_df.to_numpy(dtype=float)
d_vec = df_work['asp_active'].to_numpy(dtype=int)
y_vec = df_work['antibiotic_duration'].to_numpy(dtype=float)
age_vec = df_work['age'].to_numpy(dtype=float)


def fit_predict_xlearner(X_fe_tr, X_ml_tr, d_tr, y_tr, X_ml_ev, seed=42):
  idx0 = d_tr == 0
  idx1 = d_tr == 1

  # Estimate additive hospital and year fixed effects on untreated observations
  X_combined_0 = np.hstack([X_fe_tr[idx0], X_ml_tr[idx0]])
  fe_reg = Ridge(alpha=1e-2)
  fe_reg.fit(X_combined_0, y_tr[idx0])
  n_fe = X_fe_tr.shape[1]
  fe_offset = X_fe_tr @ fe_reg.coef_[:n_fe]
  y_adj = y_tr - fe_offset

  # Stage 1: Potential outcome models via Gradient Boosting
  mu0 = GradientBoostingRegressor(
      n_estimators=80,
      max_depth=3,
      learning_rate=0.08,
      subsample=0.8,
      random_state=seed,
  )
  mu1 = GradientBoostingRegressor(
      n_estimators=80,
      max_depth=3,
      learning_rate=0.08,
      subsample=0.8,
      random_state=seed + 1,
  )
  mu0.fit(X_ml_tr[idx0], y_adj[idx0])
  mu1.fit(X_ml_tr[idx1], y_adj[idx1])

  # Stage 2: Imputed individual treatment effects & CATE models
  d1_tilde = y_adj[idx1] - mu0.predict(X_ml_tr[idx1])
  d0_tilde = mu1.predict(X_ml_tr[idx0]) - y_adj[idx0]

  tau1 = GradientBoostingRegressor(
      n_estimators=80,
      max_depth=3,
      learning_rate=0.08,
      subsample=0.8,
      random_state=seed + 2,
  )
  tau0 = GradientBoostingRegressor(
      n_estimators=80,
      max_depth=3,
      learning_rate=0.08,
      subsample=0.8,
      random_state=seed + 3,
  )
  tau1.fit(X_ml_tr[idx1], d1_tilde)
  tau0.fit(X_ml_tr[idx0], d0_tilde)

  # Stage 3: Propensity-weighted combination
  prop_clf = GradientBoostingClassifier(
      n_estimators=60,
      max_depth=3,
      learning_rate=0.08,
      subsample=0.8,
      random_state=seed + 4,
  )
  prop_clf.fit(X_ml_tr, d_tr)
  g_ev = np.clip(prop_clf.predict_proba(X_ml_ev)[:, 1], 0.05, 0.95)

  return g_ev * tau0.predict(X_ml_ev) + (1.0 - g_ev) * tau1.predict(X_ml_ev)


cate_estimates = np.asarray(
    fit_predict_xlearner(X_fe_mat, X_ml_mat, d_vec, y_vec, X_ml_mat, seed=42),
    dtype=float,
)

rng = np.random.default_rng(42)
n_obs = len(df_work)
n_boot = 100
boot_diffs = []
t_start = time.time()

for b in range(n_boot):
  b_idx = rng.integers(0, n_obs, size=n_obs)
  if d_vec[b_idx].sum() < 20 or (len(b_idx) - d_vec[b_idx].sum()) < 20:
    continue
  cate_b = fit_predict_xlearner(
      X_fe_mat[b_idx],
      X_ml_mat[b_idx],
      d_vec[b_idx],
      y_vec[b_idx],
      X_ml_mat[b_idx],
      seed=1000 + b * 10,
  )
  age_b = age_vec[b_idx]
  diff_b = float(np.mean(cate_b[age_b >= 65]) - np.mean(cate_b[age_b < 40]))
  boot_diffs.append(diff_b)
  if b >= 59 and (time.time() - t_start) > 45.0:
    break

cate_elderly_minus_young_ci = [
    float(np.percentile(boot_diffs, 2.5)),
    float(np.percentile(boot_diffs, 97.5)),
]

# ==============================================================================
# 4. HIERARCHICAL MODELING (MIXED-EFFECTS LINEAR MODEL VS NAIVE OLS)
# ==============================================================================
hlm_formula = (
    'antibiotic_duration ~ asp_active + C(year) + age + C(sex) '
    '+ comorbidity_score + immune_compromised + severity_score '
    '+ C(ward) + C(infection_type)'
)

mixed_mod = smf.mixedlm(
    hlm_formula, data=df_work, groups=df_work['hospital_id']
)
mixed_res = mixed_mod.fit(reml=False, method=['lbfgs', 'powell'])

naive_ols_res = smf.ols(hlm_formula, data=df_work).fit()

random_effect_variance = float(mixed_res.cov_re.iloc[0, 0])
residual_variance = float(mixed_res.scale)
icc_value = float(
    random_effect_variance / (random_effect_variance + residual_variance)
)

mixed_model_aic = (
    float(mixed_res.aic)
    if not np.isnan(mixed_res.aic)
    else float(-2.0 * mixed_res.llf + 2.0 * (mixed_res.df_modelwc + 1))
)
naive_ols_aic = float(naive_ols_res.aic)
mixed_asp_coeff = float(mixed_res.fe_params['asp_active'])