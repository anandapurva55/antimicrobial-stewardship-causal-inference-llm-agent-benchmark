from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from sklearn.ensemble import GradientBoostingRegressor
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 1. SURVIVAL ANALYSIS
# ---------------------------------------------------------
kmf = KaplanMeierFitter()
kmf.fit(df['time_to_improvement'], event_observed=df['event_observed'])

cox_df = df[['time_to_improvement', 'event_observed', 'asp_active', 'hospital_id']].copy()
cph = CoxPHFitter()
cph.fit(cox_df, duration_col='time_to_improvement', event_col='event_observed', cluster_col='hospital_id')

cox_hr_asp = float(cph.hazard_ratios_['asp_active'])
cox_se_asp = float(cph.standard_errors_['asp_active'])

ph_test = proportional_hazard_test(cph, cox_df, time_transform='rank')
if hasattr(ph_test, 'summary') and 'p' in ph_test.summary.columns:
    ph_test_pvalues = {str(k): float(v) for k, v in ph_test.summary['p'].to_dict().items()}
elif hasattr(ph_test, 'p_value'):
    if isinstance(ph_test.p_value, (float, np.floating)):
        ph_test_pvalues = {'asp_active': float(ph_test.p_value)}
    else:
        ph_test_pvalues = {str(k): float(v) for k, v in zip(cph.params_.index, np.atleast_1d(ph_test.p_value))}
else:
    ph_test_pvalues = {'asp_active': float(ph_test.p_value)}

# ---------------------------------------------------------
# 2. CAUSAL IMPACT (DiD)
# ---------------------------------------------------------
did_panel = df.groupby(['hospital_id', 'year']).agg({
    'antibiotic_duration': 'mean',
    'asp_active': 'first',
    'asp_adoption_year': 'first'
}).reset_index()

# Plain Two-Way Fixed Effects (TWFE)
twfe_model = smf.ols('antibiotic_duration ~ asp_active + C(hospital_id) + C(year)', data=did_panel).fit()
twfe_att_estimate = float(twfe_model.params['asp_active'])

# Event study specification (omitting reference period event_time = -1)
did_panel['event_time'] = np.where(
    did_panel['asp_adoption_year'] < 9999,
    did_panel['year'] - did_panel['asp_adoption_year'],
    np.nan
)

event_times = [-4, -3, -2, 0, 1, 2]
dummy_cols = []
for et in event_times:
    col = f'event_{et}' if et >= 0 else f'event_m{abs(et)}'
    did_panel[col] = (did_panel['event_time'] == et).astype(int)
    dummy_cols.append(col)

formula_es = 'antibiotic_duration ~ ' + ' + '.join(dummy_cols) + ' + C(hospital_id) + C(year)'
es_mod = smf.ols(formula_es, data=did_panel).fit()

es_rows = []
for et, col in zip(event_times, dummy_cols):
    coef = es_mod.params[col]
    ci = es_mod.conf_int().loc[col]
    es_rows.append({
        'event_time': et,
        'coef': float(coef),
        'ci_low': float(ci[0]),
        'ci_high': float(ci[1])
    })
event_study = pd.DataFrame(es_rows)

pre_trend_coeff = float(es_mod.params['event_m2'])
pre_trend_pvalue = float(es_mod.pvalues['event_m2'])
pre_trend_ci = [float(es_mod.conf_int().loc['event_m2', 0]), float(es_mod.conf_int().loc['event_m2', 1])]

# Robust estimator to staggered adoption: Stacked DiD using never-treated hospitals as controls
stack_dfs = []
for g in [2020, 2021]:
    sub = did_panel[did_panel['asp_adoption_year'].isin([g, 9999])].copy()
    sub['stack'] = g
    sub['treat_post'] = ((sub['asp_adoption_year'] == g) & (sub['year'] >= g)).astype(int)
    sub['hosp_stack'] = sub['hospital_id'].astype(str) + '_' + str(g)
    sub['year_stack'] = sub['year'].astype(str) + '_' + str(g)
    stack_dfs.append(sub)

stacked_df = pd.concat(stack_dfs, ignore_index=True)
stacked_mod = smf.ols('antibiotic_duration ~ treat_post + C(hosp_stack) + C(year_stack)', data=stacked_df).fit()
did_att_estimate = float(stacked_mod.params['treat_post'])

# ---------------------------------------------------------
# 3. TREATMENT HETEROGENEITY (CATE)
# ---------------------------------------------------------
covar_cols = ['hospital_id', 'year', 'hospital_size', 'region', 'ward', 'age', 'sex', 
              'comorbidity_score', 'immune_compromised', 'infection_type', 'severity_score']
X = pd.get_dummies(df[covar_cols], columns=['hospital_id', 'year', 'hospital_size', 'region', 'ward', 'sex', 'infection_type'], drop_first=True)
w = df['asp_active'].values
y = df['antibiotic_duration'].values

# T-Learner with Gradient Boosting Regressors
m0 = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
m1 = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
m0.fit(X[w == 0], y[w == 0])
m1.fit(X[w == 1], y[w == 1])

cate_estimates = m1.predict(X) - m0.predict(X)

elderly_idx = np.where(df['age'] >= 65)[0]
young_idx = np.where(df['age'] < 40)[0]

np.random.seed(42)
boot_diffs = []
for _ in range(1000):
    b_eld = np.random.choice(cate_estimates[elderly_idx], size=len(elderly_idx), replace=True).mean()
    b_yng = np.random.choice(cate_estimates[young_idx], size=len(young_idx), replace=True).mean()
    boot_diffs.append(b_eld - b_yng)

ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
cate_elderly_minus_young_ci = [float(ci_lo), float(ci_hi)]

# ---------------------------------------------------------
# 4. HIERARCHICAL MODELING
# ---------------------------------------------------------
mixed_model = smf.mixedlm("antibiotic_duration ~ asp_active", data=df, groups=df["hospital_id"])
mixed_fit = mixed_model.fit(reml=False)

random_effect_variance = float(mixed_fit.cov_re.iloc[0, 0])
residual_variance = float(mixed_fit.scale)
icc_value = float(random_effect_variance / (random_effect_variance + residual_variance))

mixed_asp_coeff = float(mixed_fit.params['asp_active'])
mixed_model_aic = float(mixed_fit.aic if hasattr(mixed_fit, 'aic') else -2 * mixed_fit.llf + 2 * len(mixed_fit.params))

naive_ols = smf.ols("antibiotic_duration ~ asp_active", data=df).fit()
naive_ols_aic = float(naive_ols.aic)