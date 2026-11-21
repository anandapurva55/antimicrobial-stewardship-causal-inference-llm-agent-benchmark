import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test
from sklearn.ensemble import GradientBoostingRegressor

# 1. SURVIVAL ANALYSIS
kmf = KaplanMeierFitter()
kmf.fit(df['time_to_improvement'], event_observed=df['event_observed'], label='All')

cph = CoxPHFitter()
cph_df = df[['time_to_improvement', 'event_observed', 'asp_active', 'hospital_id']]
cph.fit(
    cph_df,
    duration_col='time_to_improvement',
    event_col='event_observed',
    cluster_col='hospital_id'
)

cox_hr_asp = float(cph.hazard_ratios_['asp_active'])
cox_se_asp = float(cph.standard_errors_['asp_active'])

ph_test = proportional_hazard_test(cph, cph_df, time_transform='rank')
ph_test_pvalues = {var: float(p) for var, p in zip(ph_test.summary.index, ph_test.summary['p'])}

# 2. CAUSAL IMPACT (DiD)
did_panel = df.groupby(['hospital_id', 'year']).agg(
    antibiotic_duration=('antibiotic_duration', 'mean'),
    asp_active=('asp_active', 'first'),
    asp_adoption_year=('asp_adoption_year', 'first')
).reset_index()

twfe_mod = smf.ols('antibiotic_duration ~ asp_active + C(hospital_id) + C(year)', data=did_panel).fit()
twfe_att_estimate = float(twfe_mod.params['asp_active'])

# Event study specification (omitting reference period event_time = -1)
event_times = [-4, -3, -2, 0, 1, 2]
event_cols = []
for k in event_times:
    col_name = f'ev_{k}' if k >= 0 else f'ev_m{abs(k)}'
    did_panel[col_name] = (
        (did_panel['asp_adoption_year'] < 9000) &
        (did_panel['year'] - did_panel['asp_adoption_year'] == k)
    ).astype(int)
    event_cols.append((k, col_name))

formula_es = 'antibiotic_duration ~ ' + ' + '.join([c for _, c in event_cols]) + ' + C(hospital_id) + C(year)'
es_mod = smf.ols(formula_es, data=did_panel).fit()
conf_int_es = es_mod.conf_int()

es_records = []
for k, col_name in event_cols:
    es_records.append({
        'event_time': k,
        'coef': float(es_mod.params[col_name]),
        'ci_low': float(conf_int_es.loc[col_name, 0]),
        'ci_high': float(conf_int_es.loc[col_name, 1])
    })
event_study = pd.DataFrame(es_records)

# Pre-trend test (lead immediately prior to reference year -1)
pre_trend_coeff = float(es_mod.params['ev_m2'])
pre_trend_pvalue = float(es_mod.pvalues['ev_m2'])
pre_trend_ci = [float(conf_int_es.loc['ev_m2', 0]), float(conf_int_es.loc['ev_m2', 1])]

# Robust staggered adoption estimator (Stacked DiD with never-treated hospitals)
stacks = []
for cohort in [2020, 2021]:
    sub = did_panel[did_panel['asp_adoption_year'].isin([cohort, 9999])].copy()
    sub['unit_stack'] = sub['hospital_id'].astype(str) + '_' + str(cohort)
    sub['time_stack'] = sub['year'].astype(str) + '_' + str(cohort)
    sub['treated_post'] = ((sub['asp_adoption_year'] == cohort) & (sub['year'] >= cohort)).astype(int)
    stacks.append(sub)
stacked_df = pd.concat(stacks, ignore_index=True)
stacked_mod = smf.ols('antibiotic_duration ~ treated_post + C(unit_stack) + C(time_stack)', data=stacked_df).fit()
did_att_estimate = float(stacked_mod.params['treated_post'])

# 3. TREATMENT HETEROGENEITY (T-Learner)
feature_cols = ['age', 'comorbidity_score', 'immune_compromised', 'severity_score', 'year', 'hospital_id', 'sex', 'ward', 'infection_type']
X = pd.get_dummies(df[feature_cols], columns=['hospital_id', 'sex', 'ward', 'infection_type'], drop_first=True)
y = df['antibiotic_duration'].values
w = df['asp_active'].values

m0 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
m1 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)

m0.fit(X[w == 0], y[w == 0])
m1.fit(X[w == 1], y[w == 1])

cate_estimates = m1.predict(X) - m0.predict(X)

elderly_mask = df['age'].values >= 65
young_mask = df['age'].values < 40

np.random.seed(42)
diffs = []
n_total = len(df)
for _ in range(1000):
    idx = np.random.choice(n_total, size=n_total, replace=True)
    eld_mean = cate_estimates[idx][elderly_mask[idx]].mean()
    yng_mean = cate_estimates[idx][young_mask[idx]].mean()
    diffs.append(eld_mean - yng_mean)

cate_elderly_minus_young_ci = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]

# 4. HIERARCHICAL MODELING
mixed_mod = smf.mixedlm('antibiotic_duration ~ asp_active', data=df, groups=df['hospital_id'])
mixed_res = mixed_mod.fit(reml=False)

naive_mod = smf.ols('antibiotic_duration ~ asp_active', data=df).fit()

mixed_asp_coeff = float(mixed_res.params['asp_active'])
mixed_model_aic = float(mixed_res.aic)
naive_ols_aic = float(naive_mod.aic)

random_effect_variance = float(mixed_res.cov_re.iloc[0, 0])
residual_variance = float(mixed_res.scale)
icc_value = float(random_effect_variance / (random_effect_variance + residual_variance))