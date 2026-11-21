import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from sklearn.ensemble import GradientBoostingRegressor

# ==========================================
# 1. SURVIVAL ANALYSIS
# ==========================================
surv_df = df[['time_to_improvement', 'event_observed', 'asp_active', 'hospital_id']].copy()

cph = CoxPHFitter()
cph.fit(
    surv_df,
    duration_col='time_to_improvement',
    event_col='event_observed',
    cluster_col='hospital_id'
)

cox_hr_asp = float(cph.hazard_ratios_['asp_active'])
cox_se_asp = float(cph.standard_errors_['asp_active'])

p_test = proportional_hazard_test(
    cph,
    surv_df[['time_to_improvement', 'event_observed', 'asp_active']],
    time_transform='rank'
)
ph_test_pvalues = {var: float(p) for var, p in zip(p_test.summary.index, p_test.summary['p'])}

# ==========================================
# 2. CAUSAL IMPACT (DiD)
# ==========================================
did_panel = df.groupby(['hospital_id', 'year']).agg(
    antibiotic_duration=('antibiotic_duration', 'mean'),
    asp_active=('asp_active', 'first'),
    asp_adoption_year=('asp_adoption_year', 'first')
).reset_index()

twfe_res = smf.ols(
    'antibiotic_duration ~ asp_active + C(hospital_id) + C(year)',
    data=did_panel
).fit()
twfe_att_estimate = float(twfe_res.params['asp_active'])

event_times = [-4, -3, -2, 0, 1, 2]
for e in event_times:
    did_panel[f'event_{e}'] = (
        (did_panel['asp_adoption_year'] != 9999) &
        ((did_panel['year'] - did_panel['asp_adoption_year']) == e)
    ).astype(int)

formula = 'antibiotic_duration ~ ' + ' + '.join([f'event_{e}' for e in event_times]) + ' + C(hospital_id) + C(year)'
es_res = smf.ols(formula, data=did_panel).fit()

event_study = pd.DataFrame({
    'event_time': event_times,
    'coef': [float(es_res.params[f'event_{e}']) for e in event_times],
    'ci_low': [float(es_res.conf_int().loc[f'event_{e}', 0]) for e in event_times],
    'ci_high': [float(es_res.conf_int().loc[f'event_{e}', 1]) for e in event_times]
})

pre_trend_coeff = float(es_res.params['event_-2'])
pre_trend_pvalue = float(es_res.pvalues['event_-2'])
pre_trend_ci = [
    float(es_res.conf_int().loc['event_-2', 0]),
    float(es_res.conf_int().loc['event_-2', 1])
]

# Robust DiD estimator (Callaway-Sant'Anna using never-treated controls)
att_gt_list = []
w_list = []
for g in [2020, 2021]:
    base_treated = did_panel.loc[(did_panel['asp_adoption_year'] == g) & (did_panel['year'] == g - 1), 'antibiotic_duration'].mean()
    base_control = did_panel.loc[(did_panel['asp_adoption_year'] == 9999) & (did_panel['year'] == g - 1), 'antibiotic_duration'].mean()
    n_hosp = (did_panel['asp_adoption_year'] == g).sum() // 6
    for t in range(g, 2023):
        post_treated = did_panel.loc[(did_panel['asp_adoption_year'] == g) & (did_panel['year'] == t), 'antibiotic_duration'].mean()
        post_control = did_panel.loc[(did_panel['asp_adoption_year'] == 9999) & (did_panel['year'] == t), 'antibiotic_duration'].mean()
        att = (post_treated - base_treated) - (post_control - base_control)
        att_gt_list.append(att)
        w_list.append(n_hosp)

did_att_estimate = float(np.average(att_gt_list, weights=w_list))

# ==========================================
# 3. TREATMENT HETEROGENEITY (T-Learner)
# ==========================================
features = ['age', 'comorbidity_score', 'severity_score', 'immune_compromised', 'sex', 'infection_type', 'hospital_id', 'year']
X = pd.get_dummies(df[features], columns=['sex', 'infection_type', 'hospital_id', 'year'], drop_first=True)
y = df['antibiotic_duration'].values
w = df['asp_active'].values

m0 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
m1 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)

m0.fit(X[w == 0], y[w == 0])
m1.fit(X[w == 1], y[w == 1])

cate_estimates = m1.predict(X) - m0.predict(X)

elderly_idx = (df['age'] >= 65).values
young_idx = (df['age'] < 40).values

elderly_cate = cate_estimates[elderly_idx]
young_cate = cate_estimates[young_idx]

np.random.seed(42)
boot_diffs = []
for _ in range(1000):
    b_elderly = np.random.choice(elderly_cate, size=len(elderly_cate), replace=True)
    b_young = np.random.choice(young_cate, size=len(young_cate), replace=True)
    boot_diffs.append(np.mean(b_elderly) - np.mean(b_young))

cate_elderly_minus_young_ci = [
    float(np.percentile(boot_diffs, 2.5)),
    float(np.percentile(boot_diffs, 97.5))
]

# ==========================================
# 4. HIERARCHICAL MODELING
# ==========================================
ols_model = smf.ols('antibiotic_duration ~ asp_active', data=df).fit()
naive_ols_aic = float(ols_model.aic)

mixed_model = smf.mixedlm('antibiotic_duration ~ asp_active', data=df, groups=df['hospital_id'])
mixed_res = mixed_model.fit(reml=False)

mixed_model_aic = float(mixed_res.aic)
mixed_asp_coeff = float(mixed_res.params['asp_active'])
random_effect_variance = float(mixed_res.cov_re.iloc[0, 0])
residual_variance = float(mixed_res.scale)
icc_value = float(random_effect_variance / (random_effect_variance + residual_variance))