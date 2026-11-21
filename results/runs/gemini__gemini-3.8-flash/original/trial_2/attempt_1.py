import pandas as pd
import numpy as np
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
cph.fit(surv_df, duration_col='time_to_improvement', event_col='event_observed', cluster_col='hospital_id')

cox_hr_asp = float(cph.hazard_ratios_['asp_active'])
cox_se_asp = float(cph.standard_errors_['asp_active'])

ph_test = proportional_hazard_test(cph, surv_df, time_transform='rank')
ph_test_pvalues = {str(k): float(v) for k, v in zip(ph_test.summary.index, ph_test.summary['p'])}

# ==========================================
# 2. CAUSAL IMPACT (DiD & EVENT STUDY)
# ==========================================
did_panel = df.groupby(['hospital_id', 'year'], as_index=False).agg(
    antibiotic_duration=('antibiotic_duration', 'mean'),
    asp_adoption_year=('asp_adoption_year', 'first'),
    asp_active=('asp_active', 'first')
)

# TWFE ATT
twfe_res = smf.ols('antibiotic_duration ~ asp_active + C(hospital_id) + C(year)', data=did_panel).fit()
twfe_att_estimate = float(twfe_res.params['asp_active'])

# Robust Staggered DiD ATT (Callaway & Sant'Anna)
never_treated = did_panel[did_panel['asp_adoption_year'] == 9999]
cohorts = sorted([g for g in did_panel['asp_adoption_year'].unique() if g < 9999])

att_gt_list = []
weight_list = []
for g in cohorts:
    g_panel = did_panel[did_panel['asp_adoption_year'] == g]
    n_hosp = g_panel['hospital_id'].nunique()
    y_g_base = g_panel[g_panel['year'] == g - 1]['antibiotic_duration'].mean()
    y_c_base = never_treated[never_treated['year'] == g - 1]['antibiotic_duration'].mean()
    for t in sorted(did_panel['year'].unique()):
        if t >= g:
            y_g_t = g_panel[g_panel['year'] == t]['antibiotic_duration'].mean()
            y_c_t = never_treated[never_treated['year'] == t]['antibiotic_duration'].mean()
            att_gt = (y_g_t - y_g_base) - (y_c_t - y_c_base)
            att_gt_list.append(att_gt)
            weight_list.append(n_hosp)

did_att_estimate = float(np.average(att_gt_list, weights=weight_list))

# Event Study & Pre-trend Testing
did_panel['rel_year'] = np.where(
    did_panel['asp_adoption_year'] < 9999,
    did_panel['year'] - did_panel['asp_adoption_year'],
    -999
)
event_times = sorted([int(k) for k in did_panel.loc[did_panel['asp_adoption_year'] < 9999, 'rel_year'].unique() if k != -1])

for k in event_times:
    col_name = f'ev_m{abs(k)}' if k < 0 else f'ev_{k}'
    did_panel[col_name] = ((did_panel['asp_adoption_year'] < 9999) & (did_panel['rel_year'] == k)).astype(int)

ev_cols = [f'ev_m{abs(k)}' if k < 0 else f'ev_{k}' for k in event_times]
es_formula = 'antibiotic_duration ~ ' + ' + '.join(ev_cols) + ' + C(hospital_id) + C(year)'
es_res = smf.ols(es_formula, data=did_panel).fit()

event_records = []
for k, col in zip(event_times, ev_cols):
    ci = es_res.conf_int().loc[col]
    event_records.append({
        'event_time': k,
        'coef': float(es_res.params[col]),
        'ci_low': float(ci[0]),
        'ci_high': float(ci[1])
    })
event_study = pd.DataFrame(event_records)

pre_trend_coeff = float(es_res.params['ev_m2'])
pre_trend_pvalue = float(es_res.pvalues['ev_m2'])
pre_trend_ci = [float(es_res.conf_int().loc['ev_m2', 0]), float(es_res.conf_int().loc['ev_m2', 1])]

# ==========================================
# 3. TREATMENT HETEROGENEITY (CATE)
# ==========================================
feature_cols = ['age', 'sex', 'comorbidity_score', 'immune_compromised',
                'infection_type', 'severity_score', 'ward', 'hospital_size',
                'region', 'hospital_id', 'year']
X = pd.get_dummies(df[feature_cols], columns=['sex', 'infection_type', 'ward', 'hospital_size', 'region', 'hospital_id', 'year'], drop_first=True)
y = df['antibiotic_duration'].values
w = df['asp_active'].values

m0 = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
m1 = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
m0.fit(X[w == 0], y[w == 0])
m1.fit(X[w == 1], y[w == 1])

cate_estimates = m1.predict(X) - m0.predict(X)

np.random.seed(42)
elderly_cates = cate_estimates[df['age'] >= 65]
young_cates = cate_estimates[df['age'] < 40]

boot_diffs = []
for _ in range(1000):
    b_e = np.random.choice(elderly_cates, size=len(elderly_cates), replace=True).mean()
    b_y = np.random.choice(young_cates, size=len(young_cates), replace=True).mean()
    boot_diffs.append(b_e - b_y)

cate_elderly_minus_young_ci = [float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))]

# ==========================================
# 4. HIERARCHICAL MODELING
# ==========================================
mixed_model = smf.mixedlm("antibiotic_duration ~ asp_active", data=df, groups=df["hospital_id"])
mixed_res = mixed_model.fit(reml=False)

naive_ols = smf.ols("antibiotic_duration ~ asp_active", data=df).fit()

mixed_asp_coeff = float(mixed_res.params['asp_active'])
random_effect_variance = float(mixed_res.cov_re.iloc[0, 0])
residual_variance = float(mixed_res.scale)

icc_value = float(random_effect_variance / (random_effect_variance + residual_variance))
mixed_model_aic = float(mixed_res.aic)
naive_ols_aic = float(naive_ols.aic)