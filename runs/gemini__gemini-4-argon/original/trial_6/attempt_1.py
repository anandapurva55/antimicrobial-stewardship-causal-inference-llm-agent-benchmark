import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from sklearn.ensemble import GradientBoostingRegressor

# ==========================================
# 1. SURVIVAL ANALYSIS
# ==========================================
kmf = KaplanMeierFitter()
for val, grp in df.groupby('asp_active'):
    kmf.fit(grp['time_to_improvement'], event_observed=grp['event_observed'], label=f'ASP_{val}')

cox_df = df[['time_to_improvement', 'event_observed', 'asp_active', 'age', 'comorbidity_score', 
             'immune_compromised', 'severity_score', 'sex', 'ward', 'infection_type', 'hospital_id']].copy()
cox_df = pd.get_dummies(cox_df, columns=['sex', 'ward', 'infection_type'], drop_first=True, dtype=float)

cph = CoxPHFitter()
cph.fit(cox_df, duration_col='time_to_improvement', event_col='event_observed', cluster_col='hospital_id')

cox_hr_asp = float(cph.hazard_ratios_['asp_active'])
cox_se_asp = float(cph.standard_errors_['asp_active'])

ph_res = proportional_hazard_test(cph, cox_df, time_transform='rank')
ph_test_pvalues = {str(k): float(v) for k, v in ph_res.summary['p'].to_dict().items()}

# ==========================================
# 2. CAUSAL IMPACT (DiD)
# ==========================================
did_panel = df.groupby(['hospital_id', 'year'], as_index=False).agg(
    antibiotic_duration=('antibiotic_duration', 'mean'),
    asp_active=('asp_active', 'mean'),
    asp_adoption_year=('asp_adoption_year', 'first'),
    age=('age', 'mean'),
    comorbidity_score=('comorbidity_score', 'mean'),
    severity_score=('severity_score', 'mean'),
    n_patients=('patient_id', 'count')
)

# Plain TWFE on the hospital-year panel
twfe_mod = smf.ols('antibiotic_duration ~ asp_active + C(hospital_id) + C(year)', data=did_panel).fit(
    cov_type='cluster', cov_kwds={'groups': did_panel['hospital_id']}
)
twfe_att_estimate = float(twfe_mod.params['asp_active'])

# Event study setup
did_panel['ever_treated'] = (did_panel['asp_adoption_year'] != 9999).astype(int)
did_panel['rel_time'] = np.where(
    did_panel['ever_treated'] == 1,
    did_panel['year'] - did_panel['asp_adoption_year'],
    -1
)

rel_times = sorted([int(rt) for rt in did_panel.loc[did_panel['ever_treated'] == 1, 'rel_time'].unique() if rt != -1])
es_terms = []
rt_map = {}
for rt in rel_times:
    col_name = f"rt_m{abs(rt)}" if rt < 0 else f"rt_p{rt}"
    did_panel[col_name] = ((did_panel['ever_treated'] == 1) & (did_panel['rel_time'] == rt)).astype(int)
    es_terms.append(col_name)
    rt_map[rt] = col_name

es_formula = 'antibiotic_duration ~ ' + ' + '.join(es_terms) + ' + C(hospital_id) + C(year)'
es_mod = smf.ols(es_formula, data=did_panel).fit(
    cov_type='cluster', cov_kwds={'groups': did_panel['hospital_id']}
)

es_rows = []
for rt in rel_times:
    col = rt_map[rt]
    ci = es_mod.conf_int().loc[col]
    es_rows.append({
        'event_time': int(rt),
        'coef': float(es_mod.params[col]),
        'ci_low': float(ci[0]),
        'ci_high': float(ci[1])
    })
event_study = pd.DataFrame(es_rows).sort_values('event_time').reset_index(drop=True)

# Pre-trend test using lead event_time = -2 (adjacent to reference -1)
lead_col = rt_map[-2] if -2 in rt_map else rt_map[min([r for r in rel_times if r < 0])]
pre_trend_coeff = float(es_mod.params[lead_col])
pre_trend_pvalue = float(es_mod.pvalues[lead_col])
pre_trend_ci = [float(x) for x in es_mod.conf_int().loc[lead_col].values]

# Staggered-adoption robust DiD (Callaway-Sant'Anna style cohort-time ATT aggregation using never-treated controls)
never_treated = did_panel[did_panel['asp_adoption_year'] == 9999]
cohorts = [2020, 2021]
att_gt_list = []
weight_list = []

for g in cohorts:
    base_year = g - 1
    treated_g = did_panel[did_panel['asp_adoption_year'] == g]
    y_t_base = treated_g.loc[treated_g['year'] == base_year, 'antibiotic_duration'].mean()
    y_c_base = never_treated.loc[never_treated['year'] == base_year, 'antibiotic_duration'].mean()
    
    for t in [y for y in sorted(did_panel['year'].unique()) if y >= g]:
        y_t_curr = treated_g.loc[treated_g['year'] == t, 'antibiotic_duration'].mean()
        y_c_curr = never_treated.loc[never_treated['year'] == t, 'antibiotic_duration'].mean()
        
        att_gt = (y_t_curr - y_t_base) - (y_c_curr - y_c_base)
        n_treated_hospitals = len(treated_g[treated_g['year'] == t])
        
        att_gt_list.append(att_gt)
        weight_list.append(n_treated_hospitals)

did_att_estimate = float(np.average(att_gt_list, weights=weight_list))

# ==========================================
# 3. TREATMENT HETEROGENEITY (T-Learner)
# ==========================================
feature_cols = ['age', 'comorbidity_score', 'immune_compromised', 'severity_score', 
                'sex', 'ward', 'infection_type', 'hospital_size', 'region', 'hospital_id', 'year']
X_raw = df[feature_cols].copy()
X_enc = pd.get_dummies(X_raw, columns=['sex', 'ward', 'infection_type', 'hospital_size', 'region', 'hospital_id', 'year'], drop_first=True, dtype=float)

y = df['antibiotic_duration'].values
T = df['asp_active'].values

m0 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
m1 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)

m0.fit(X_enc[T == 0], y[T == 0])
m1.fit(X_enc[T == 1], y[T == 1])

cate_estimates = np.asarray(m1.predict(X_enc) - m0.predict(X_enc), dtype=float)

# Bootstrap 95% CI for mean CATE(age >= 65) - mean CATE(age < 40)
idx_elderly = np.where(df['age'].values >= 65)[0]
idx_young = np.where(df['age'].values < 40)[0]

rng = np.random.default_rng(42)
boot_diffs = np.empty(1000, dtype=float)
for b in range(1000):
    sample_e = rng.choice(cate_estimates[idx_elderly], size=len(idx_elderly), replace=True)
    sample_y = rng.choice(cate_estimates[idx_young], size=len(idx_young), replace=True)
    boot_diffs[b] = np.mean(sample_e) - np.mean(sample_y)

cate_elderly_minus_young_ci = [float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))]

# ==========================================
# 4. HIERARCHICAL MODELING
# ==========================================
fe_formula = ('antibiotic_duration ~ asp_active + age + C(sex) + comorbidity_score + '
              'immune_compromised + severity_score + C(infection_type) + C(ward) + C(year)')

mixed_mod = smf.mixedlm(fe_formula, data=df, groups=df['hospital_id']).fit(reml=False)
naive_ols = smf.ols(fe_formula, data=df).fit()

random_effect_variance = float(mixed_mod.cov_re.iloc[0, 0])
residual_variance = float(mixed_mod.scale)
icc_value = float(random_effect_variance / (random_effect_variance + residual_variance))

mixed_model_aic = float(mixed_mod.aic)
naive_ols_aic = float(naive_ols.aic)
mixed_asp_coeff = float(mixed_mod.fe_params['asp_active'])