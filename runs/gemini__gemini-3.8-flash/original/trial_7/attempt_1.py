from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from sklearn.ensemble import GradientBoostingRegressor

# -------------------------------------------------------------
# 1. SURVIVAL ANALYSIS
# -------------------------------------------------------------
kmf = KaplanMeierFitter()
kmf.fit(df['time_to_improvement'], event_observed=df['event_observed'])

cox_df = df[['time_to_improvement', 'event_observed', 'asp_active', 'hospital_id']].copy()
cox_df['time_to_improvement'] = cox_df['time_to_improvement'].clip(lower=0.01)

cph = CoxPHFitter()
cph.fit(cox_df, duration_col='time_to_improvement', event_col='event_observed', cluster_col='hospital_id')

cox_hr_asp = float(cph.hazard_ratios_['asp_active'])
cox_se_asp = float(cph.standard_errors_['asp_active'])

try:
    ph_test = proportional_hazard_test(cph, cox_df, time_transform='rank')
except Exception:
    ph_test = proportional_hazard_test(cph, cox_df.drop(columns=['hospital_id']), time_transform='rank')

ph_test_pvalues = {str(cov): float(p) for cov, p in ph_test.summary['p'].items()}

# -------------------------------------------------------------
# 2. CAUSAL IMPACT (DiD)
# -------------------------------------------------------------
did_panel = df.groupby(['hospital_id', 'year'], as_index=False).agg(
    antibiotic_duration=('antibiotic_duration', 'mean'),
    asp_active=('asp_active', 'first'),
    asp_adoption_year=('asp_adoption_year', 'first')
)

twfe_model = smf.ols("antibiotic_duration ~ asp_active + C(hospital_id) + C(year)", data=did_panel).fit()
twfe_att_estimate = float(twfe_model.params['asp_active'])

did_panel['event_time'] = np.where(
    did_panel['asp_adoption_year'] < 9000,
    did_panel['year'] - did_panel['asp_adoption_year'],
    np.nan
)

event_times = sorted([int(e) for e in did_panel['event_time'].dropna().unique() if e != -1])
for e in event_times:
    did_panel[f'event_{e}'] = (did_panel['event_time'] == e).astype(int)

formula_es = "antibiotic_duration ~ " + " + ".join([f"Q('event_{e}')" for e in event_times]) + " + C(hospital_id) + C(year)"
try:
    es_model = smf.ols(formula_es, data=did_panel).fit(cov_type='cluster', cov_kwds={'groups': did_panel['hospital_id']})
except Exception:
    es_model = smf.ols(formula_es, data=did_panel).fit()

conf = es_model.conf_int()
es_rows = []
for e in event_times:
    var_name = f"Q('event_{e}')"
    es_rows.append({
        'event_time': int(e),
        'coef': float(es_model.params[var_name]),
        'ci_low': float(conf.loc[var_name, 0]),
        'ci_high': float(conf.loc[var_name, 1])
    })
event_study = pd.DataFrame(es_rows)

lead_var = "Q('event_-2')"
pre_trend_coeff = float(es_model.params[lead_var])
pre_trend_pvalue = float(es_model.pvalues[lead_var])
pre_trend_ci = [float(conf.loc[lead_var, 0]), float(conf.loc[lead_var, 1])]

# Callaway-Sant'Anna robust DiD estimator
cohort_year_means = did_panel.groupby(['asp_adoption_year', 'year'])['antibiotic_duration'].mean()
n_hosp = did_panel.groupby('asp_adoption_year')['hospital_id'].nunique()

cs_atts = []
cs_weights = []
for g in [2020, 2021]:
    t_base = g - 1
    for t in range(g, 2023):
        att_gt = (cohort_year_means.loc[(g, t)] - cohort_year_means.loc[(g, t_base)]) - \
                 (cohort_year_means.loc[(9999, t)] - cohort_year_means.loc[(9999, t_base)])
        cs_atts.append(att_gt)
        cs_weights.append(n_hosp[g])

did_att_estimate = float(np.average(cs_atts, weights=cs_weights))

# -------------------------------------------------------------
# 3. TREATMENT HETEROGENEITY (T-Learner)
# -------------------------------------------------------------
feature_cols = [
    'age', 'severity_score', 'comorbidity_score', 'immune_compromised',
    'hospital_id', 'year', 'sex', 'ward', 'infection_type', 'hospital_size'
]
X = pd.get_dummies(df[feature_cols], columns=['hospital_id', 'year', 'sex', 'ward', 'infection_type', 'hospital_size'], drop_first=True)
y = df['antibiotic_duration'].values
w = df['asp_active'].values

m0 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
m1 = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)

m0.fit(X[w == 0], y[w == 0])
m1.fit(X[w == 1], y[w == 1])

cate_estimates = m1.predict(X) - m0.predict(X)

elderly_idx = np.where(df['age'] >= 65)[0]
young_idx = np.where(df['age'] < 40)[0]

np.random.seed(42)
n_boot = 1000
boot_diffs = np.empty(n_boot)
for b in range(n_boot):
    b_elderly = np.random.choice(cate_estimates[elderly_idx], size=len(elderly_idx), replace=True).mean()
    b_young = np.random.choice(cate_estimates[young_idx], size=len(young_idx), replace=True).mean()
    boot_diffs[b] = b_elderly - b_young

cate_elderly_minus_young_ci = [
    float(np.percentile(boot_diffs, 2.5)),
    float(np.percentile(boot_diffs, 97.5))
]

# -------------------------------------------------------------
# 4. HIERARCHICAL MODELING
# -------------------------------------------------------------
mixed_mod = smf.mixedlm("antibiotic_duration ~ asp_active", data=df, groups=df["hospital_id"])
mixed_res = mixed_mod.fit(reml=False)

mixed_asp_coeff = float(mixed_res.fe_params['asp_active'])
random_effect_variance = float(mixed_res.cov_re.iloc[0, 0])
residual_variance = float(mixed_res.scale)
icc_value = float(random_effect_variance / (random_effect_variance + residual_variance))

if hasattr(mixed_res, 'aic') and mixed_res.aic is not None and not np.isnan(mixed_res.aic):
    mixed_model_aic = float(mixed_res.aic)
else:
    k_params = len(mixed_res.fe_params) + 2
    mixed_model_aic = float(-2 * mixed_res.llf + 2 * k_params)

naive_ols = smf.ols("antibiotic_duration ~ asp_active", data=df).fit()
naive_ols_aic = float(naive_ols.aic)