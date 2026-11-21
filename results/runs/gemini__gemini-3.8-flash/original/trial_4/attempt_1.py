from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test
from sklearn.ensemble import GradientBoostingRegressor

# -------------------------------------------------------------
# 1. SURVIVAL ANALYSIS
# -------------------------------------------------------------
kmf = KaplanMeierFitter()
kmf.fit(df['time_to_improvement'], event_observed=df['event_observed'])

cph_df = df[['time_to_improvement', 'event_observed', 'asp_active', 'hospital_id']].copy()
cph = CoxPHFitter()
cph.fit(
    cph_df,
    duration_col='time_to_improvement',
    event_col='event_observed',
    cluster_col='hospital_id'
)

cox_hr_asp = float(cph.hazard_ratios_['asp_active'])
cox_se_asp = float(cph.standard_errors_['asp_active'])

ph_test = proportional_hazard_test(cph, cph_df, time_transform='rank')
if hasattr(ph_test, 'summary') and 'p' in ph_test.summary.columns:
    ph_test_pvalues = {str(k): float(v) for k, v in ph_test.summary['p'].items()}
else:
    names = getattr(ph_test, 'name', ['asp_active'])
    if isinstance(names, str):
        names = [names]
    pvals = ph_test.p_value if not np.isscalar(ph_test.p_value) else [ph_test.p_value]
    ph_test_pvalues = {str(k): float(v) for k, v in zip(names, pvals)}

# -------------------------------------------------------------
# 2. CAUSAL IMPACT (DiD)
# -------------------------------------------------------------
did_panel = df.groupby(['hospital_id', 'year']).agg(
    antibiotic_duration=('antibiotic_duration', 'mean'),
    asp_active=('asp_active', 'first'),
    asp_adoption_year=('asp_adoption_year', 'first')
).reset_index()

twfe_mod = smf.ols('antibiotic_duration ~ asp_active + C(hospital_id) + C(year)', data=did_panel).fit()
twfe_att_estimate = float(twfe_mod.params['asp_active'])

treated_mask = did_panel['asp_adoption_year'] < 9000
did_panel['rel_year'] = np.where(
    treated_mask,
    did_panel['year'] - did_panel['asp_adoption_year'],
    np.nan
)

unique_rels = sorted([int(r) for r in did_panel.loc[treated_mask, 'rel_year'].unique() if r != -1])

ev_cols = []
for r in unique_rels:
    col = f'ev_{r}'.replace('-', 'm')
    did_panel[col] = np.where(treated_mask & (did_panel['rel_year'] == r), 1, 0)
    ev_cols.append((r, col))

es_formula = 'antibiotic_duration ~ ' + ' + '.join([col for _, col in ev_cols]) + ' + C(hospital_id) + C(year)'
es_mod = smf.ols(es_formula, data=did_panel).fit()

conf_ints = es_mod.conf_int()
es_records = []
for r, col in ev_cols:
    coef_val = float(es_mod.params[col])
    ci_low = float(conf_ints.loc[col, 0])
    ci_high = float(conf_ints.loc[col, 1])
    es_records.append({
        'event_time': r,
        'coef': coef_val,
        'ci_low': ci_low,
        'ci_high': ci_high
    })
event_study = pd.DataFrame(es_records).sort_values('event_time').reset_index(drop=True)

lead_col = 'ev_m2' if 'ev_m2' in es_mod.params else ev_cols[0][1]
pre_trend_coeff = float(es_mod.params[lead_col])
pre_trend_pvalue = float(es_mod.pvalues[lead_col])
pre_trend_ci = [float(conf_ints.loc[lead_col, 0]), float(conf_ints.loc[lead_col, 1])]

# Callaway-Sant'Anna robust DiD estimator using never-treated controls
never_panel = did_panel[did_panel['asp_adoption_year'] == 9999]
att_list = []
w_list = []
for g in [2020, 2021]:
    g_panel = did_panel[did_panel['asp_adoption_year'] == g]
    n_hosp = len(g_panel['hospital_id'].unique())
    base_y_g = g_panel[g_panel['year'] == (g - 1)]['antibiotic_duration'].mean()
    base_y_c = never_panel[never_panel['year'] == (g - 1)]['antibiotic_duration'].mean()
    for t in sorted(g_panel[g_panel['year'] >= g]['year'].unique()):
        y_g_t = g_panel[g_panel['year'] == t]['antibiotic_duration'].mean()
        y_c_t = never_panel[never_panel['year'] == t]['antibiotic_duration'].mean()
        att_gt = (y_g_t - base_y_g) - (y_c_t - base_y_c)
        att_list.append(att_gt)
        w_list.append(n_hosp)

did_att_estimate = float(np.average(att_list, weights=w_list))

# -------------------------------------------------------------
# 3. TREATMENT HETEROGENEITY (CATE)
# -------------------------------------------------------------
feature_cols = [
    'age', 'comorbidity_score', 'immune_compromised', 'severity_score',
    'hospital_id', 'year', 'hospital_size', 'region', 'ward', 'sex', 'infection_type'
]
X = pd.get_dummies(df[feature_cols], columns=['hospital_id', 'year', 'hospital_size', 'region', 'ward', 'sex', 'infection_type'], drop_first=True)
y = df['antibiotic_duration'].values
w = df['asp_active'].values

m0 = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
m1 = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)

m0.fit(X[w == 0], y[w == 0])
m1.fit(X[w == 1], y[w == 1])

cate_estimates = np.asarray(m1.predict(X) - m0.predict(X), dtype=float).flatten()

np.random.seed(42)
elderly_idx = (df['age'] >= 65).values
young_idx = (df['age'] < 40).values
n_obs = len(df)
n_boot = 1000
boot_diffs = np.empty(n_boot)

for b in range(n_boot):
    b_idx = np.random.choice(n_obs, size=n_obs, replace=True)
    mean_elderly = cate_estimates[b_idx][elderly_idx[b_idx]].mean()
    mean_young = cate_estimates[b_idx][young_idx[b_idx]].mean()
    boot_diffs[b] = mean_elderly - mean_young

cate_elderly_minus_young_ci = [
    float(np.percentile(boot_diffs, 2.5)),
    float(np.percentile(boot_diffs, 97.5))
]

# -------------------------------------------------------------
# 4. HIERARCHICAL MODELING
# -------------------------------------------------------------
naive_ols = smf.ols('antibiotic_duration ~ asp_active', data=df).fit()
naive_ols_aic = float(naive_ols.aic)

mixed_md = smf.mixedlm('antibiotic_duration ~ asp_active', data=df, groups=df['hospital_id'])
mixed_res = mixed_md.fit(reml=False)

mixed_asp_coeff = float(mixed_res.params['asp_active'])
mixed_model_aic = float(mixed_res.aic)

random_effect_variance = float(mixed_res.cov_re.iloc[0, 0])
residual_variance = float(mixed_res.scale)
icc_value = float(random_effect_variance / (random_effect_variance + residual_variance))