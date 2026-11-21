from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from linearmodels.panel import PanelOLS
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ==============================================================================
# 1. SURVIVAL ANALYSIS: Kaplan-Meier & Cox PH with hospital-clustered SEs
# ==============================================================================
kmf_treated = KaplanMeierFitter()
kmf_control = KaplanMeierFitter()
mask_treated = df['asp_active'] == 1
kmf_treated.fit(
    df.loc[mask_treated, 'time_to_improvement'],
    event_observed=df.loc[mask_treated, 'event_observed'],
    label='ASP Active',
)
kmf_control.fit(
    df.loc[~mask_treated, 'time_to_improvement'],
    event_observed=df.loc[~mask_treated, 'event_observed'],
    label='No ASP',
)

cox_cat_cols = ['sex', 'ward', 'infection_type', 'hospital_size', 'region', 'year']
cox_dummies = pd.get_dummies(
    df[cox_cat_cols].astype(str),
    drop_first=True,
    dtype=float,
)
cox_num_cols = [
    'asp_active',
    'age',
    'comorbidity_score',
    'immune_compromised',
    'severity_score',
]
cox_df = pd.concat(
    [
        df[['time_to_improvement', 'event_observed', 'hospital_id']].copy(),
        df[cox_num_cols].astype(float),
        cox_dummies,
    ],
    axis=1,
)

cph = CoxPHFitter()
try:
    cph.fit(
        cox_df,
        duration_col='time_to_improvement',
        event_col='event_observed',
        cluster_col='hospital_id',
    )
except Exception:
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(
        cox_df,
        duration_col='time_to_improvement',
        event_col='event_observed',
        cluster_col='hospital_id',
    )

cox_hr_asp = float(cph.hazard_ratios_['asp_active'])
cox_se_asp = float(cph.standard_errors_['asp_active'])

ph_results = proportional_hazard_test(cph, cox_df, time_transform='rank')
ph_test_pvalues = {
    str(cov): float(pval)
    for cov, pval in ph_results.summary['p'].items()
}

# ==============================================================================
# 2. CAUSAL IMPACT (DiD): Panel, Pre-trends, TWFE, and Staggered Event Study
# ==============================================================================
did_panel = (
    df.groupby(['hospital_id', 'year'], as_index=False)
    .agg(
        antibiotic_duration=('antibiotic_duration', 'mean'),
        asp_active=('asp_active', 'max'),
        asp_adoption_year=('asp_adoption_year', 'first'),
    )
    .sort_values(['hospital_id', 'year'])
    .reset_index(drop=True)
)

panel_indexed = did_panel.set_index(['hospital_id', 'year'])
twfe_mod = PanelOLS.from_formula(
    'antibiotic_duration ~ asp_active + EntityEffects + TimeEffects',
    data=panel_indexed,
)
twfe_res = twfe_mod.fit(cov_type='clustered', cluster_entity=True)
twfe_att_estimate = float(twfe_res.params['asp_active'])

# Sun-Abraham (2021) interaction-weighted estimator using never-treated controls
es_df = did_panel.copy()
treated_cohorts = sorted(
    [int(g) for g in es_df['asp_adoption_year'].unique() if int(g) != 9999]
)
cohort_sizes = (
    es_df.groupby('hospital_id')['asp_adoption_year']
    .first()
    .astype(int)
    .value_counts()
    .to_dict()
)

es_df['event_time'] = np.where(
    es_df['asp_adoption_year'] != 9999,
    es_df['year'] - es_df['asp_adoption_year'],
    np.nan,
)

interaction_cols = []
cell_meta = []
for g in treated_cohorts:
    rel_times = sorted(
        es_df.loc[es_df['asp_adoption_year'] == g, 'event_time']
        .dropna()
        .astype(int)
        .unique()
    )
    for e in rel_times:
        if e == -1:
            continue
        col_name = f'd_g{g}_e{e}'.replace('-', 'm')
        es_df[col_name] = (
            (es_df['asp_adoption_year'] == g) & (es_df['event_time'] == e)
        ).astype(float)
        interaction_cols.append(col_name)
        cell_meta.append((col_name, g, e))

sa_formula = (
    'antibiotic_duration ~ '
    + ' + '.join(interaction_cols)
    + ' + C(hospital_id) + C(year)'
)
sa_res = smf.ols(sa_formula, data=es_df).fit(
    cov_type='cluster',
    cov_kwds={'groups': es_df['hospital_id']},
)

param_index = {name: i for i, name in enumerate(sa_res.params.index)}
n_params = len(sa_res.params)

all_event_times = sorted(set([e for _, _, e in cell_meta] + [-1]))
event_study_rows = []
for e in all_event_times:
    if e == -1:
        event_study_rows.append(
            {'event_time': -1, 'coef': 0.0, 'ci_low': 0.0, 'ci_high': 0.0}
        )
        continue
    cells_e = [(col, g) for col, g, e_val in cell_meta if e_val == e]
    total_w = sum(cohort_sizes[g] for _, g in cells_e)
    r_vec = np.zeros((1, n_params), dtype=float)
    for col, g in cells_e:
        r_vec[0, param_index[col]] = cohort_sizes[g] / total_w
    tt = sa_res.t_test(r_vec)
    ci = np.asarray(tt.conf_int()).reshape(-1)
    event_study_rows.append(
        {
            'event_time': int(e),
            'coef': float(np.asarray(tt.effect).squeeze()),
            'ci_low': float(ci[0]),
            'ci_high': float(ci[1]),
        }
    )

event_study = (
    pd.DataFrame(event_study_rows)
    .sort_values('event_time')
    .reset_index(drop=True)
)

# Pre-trend test across identifiable pre-period leads (event_time < -1)
pre_cells = [(col, g) for col, g, e_val in cell_meta if e_val < -1]
total_pre_w = sum(cohort_sizes[g] for _, g in pre_cells)
r_pre = np.zeros((1, n_params), dtype=float)
for col, g in pre_cells:
    r_pre[0, param_index[col]] = cohort_sizes[g] / total_pre_w
tt_pre = sa_res.t_test(r_pre)
ci_pre = np.asarray(tt_pre.conf_int()).reshape(-1)
pre_trend_coeff = float(np.asarray(tt_pre.effect).squeeze())
pre_trend_pvalue = float(np.asarray(tt_pre.pvalue).squeeze())
pre_trend_ci = [float(ci_pre[0]), float(ci_pre[1])]

# Overall staggered-adoption ATT across post-treatment periods (event_time >= 0)
post_cells = [(col, g) for col, g, e_val in cell_meta if e_val >= 0]
total_post_w = sum(cohort_sizes[g] for _, g in post_cells)
r_post = np.zeros((1, n_params), dtype=float)
for col, g in post_cells:
    r_post[0, param_index[col]] = cohort_sizes[g] / total_post_w
tt_post = sa_res.t_test(r_post)
did_att_estimate = float(np.asarray(tt_post.effect).squeeze())

# ==============================================================================
# 3. TREATMENT HETEROGENEITY: Cross-fitted T-Learner with hospital & year adjustment
# ==============================================================================
X_hosp = pd.get_dummies(df['hospital_id'], prefix='hosp', drop_first=True, dtype=float)
X_year = pd.get_dummies(df['year'], prefix='yr', drop_first=True, dtype=float)
X_pat = pd.concat(
    [
        df[['age', 'comorbidity_score', 'immune_compromised', 'severity_score']].astype(float),
        pd.get_dummies(
            df[['sex', 'ward', 'infection_type']], drop_first=True, dtype=float
        ),
    ],
    axis=1,
)
X_hosp_chars = pd.get_dummies(
    df[['hospital_size', 'region']], drop_first=True, dtype=float
)
X_learner_full = pd.concat([X_pat, X_hosp_chars], axis=1)

W_arr = df['asp_active'].to_numpy(dtype=int)
y_arr = df['antibiotic_duration'].to_numpy(dtype=float)
age_arr = df['age'].to_numpy(dtype=float)

# Build design matrix to partial out additive hospital and year fixed effects
X_fe_block = np.hstack([X_hosp.to_numpy(), X_year.to_numpy()])
n_fe_cols = X_fe_block.shape[1]
X_pat_mat = X_pat.to_numpy()
W_col = W_arr.reshape(-1, 1).astype(float)
X_adj_design = np.hstack(
    [
        np.ones((len(df), 1), dtype=float),
        X_fe_block,
        X_pat_mat,
        W_col,
        W_col * X_pat_mat,
    ]
)


def compute_fe_adjusted_outcome(design_mat, fe_mat, y_vec, n_fe):
    coef, _, _, _ = np.linalg.lstsq(design_mat, y_vec, rcond=None)
    fe_coef = coef[1 : 1 + n_fe]
    return y_vec - fe_mat @ fe_coef


y_adj = compute_fe_adjusted_outcome(X_adj_design, X_fe_block, y_arr, n_fe_cols)
X_learner_mat = X_learner_full.to_numpy()

# 5-fold cross-fitted T-learner for out-of-fold CATE estimates
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mu0_oof = np.zeros(len(df), dtype=float)
mu1_oof = np.zeros(len(df), dtype=float)

gbr_params = dict(
    n_estimators=80,
    max_depth=3,
    learning_rate=0.06,
    min_samples_leaf=25,
    subsample=0.8,
    random_state=42,
)

for train_idx, val_idx in kf.split(X_learner_mat):
    w_tr = W_arr[train_idx]
    m0 = GradientBoostingRegressor(**gbr_params)
    m1 = GradientBoostingRegressor(**gbr_params)
    m0.fit(X_learner_mat[train_idx[w_tr == 0]], y_adj[train_idx[w_tr == 0]])
    m1.fit(X_learner_mat[train_idx[w_tr == 1]], y_adj[train_idx[w_tr == 1]])
    mu0_oof[val_idx] = m0.predict(X_learner_mat[val_idx])
    mu1_oof[val_idx] = m1.predict(X_learner_mat[val_idx])

cate_estimates = (mu1_oof - mu0_oof).astype(float)

# Bootstrap 95% CI for mean CATE(age >= 65) - mean CATE(age < 40)
rng = np.random.default_rng(42)
n_obs = len(df)
n_boot = 100
boot_diffs = np.empty(n_boot, dtype=float)

for b in range(n_boot):
    b_idx = rng.integers(0, n_obs, size=n_obs)
    w_b = W_arr[b_idx]
    y_b = y_arr[b_idx]
    age_b = age_arr[b_idx]
    y_adj_b = compute_fe_adjusted_outcome(
        X_adj_design[b_idx], X_fe_block[b_idx], y_b, n_fe_cols
    )
    X_b = X_learner_mat[b_idx]

    m0_b = GradientBoostingRegressor(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.06,
        min_samples_leaf=25,
        subsample=0.8,
        random_state=100 + b,
    )
    m1_b = GradientBoostingRegressor(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.06,
        min_samples_leaf=25,
        subsample=0.8,
        random_state=1000 + b,
    )
    m0_b.fit(X_b[w_b == 0], y_adj_b[w_b == 0])
    m1_b.fit(X_b[w_b == 1], y_adj_b[w_b == 1])
    cate_b = m1_b.predict(X_b) - m0_b.predict(X_b)
    boot_diffs[b] = cate_b[age_b >= 65].mean() - cate_b[age_b < 40].mean()

cate_elderly_minus_young_ci = [
    float(np.percentile(boot_diffs, 2.5)),
    float(np.percentile(boot_diffs, 97.5)),
]

# ==============================================================================
# 4. HIERARCHICAL MODELING: Hospital Random Intercept vs Naive OLS
# ==============================================================================
mixed_formula = (
    'antibiotic_duration ~ asp_active + age + C(sex) + comorbidity_score + '
    'immune_compromised + severity_score + C(ward) + C(infection_type) + C(year)'
)
mixed_mod = smf.mixedlm(mixed_formula, data=df, groups=df['hospital_id'])
mixed_res = mixed_mod.fit(reml=False)
if not mixed_res.converged:
    mixed_res = mixed_mod.fit(reml=False, method='bfgs')

random_effect_variance = float(mixed_res.cov_re.iloc[0, 0])
residual_variance = float(mixed_res.scale)
icc_value = float(
    random_effect_variance / (random_effect_variance + residual_variance)
)
mixed_asp_coeff = float(mixed_res.params['asp_active'])

if np.isfinite(mixed_res.aic):
    mixed_model_aic = float(mixed_res.aic)
else:
    mixed_model_aic = float(
        -2.0 * mixed_res.llf + 2.0 * (len(mixed_res.fe_params) + 2)
    )

ols_res = smf.ols(mixed_formula, data=df).fit()
naive_ols_aic = float(ols_res.aic)