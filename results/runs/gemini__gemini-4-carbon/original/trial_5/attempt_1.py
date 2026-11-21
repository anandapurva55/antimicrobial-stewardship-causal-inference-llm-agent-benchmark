import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from linearmodels.panel import PanelOLS
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

# =============================================================================
# 1. SURVIVAL ANALYSIS: Kaplan-Meier, Clustered Cox PH, Schoenfeld PH Test
# =============================================================================
kmf_treated = KaplanMeierFitter()
kmf_control = KaplanMeierFitter()

mask_asp_on = df["asp_active"] == 1
kmf_treated.fit(
    durations=df.loc[mask_asp_on, "time_to_improvement"],
    event_observed=df.loc[mask_asp_on, "event_observed"],
    label="ASP Active",
)
kmf_control.fit(
    durations=df.loc[~mask_asp_on, "time_to_improvement"],
    event_observed=df.loc[~mask_asp_on, "event_observed"],
    label="ASP Inactive",
)

cox_covariate_cols = [
    "asp_active",
    "age",
    "sex",
    "comorbidity_score",
    "immune_compromised",
    "infection_type",
    "severity_score",
    "ward",
    "hospital_size",
    "region",
    "year",
]
cox_X = pd.get_dummies(
    df[cox_covariate_cols],
    columns=["sex", "infection_type", "ward", "hospital_size", "region"],
    drop_first=True,
    dtype=float,
)
cox_df = cox_X.copy()
cox_df["time_to_improvement"] = df["time_to_improvement"].astype(float)
cox_df["event_observed"] = df["event_observed"].astype(int)
cox_df["hospital_id"] = df["hospital_id"]

cph = CoxPHFitter()
try:
    cph.fit(
        cox_df,
        duration_col="time_to_improvement",
        event_col="event_observed",
        cluster_col="hospital_id",
    )
except Exception:
    cph = CoxPHFitter(penalizer=1e-4)
    cph.fit(
        cox_df,
        duration_col="time_to_improvement",
        event_col="event_observed",
        cluster_col="hospital_id",
    )

cox_hr_asp = float(cph.hazard_ratios_["asp_active"])
cox_se_asp = float(cph.standard_errors_["asp_active"])

ph_res = proportional_hazard_test(cph, cox_df, time_transform="rank")
ph_test_pvalues = {str(cov): float(pval) for cov, pval in ph_res.summary["p"].items()}

# =============================================================================
# 2. CAUSAL IMPACT (DiD): Hospital-Year Panel, Pre-Trends, TWFE, Sun-Abraham/CS
# =============================================================================
did_panel = (
    df.groupby(["hospital_id", "year"], as_index=False)
    .agg(
        antibiotic_duration=("antibiotic_duration", "mean"),
        asp_active=("asp_active", "mean"),
        asp_adoption_year=("asp_adoption_year", "first"),
        hospital_size=("hospital_size", "first"),
        region=("region", "first"),
        n_patients=("patient_id", "count"),
    )
    .sort_values(["hospital_id", "year"])
    .reset_index(drop=True)
)
did_panel["asp_active"] = did_panel["asp_active"].round().astype(int)

# Plain TWFE on the hospital-year panel
panel_indexed = did_panel.set_index(["hospital_id", "year"])
twfe_res = PanelOLS.from_formula(
    "antibiotic_duration ~ asp_active + EntityEffects + TimeEffects",
    data=panel_indexed,
).fit(cov_type="clustered", cluster_entity=True)
twfe_att_estimate = float(twfe_res.params["asp_active"])

# Staggered-adoption-robust estimator (Sun-Abraham / Callaway-Sant'Anna equivalent
# on balanced panel with never-treated controls and universal base period k = -1)
sa_df = did_panel.copy()
treated_cohorts = sorted(
    [int(g) for g in sa_df["asp_adoption_year"].unique() if g < 9999]
)
all_years = sorted([int(y) for y in sa_df["year"].unique()])
hosp_counts_by_cohort = (
    sa_df.groupby("asp_adoption_year")["hospital_id"].nunique().to_dict()
)

sa_term_map = {}
sa_cols = []
for g in treated_cohorts:
    for y in all_years:
        rel_k = y - g
        if rel_k == -1:
            continue
        col_name = f"sa_g{g}_y{y}"
        sa_df[col_name] = (
            (sa_df["asp_adoption_year"] == g) & (sa_df["year"] == y)
        ).astype(float)
        sa_cols.append(col_name)
        sa_term_map[(g, y)] = col_name

sa_formula = (
    "antibiotic_duration ~ "
    + " + ".join(sa_cols)
    + " + C(hospital_id) + C(year)"
)
sa_fit = smf.ols(sa_formula, data=sa_df).fit(
    cov_type="cluster", cov_kwds={"groups": sa_df["hospital_id"]}
)
sa_param_names = list(sa_fit.params.index)

# Event study aggregation across cohorts at each relative time k != -1
rel_times = sorted(
    {y - g for (g, y) in sa_term_map.keys()}
)
es_rows = []
lead_minus2_stats = None

for k in rel_times:
    valid_cohorts = [g for g in treated_cohorts if (g, g + k) in sa_term_map]
    total_hosp = float(sum(hosp_counts_by_cohort[g] for g in valid_cohorts))
    L = np.zeros(len(sa_param_names))
    for g in valid_cohorts:
        col_name = sa_term_map[(g, g + k)]
        L[sa_param_names.index(col_name)] = hosp_counts_by_cohort[g] / total_hosp

    t_res = sa_fit.t_test(L)
    coef_k = float(np.asarray(t_res.effect).item())
    pval_k = float(np.asarray(t_res.pvalue).item())
    ci_k = np.asarray(t_res.conf_int(alpha=0.05)).reshape(-1)
    ci_lo, ci_hi = float(ci_k[0]), float(ci_k[1])

    es_rows.append(
        {
            "event_time": int(k),
            "coef": coef_k,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
        }
    )
    if k == -2:
        lead_minus2_stats = (coef_k, pval_k, [ci_lo, ci_hi])

event_study = pd.DataFrame(es_rows).sort_values("event_time").reset_index(drop=True)

pre_trend_coeff = float(lead_minus2_stats[0])
pre_trend_pvalue = float(lead_minus2_stats[1])
pre_trend_ci = [float(lead_minus2_stats[2][0]), float(lead_minus2_stats[2][1])]

# Overall staggered-robust ATT across all post-treatment (g, t >= g) cells
post_cells = [(g, y) for (g, y) in sa_term_map.keys() if y >= g]
total_post_weight = float(sum(hosp_counts_by_cohort[g] for (g, _) in post_cells))
L_att = np