"""Shared helpers: result (de)serialisation without pickle, required outputs, agent prompts."""
import json
import numpy as np
import pandas as pd

# name -> description shown to the agent. No boolean "I did X" flags: the tests recompute
# reference values from the data and compare, so nothing can be passed by self-declaration.
REQUIRED_OUTPUTS = {
    "cox_hr_asp": "float, hazard ratio of asp_active from a Cox PH model for time_to_improvement (event=event_observed)",
    "cox_se_asp": "float, standard error of the asp_active log-hazard coefficient, robust/clustered by hospital_id",
    "ph_test_pvalues": "dict {covariate_name: float p-value} from a Schoenfeld-residual proportional-hazards test",
    "did_panel": "DataFrame, exactly one row per hospital-year (15 hospitals x 6 years = 90 rows), with hospital_id, year and the mean primary-outcome per hospital-year",
    "pre_trend_coeff": "float, estimated pre-treatment (lead) coefficient from your event study / pre-trend test",
    "pre_trend_pvalue": "float, p-value for that pre-trend coefficient",
    "pre_trend_ci": "list [lo, hi], 95% CI of the pre-trend coefficient",
    "event_study": "DataFrame with columns event_time, coef, ci_low, ci_high (reference period event_time=-1 may be omitted); include pre-period leads where identifiable",
    "twfe_att_estimate": "float, ATT from a plain two-way-fixed-effects regression on the hospital-year panel",
    "did_att_estimate": "float, ATT (days) from an estimator robust to staggered adoption (e.g. Callaway-Sant'Anna / Sun-Abraham / stacked), using never-treated hospitals as controls",
    "cate_estimates": "numpy array, one individual treatment-effect estimate per patient (row order of df) from a T- or X-learner",
    "cate_elderly_minus_young_ci": "list [lo, hi], bootstrap 95% CI for mean CATE(age>=65) minus mean CATE(age<40)",
    "icc_value": "float, ICC of a hospital random-intercept model of the primary outcome",
    "mixed_model_aic": "float, AIC of the mixed model (ML fit)",
    "naive_ols_aic": "float, AIC of the same fixed effects without the random intercept",
    "mixed_asp_coeff": "float, asp_active coefficient from the mixed model",
    "random_effect_variance": "float, hospital random-intercept variance",
}


def _enc(v):
    if isinstance(v, pd.DataFrame):
        return {"__df__": json.loads(v.to_json(orient="records"))}
    if isinstance(v, np.ndarray):
        return {"__nd__": v.tolist()}
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, dict):
        return {str(k): _enc(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_enc(x) for x in v]
    return v


def _dec(v):
    if isinstance(v, dict) and "__df__" in v:
        return pd.DataFrame(v["__df__"])
    if isinstance(v, dict) and "__nd__" in v:
        return np.array(v["__nd__"], dtype=float)
    if isinstance(v, dict):
        return {k: _dec(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_dec(x) for x in v]
    return v


def save_results(path, ns):
    """Serialise the REQUIRED_OUTPUTS found in namespace `ns` to JSON (no pickle)."""
    out = {}
    for k in REQUIRED_OUTPUTS:
        if k in ns and ns[k] is not None:
            try:
                out[k] = _enc(ns[k])
            except Exception as e:  # unserialisable -> treated as missing
                out[k] = None
    with open(path, "w") as f:
        json.dump(out, f, default=str)
    return sorted(out)


def load_results(path):
    with open(path) as f:
        return {k: _dec(v) for k, v in json.load(f).items()}


# ---------------------------------------------------------------- agent prompt
def build_prompt(df, condition="original"):
    """condition 'original': objective 4 does not name the outcome (as in the first benchmark run).
    condition 'pinned'   : the primary outcome is stated once and required for objectives 2-4."""
    assert condition in ("original", "pinned")
    data_context = f"""Dataset shape : {df.shape[0]:,} rows x {df.shape[1]} columns
Columns       : {list(df.columns)}
Dtypes        :
{df.dtypes.to_string()}

First 5 rows:
{df.head(5).to_string()}

Unique hospitals: {df['hospital_id'].nunique()}; ASP adoption years: {sorted(df['asp_adoption_year'].unique())};
Years: {sorted(df['year'].unique())}; censoring rate: {1 - df['event_observed'].mean():.1%}
"""
    pinned = ""
    if condition == "pinned":
        pinned = ("- PRIMARY OUTCOME: antibiotic_duration. Use it as the outcome in objectives 2, 3 and 4 "
                  "(DiD, CATE and mixed model). Do not substitute another outcome.\n")
    outputs = "\n".join(f"- {k}: {v}" for k, v in REQUIRED_OUTPUTS.items())
    return f"""You are an expert data scientist specialising in clinical research and causal inference.

CONTEXT:
A consortium of 15 hospitals across five regions participated in a longitudinal observational
study (2017-2022) of Antimicrobial Stewardship Programs (ASPs). ASPs were voluntarily adopted at
different points (2020, 2021, or never). The DataFrame `df` is already loaded (one row per patient admission).

{data_context}
TASK: produce a rigorous causal analysis of ASP effectiveness. Address all four objectives.

1. SURVIVAL ANALYSIS: Quantify how ASP affects time-to-clinical-improvement, accounting for
   right-censored patients. Use Kaplan-Meier and a Cox PH model (SEs clustered by hospital_id).
   Test the proportional hazards assumption.

2. CAUSAL IMPACT (DiD): Estimate the causal impact of ASP on antibiotic_duration. Account for the
   voluntary, staggered nature of adoption. Build a hospital-year panel, test parallel pre-trends,
   run TWFE, and also run an estimator that is robust to staggered adoption. Produce an event study with CIs.

3. TREATMENT HETEROGENEITY: Estimate individual-level CATE with a T- or X-Learner using gradient boosting
   or random forests. Adjust for hospital and year, and bootstrap a CI for the elderly-vs-young difference.

4. HIERARCHICAL MODELING: Fit a mixed-effects linear model with hospital as a random intercept.
   Calculate and report the ICC. Compare AIC against naive OLS.

CONSTRAINTS:
- event_observed=0 means censored. Do NOT treat these as failures.
- Validate parallel trends before reporting DiD estimates.
- ICC = random_effect_variance / (random_effect_variance + residual_variance).
{pinned}- Available packages only: lifelines, linearmodels, statsmodels, sklearn, pandas, numpy, scipy.

REQUIRED OUTPUT VARIABLES (define exactly these names; they are recomputed and checked against the data):
{outputs}

Return ONLY valid, executable Python code. No markdown, no explanations, no code fences.
`df` and the imports pd, np, smf (statsmodels.formula.api), sm, lifelines classes (KaplanMeierFitter, CoxPHFitter),
PanelOLS, sklearn ensembles are already available.
"""
