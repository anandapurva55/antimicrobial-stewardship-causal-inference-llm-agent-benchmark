# Causal Evaluation of Antimicrobial Stewardship Programs Across a Hospital Network

A long-horizon benchmark task evaluating Gemini 2.5 Pro as an agentic data scientist
on expert-level causal inference applied to a real-world clinical scenario.

---

## Summary

| Field | Detail |
|---|---|
| Task Name | Multi-Hospital ASP Causal Study |
| Scenario | 15 hospitals, 4-year observational study (2019-2022), staggered voluntary ASP adoption, 11,629 patient admissions across 5 regions |
| Methods Required | Survival Analysis, DiD Estimation, Heterogeneous Treatment Effects, Mixed Effects Modeling |
| Agent Failure | Mixed Effects Modeling: Gemini used wrong outcome variable, producing ICC=0.009 vs true ICC=0.22 |
| Agent Result | 17/18 unit tests passed |

---

## Repository Structure

```
asp-causal-inference/
├── golden_notebook.ipynb      # Expert solution, passes all 18 unit tests
├── unit_test_notebook.ipynb   # Standalone unit test validation notebook
├── agent_notebook.ipynb       # Gemini agent attempt, demonstrates failure
├── asp_study_data.csv         # Synthetic dataset (11,629 patients x 22 columns)
├── requirements.txt           # All Python dependencies
├── .env.example               # Template for API key setup
└── .gitignore
```

---

## Problem Description and Significance

Antimicrobial resistance (AMR) causes an estimated 700,000 deaths annually and is
classified by the WHO as one of the ten greatest global health threats. Antimicrobial
Stewardship Programs (ASPs) are hospital-led governance frameworks that enforce
evidence-based antibiotic prescribing, restricting broad-spectrum use, shortening
treatment durations, and reducing C. difficile infections. Despite their importance,
rigorous causal evaluation of ASPs is rare because adoption is voluntary and staggered,
and better-resourced hospitals adopt earlier while having better baseline outcomes
regardless of ASP, creating confounding that naive comparisons cannot handle.

This study frames ASP evaluation as a long-horizon causal inference problem over a
four-year multi-site study (2019-2022) across 15 hospitals and 11,629 patient admissions.
The expert solution requires four interlinked methodologies:

1. **Survival Analysis:** Kaplan-Meier estimators and Cox Proportional Hazards
   correctly handle the 32.8% of patients who were right-censored at study end.
2. **Difference-in-Differences:** TWFE regression with mandatory parallel
   pre-trends validation controls for hospital confounders and secular trends.
3. **Heterogeneous Treatment Effects:** T-Learner and X-Learner meta-learners reveal that
   elderly patients gain 32% more benefit from ASP than younger patients.
4. **Mixed Effects Modeling:** A hospital random intercept quantifies
   ICC = 0.22, proving 22% of variance in antibiotic duration is attributable to
   between-hospital differences.

Completing this correctly requires 4-6 hours for a senior data scientist with expertise
in causal inference and biostatistics. It cannot be reduced to a standard ML workflow.

---

## Agent Performance and Failure Analysis

Gemini 2.5 Pro was evaluated using a multi-turn agentic loop (up to 5 attempts), where
the agent observed its own execution errors and corrected them iteratively. The agent
required 3 attempts to produce executing code, autonomously fixing two technical bugs:
a special-character column naming issue (`age_group_<40` causing a patsy formula error)
and an incorrect propensity model type in the X-Learner (regressor instead of classifier).

**Unit test results: 17 of 18 passed.**

Gemini passed all tests for Survival Analysis (A1-A6), Difference-in-Differences
(B1-B5), and Heterogeneous Treatment Effects (C1-C5). The Cox model correctly handled
censoring (HR=1.86), parallel trends were verified (p=0.95), TWFE ATT was accurately
estimated at -2.87 days, and individual-level CATE was produced via the X-Learner
(std=1.39), correctly showing elderly patients benefit more from ASP.

**The single failure: D1, ICC = 0.0091, expected range 0.02-0.50.**

Gemini computed the ICC using `length_of_stay` as the outcome for the mixed-effects
model, while the entire analysis centres on `antibiotic_duration`. The `length_of_stay`
variable has weak hospital-level clustering (ICC ~ 0.009) because it is driven by patient
acuity rather than prescribing practices. The true ICC for `antibiotic_duration` is 0.22,
reflecting the direct influence of hospital prescribing culture on treatment duration.

This is not a technical bug. It is a failure of cross-analysis coherence. A human
expert maintains a consistent primary outcome variable across all analytical sections.
Gemini lost track of the research question when transitioning to hierarchical modeling,
defaulting to `length_of_stay` despite the entire preceding analysis being anchored to
`antibiotic_duration`.

---

## How the Agent Could Be Improved

**Short-term: Outcome consistency enforcement**
A structured reasoning step requiring the agent to confirm the primary outcome variable
before each analytical section would prevent this failure. Prompt scaffolding that pins
the outcome variable globally and requires justification for any deviation would suffice.

**Medium-term: Causal inference fine-tuning**
The failure is in cross-section coherence, not individual methodology. Fine-tuning on
multi-step causal workflows where outcome variable consistency is evaluated across all
sections would address this directly.

**Long-term: Structured intermediate validation**
Agents could be trained to perform explicit sanity checks between analytical phases:
does the outcome in the mixed effects model match the outcome used in the DiD?
This mirrors iterative verification that distinguishes senior data scientists.

---

## How to Run

### Prerequisites

- Python 3.9+
- A Gemini API key (only needed for `agent_notebook.ipynb`)
  Get one free at [aistudio.google.com](https://aistudio.google.com)

### Local Setup

```bash
git clone https://github.com/anandapurva55/asp-causal-inference
cd asp-causal-inference

pip install -r requirements.txt

# Set up API key
cp .env.example .env
# Edit .env and add your key: GEMINI_API_KEY=your_key_here

jupyter notebook
```

### Running golden_notebook.ipynb (Expert Solution)

1. Open `golden_notebook.ipynb` in Jupyter or Colab
2. Upload `asp_study_data.csv` to the session if using Colab
3. Run All. Results are saved to `golden_results.pkl` at the end.

### Running unit_test_notebook.ipynb (Validation)

1. Run `golden_notebook.ipynb` or `agent_notebook.ipynb` to generate a pkl file
2. Open `unit_test_notebook.ipynb`
3. Set `PKL_FILE` to the correct filename (`golden_results.pkl` or `agent_results.pkl`)
4. Run All.

### Running agent_notebook.ipynb (Gemini Agent Demo)

**Local:**
```bash
# Ensure .env has your GEMINI_API_KEY
jupyter notebook agent_notebook.ipynb
# Run All
```

**Colab:**
1. Open `agent_notebook.ipynb` in Colab
2. Add `GEMINI_API_KEY` to Colab Secrets (key icon in left sidebar)
3. Upload `asp_study_data.csv` and `requirements.txt` to the session
4. Run All. Results are saved to `agent_results.pkl`.

---

## Dataset

`asp_study_data.csv`: 11,629 rows x 22 columns (synthetic, generated with fixed seed 42)

| Column | Description |
|---|---|
| `patient_id` | Unique patient identifier |
| `hospital_id` | Hospital (1-15) |
| `year` | Admission year (2019-2022) |
| `asp_active` | 1 if ASP was active at admission, 0 otherwise |
| `asp_adoption_year` | Year hospital adopted ASP (9999 = never) |
| `time_to_improvement` | Days until clinical improvement (survival time) |
| `event_observed` | 1 = improved, 0 = right-censored |
| `antibiotic_duration` | Days of antibiotic treatment (primary outcome) |
| `age`, `sex`, `age_group` | Patient demographics |
| `severity_score` | Clinical severity (10-100) |
| `comorbidity_score` | Charlson comorbidity index (0-8) |
| `immune_compromised` | Binary immunocompromised flag |
| `infection_type` | respiratory / urinary / bloodstream / wound |
| `ward` | ICU / general / surgical |
| `mortality` | In-hospital mortality (binary) |
| `readmission_30d` | 30-day readmission (binary) |
| `c_diff_infection` | C. difficile infection (binary) |
| `length_of_stay` | Total length of stay (days) |

---

## Methods

| Method | Why Required |
|---|---|
| Survival Analysis | 32.8% of patients right-censored, naive analysis is biased |
| DiD Estimation | Staggered voluntary adoption requires TWFE and parallel trends test |
| Heterogeneous Treatment Effects | Population ATT hides patient-level heterogeneity |
| Mixed Effects Modeling | Patients nested in hospitals, ICC must be modelled |
