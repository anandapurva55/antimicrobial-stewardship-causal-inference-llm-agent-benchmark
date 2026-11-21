"""Regenerate the three notebooks from source so they never drift from the scripts.

    python build_notebooks.py            # builds + executes golden and unit-test notebooks (outputs stored)
"""
import subprocess, sys
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell as md, new_code_cell as code

PY = sys.executable
sh = lambda *a: subprocess.run([PY, "-m", *a], check=True)

# 1. golden notebook = golden_solution.py (jupytext percent format), executed with stored outputs
sh("jupytext", "--to", "ipynb", "golden_solution.py", "-o", "golden_notebook.ipynb")
sh("jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
   "--ExecutePreprocessor.timeout=1800", "golden_notebook.ipynb")

# 2. unit-test notebook: thin wrapper over asp_tests.py
nb = new_notebook(cells=[
    md("# Unit Tests: ASP Causal Study\nScores a results file with `asp_tests.py`: 14 behavioural tests, each run independently "
       "(a failure never hides the others). No test trusts a self-declared flag; reference values are recomputed from "
       "`asp_study_data.csv` or compared with `ground_truth.json`.\n\nRun `golden_notebook.ipynb` or `run_agent.py` first to produce a results JSON."),
    code("RESULTS_FILE = 'golden_results.json'   # or runs/<backend>__<model>/<condition>/trial_<k>/results.json"),
    code("import pandas as pd\nfrom asp_common import load_results\nfrom asp_tests import run_tests, report, preconditions\n"
         "for tid, desc, ok in preconditions(pd.read_csv('asp_study_data.csv')):\n    print(tid, 'ok' if ok else 'BAD', '(unscored)', desc)\n"
         "rows = run_tests(load_results(RESULTS_FILE))\nk, n = report(rows)"),
])
nbf.write(nb, "unit_test_notebook.ipynb")
sh("jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", "unit_test_notebook.ipynb")

# 3. agent notebook: thin wrapper over run_agent.py (NOT executed here: it calls a paid API / local LLM)
nb = new_notebook(cells=[
    md("# Agent Notebook: ASP Causal Study\nRuns one trial of an LLM agent through `run_agent.py` and scores it. "
       "Generated code runs in a subprocess with a timeout (not `exec`), the model is pinned (no silent fallback) and "
       "every artifact (prompt, each attempt, errors, model name, score) is saved under `runs/`.\n\n"
       "Put `GEMINI_API_KEY` in `.env` (or export it from Colab Secrets).\n\n"
       "`CONDITION='original'` leaves the outcome of the mixed-model objective unstated; `'pinned'` names `antibiotic_duration` as the primary outcome for objectives 2-4."),
    code("BACKEND, MODEL = 'gemini', 'gemini-2.5-pro'\nCONDITION, TRIAL = 'original', 0   # 'original' | 'pinned'"),
    code("from run_agent import run_trial\nscore = run_trial(BACKEND, MODEL, CONDITION, TRIAL, temperature=0.7, max_attempts=5, exec_timeout=900)\n"
         "for t in score['tests']:\n    print(t['id'], 'PASS' if t['passed'] else 'FAIL', t['msg'])"),
    md("Aggregate across trials with `python summarize_runs.py`."),
])
nbf.write(nb, "agent_notebook.ipynb")
