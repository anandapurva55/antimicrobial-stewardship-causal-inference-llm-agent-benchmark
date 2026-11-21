"""Regenerate the three notebooks from source so they never drift from the scripts.

    python scripts/build_notebooks.py    # builds the notebooks and executes two of them (outputs are stored)
"""
import subprocess
import sys

import nbformat as nbf
from nbformat.v4 import new_code_cell as code
from nbformat.v4 import new_markdown_cell as md
from nbformat.v4 import new_notebook

from asp_benchmark.paths import ROOT

NOTEBOOKS = ROOT / "notebooks"
GOLDEN_SOURCE = ROOT / "src" / "asp_benchmark" / "golden.py"


def sh(*args):
    subprocess.run([sys.executable, "-m", *args], check=True, cwd=ROOT)


# 1. Golden notebook: golden.py in jupytext percent format, executed so the outputs are stored.
sh("jupytext", "--to", "ipynb", str(GOLDEN_SOURCE), "-o", str(NOTEBOOKS / "golden_notebook.ipynb"))
sh("jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
   "--ExecutePreprocessor.timeout=1800", str(NOTEBOOKS / "golden_notebook.ipynb"))

# 2. Unit-test notebook: a thin wrapper over asp_benchmark.scoring.
nb = new_notebook(cells=[
    md("# Unit Tests: Antimicrobial Stewardship Causal Study\nScores a results file with `asp_benchmark.scoring`: 14 behavioural tests, "
       "each run independently (a failure never hides the others). No test trusts a self-declared flag; reference values are "
       "recomputed from `data/asp_study_data.csv` or compared with `data/ground_truth.json`.\n\n"
       "Run `golden_notebook.ipynb` or `evaluation/run_agent.py` first to produce a results JSON."),
    code("from asp_benchmark.paths import GOLDEN_RESULTS\n\n"
         "RESULTS_FILE = GOLDEN_RESULTS   # or results/runs/<backend>__<model>/<condition>/trial_<k>/results.json"),
    code("import pandas as pd\n"
         "from asp_benchmark.common import load_results\n"
         "from asp_benchmark.paths import CSV\n"
         "from asp_benchmark.scoring import preconditions, report, run_tests\n\n"
         "for tid, desc, ok in preconditions(pd.read_csv(CSV)):\n"
         "    print(tid, 'ok' if ok else 'BAD', '(unscored)', desc)\n"
         "rows = run_tests(load_results(RESULTS_FILE))\n"
         "k, n = report(rows)"),
])
nbf.write(nb, NOTEBOOKS / "unit_test_notebook.ipynb")
sh("jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", str(NOTEBOOKS / "unit_test_notebook.ipynb"))

# 3. Agent notebook: a thin wrapper over asp_benchmark.agent (not executed here because it calls a paid API).
nb = new_notebook(cells=[
    md("# Agent Notebook: Antimicrobial Stewardship Causal Study\nRuns one trial of a Gemini agent and scores it. "
       "Generated code runs in a separate process with a time limit (not `exec`), the model is pinned (no silent fallback) and "
       "every artifact (prompt, each attempt, errors, model name, score) is saved under `results/runs/`.\n\n"
       "Put `GEMINI_API_KEY` in `.env`.\n\n"
       "`CONDITION='original'` leaves the outcome of the mixed-model objective unstated; `'pinned'` names `antibiotic_duration` "
       "as the primary outcome for objectives 2 to 4."),
    code("BACKEND, MODEL = 'gemini', 'gemini-2.5-pro'\nCONDITION, TRIAL = 'original', 0   # 'original' | 'pinned'"),
    code("from asp_benchmark.agent import run_trial\n\n"
         "score = run_trial(BACKEND, MODEL, CONDITION, TRIAL, temperature=0.7, max_attempts=5, exec_timeout=900)\n"
         "for t in score['tests']:\n    print(t['id'], 'PASS' if t['passed'] else 'FAIL', t['msg'])"),
    md("Aggregate across trials with `python evaluation/summarize_runs.py`."),
])
nbf.write(nb, NOTEBOOKS / "agent_notebook.ipynb")
