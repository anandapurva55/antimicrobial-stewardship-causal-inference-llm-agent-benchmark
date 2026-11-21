"""Repository paths. Every script uses these, so nothing depends on the current working directory.

To run the whole benchmark on a freshly generated dataset without touching the committed files, set
ASP_DATA_DIR and ASP_RESULTS_DIR to other folders before running any script.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
DATA_DIR = Path(os.environ.get("ASP_DATA_DIR", ROOT / "data")).resolve()
RESULTS_DIR = Path(os.environ.get("ASP_RESULTS_DIR", ROOT / "results")).resolve()
RUNS_DIR = RESULTS_DIR / "runs"
DOCS_DIR = ROOT / "docs"

CSV = DATA_DIR / "asp_study_data.csv"
TRUTH = DATA_DIR / "ground_truth.json"
GOLDEN_RESULTS = RESULTS_DIR / "golden_results.json"
