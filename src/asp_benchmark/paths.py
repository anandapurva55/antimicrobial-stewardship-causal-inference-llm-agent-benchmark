"""Repository paths. Every script uses these, so nothing depends on the current working directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
DOCS_DIR = ROOT / "docs"

CSV = DATA_DIR / "asp_study_data.csv"
TRUTH = DATA_DIR / "ground_truth.json"
GOLDEN_RESULTS = RESULTS_DIR / "golden_results.json"
