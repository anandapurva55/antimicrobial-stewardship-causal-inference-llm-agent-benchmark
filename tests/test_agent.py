"""Generated code runs in a separate process and errors are reported, not raised."""
from asp_benchmark.agent import extract_code, run_generated_code
from asp_benchmark.common import build_prompt
from asp_benchmark.paths import CSV

import pandas as pd


def test_extract_code_removes_fences():
    assert extract_code("```python\nx = 1\n```") == "x = 1"
    assert extract_code("y = 2") == "y = 2"


def test_working_code_returns_no_error_and_writes_results(tmp_path):
    err = run_generated_code("cox_hr_asp = 1.5\n", str(tmp_path), timeout=120)
    assert err == ""
    assert '"cox_hr_asp": 1.5' in (tmp_path / "results.json").read_text()


def test_failing_code_returns_the_error_text(tmp_path):
    err = run_generated_code("raise ValueError('boom')\n", str(tmp_path), timeout=120)
    assert "ValueError: boom" in err


def test_timeout_is_reported(tmp_path):
    err = run_generated_code("import time\ntime.sleep(30)\n", str(tmp_path), timeout=2)
    assert err.startswith("TimeoutError")


def test_pinned_prompt_names_the_outcome_and_original_does_not():
    df = pd.read_csv(CSV)
    assert "PRIMARY OUTCOME: antibiotic_duration" in build_prompt(df, "pinned")
    assert "PRIMARY OUTCOME" not in build_prompt(df, "original")


def test_generated_code_does_not_receive_the_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-for-test")
    err = run_generated_code("import os\nraise SystemExit(os.environ.get('GEMINI_API_KEY', 'absent'))\n", str(tmp_path), timeout=120)
    assert "absent" in err and "secret-for-test" not in err


def test_failed_client_creation_leaves_no_trial_directory(tmp_path, monkeypatch):
    import pytest

    import asp_benchmark.agent as agent

    class Unavailable:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("402 credits depleted")

    monkeypatch.setattr(agent, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(agent, "Gemini", Unavailable)
    with pytest.raises(RuntimeError):
        agent.run_trial("gemini", "some-model", "original", 0, 0.7, 1, 10)
    assert list(tmp_path.iterdir()) == []
