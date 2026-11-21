"""The 14 behavioural tests accept the golden solution and reject specific, known mistakes."""
import copy

import pytest

from asp_benchmark.common import load_results
from asp_benchmark.paths import GOLDEN_RESULTS
from asp_benchmark.scoring import run_tests


@pytest.fixture(scope="module")
def golden():
    return load_results(GOLDEN_RESULTS)


def failed_ids(results):
    return {r["id"] for r in run_tests(results) if not r["passed"]}


def test_golden_solution_passes_all_tests(golden):
    rows = run_tests(golden)
    assert len(rows) == 14
    assert failed_ids(golden) == set()


def test_length_of_stay_icc_is_detected(golden):
    wrong = copy.deepcopy(golden)
    wrong["icc_value"] = 0.037  # the ICC of length_of_stay
    rows = {r["id"]: r for r in run_tests(wrong)}
    assert not rows["S12"]["passed"] and "length_of_stay" in rows["S12"]["msg"]


def test_twfe_reported_as_staggered_estimate_is_detected(golden):
    wrong = copy.deepcopy(golden)
    wrong["did_att_estimate"] = golden["twfe_att_estimate"]  # -2.58 instead of about -3.18
    assert "S8" in failed_ids(wrong)


def test_missing_outputs_fail_instead_of_stopping_the_run(golden):
    partial = {k: v for k, v in golden.items() if k not in ("icc_value", "cox_hr_asp")}
    assert {"S1", "S12"} <= failed_ids(partial)
    assert len(run_tests(partial)) == 14  # every test still ran


def test_ph_dictionary_with_only_the_treatment_covariate_is_accepted(golden):
    ok = copy.deepcopy(golden)
    ok["ph_test_pvalues"] = {"asp_active": 0.0001}
    assert "S3" not in failed_ids(ok)
    bad = copy.deepcopy(golden)
    bad["ph_test_pvalues"] = {"age": 0.5}  # no entry for asp_active
    assert "S3" in failed_ids(bad)
