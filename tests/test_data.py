"""The data generator is deterministic and the committed dataset matches it."""
import json

import pandas as pd
import pytest

from asp_benchmark.data_generation import generate
from asp_benchmark.paths import CSV, TRUTH


def test_generator_is_deterministic():
    a, truth_a = generate(seed=42)
    b, truth_b = generate(seed=42)
    pd.testing.assert_frame_equal(a, b)
    assert truth_a == truth_b


def test_committed_dataset_matches_generator():
    data, truth = generate(seed=42)
    committed = pd.read_csv(CSV)
    assert committed.shape == data.shape == (11830, 22)
    assert json.load(open(TRUTH))["true_att_treated_days"] == pytest.approx(truth["true_att_treated_days"])


def test_design_properties():
    df = pd.read_csv(CSV)
    assert df.hospital_id.nunique() == 15 and df.year.nunique() == 6
    assert (df.asp_active == (df.year >= df.asp_adoption_year).astype(int)).all()
    assert 0.25 < 1 - df.event_observed.mean() < 0.35  # about 29.5% censored
