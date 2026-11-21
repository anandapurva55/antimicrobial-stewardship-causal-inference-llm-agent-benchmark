"""Bootstrap intervals shared by the summary script and the figures."""
import numpy as np


def bootstrap_ci(values, resamples=10_000, seed=0):
    """95% percentile bootstrap interval of the mean."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, float)
    means = rng.choice(v, (resamples, len(v))).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bootstrap_diff_ci(a, b, resamples=10_000, seed=0):
    """95% percentile bootstrap interval of mean(b) minus mean(a)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    diffs = rng.choice(b, (resamples, len(b))).mean(axis=1) - rng.choice(a, (resamples, len(a))).mean(axis=1)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
