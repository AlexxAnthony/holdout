"""Tests that check the statistics do what they claim, mostly by simulation."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from holdout import stats as S  # noqa: E402

rng = np.random.default_rng(0)


def test_srm_flags_broken_split_and_passes_good_one():
    assert S.srm_test([50_000, 50_300])["p"] > 0.01
    assert S.srm_test([50_000, 51_500])["p"] < 0.001


def test_compare_means_ci_coverage():
    """95% CIs for the difference should cover the true value ~95% of the time."""
    hits = 0
    for _ in range(400):
        t, c = rng.normal(1.1, 2, 2000), rng.normal(1.0, 2, 2000)
        e = S.compare_means(t, c)
        hits += e.diff_lo <= 0.1 <= e.diff_hi
    assert 0.92 <= hits / 400 <= 0.98


def test_delta_method_lift_matches_bootstrap():
    t, c = rng.exponential(1.1, 5000), rng.exponential(1.0, 5000)
    e = S.compare_means(t, c)
    boot = [rng.choice(t, len(t)).mean() / rng.choice(c, len(c)).mean() - 1 for _ in range(1500)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    assert abs((e.lift_hi - e.lift_lo) - (hi - lo)) / (hi - lo) < 0.1


def test_ratio_metric_variance_matches_simulation():
    """Delta-method SE for a per-visit ratio should match its true sampling SD."""
    def arm(n):
        visits = rng.poisson(2, n)
        revenue = visits * rng.gamma(2, 5, n)
        return revenue, visits
    ests, ses = [], []
    for _ in range(300):
        nt, dt = arm(3000); nc, dc = arm(3000)
        e = S.ratio_metric(nt, dt, nc, dc)
        ests.append(e.diff); ses.append(e.se)
    assert abs(np.mean(ses) / np.std(ests) - 1) < 0.12


def test_holm():
    adj = S.holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_cuped_unbiased_and_reduces_variance():
    diffs_raw, diffs_adj, vrs = [], [], []
    for _ in range(300):
        x = rng.normal(0, 1, 4000)
        treat = rng.random(4000) < 0.5
        y = 2 + 1.5 * x + rng.normal(0, 1, 4000) + 0.2 * treat
        r = S.cuped(y[treat], y[~treat], x[treat], x[~treat])
        diffs_raw.append(r.raw.diff); diffs_adj.append(r.adjusted.diff); vrs.append(r.variance_reduction)
    assert abs(np.mean(diffs_adj) - 0.2) < 0.01           # unbiased
    assert np.std(diffs_adj) < 0.6 * np.std(diffs_raw)     # much tighter
    assert abs(np.mean(vrs) - 1.5**2 / (1.5**2 + 1)) < 0.03  # ~ corr^2


def test_bayes_binary_direction():
    b = S.bayes_binary(600, 5000, 500, 5000)
    assert b["p_better"] > 0.99 and b["loss_ship_t"] < b["loss_ship_c"]


def test_peeking_inflates_and_msprt_controls_error():
    r = S.simulate_peeking(p=0.1, n_per_arm=10_000, looks=20, sims=3000, seed=1)
    assert 0.03 < r["fixed_horizon_fpr"] < 0.07
    assert r["peeking_fpr"] > 0.15
    assert r["msprt_fpr"] <= 0.05


def test_sample_size_and_mde_are_inverse():
    n = S.sample_size_means(sd=10, mde_abs=1)
    assert abs(S.mde_for_n(10, n) - 1) < 0.01
    assert S.sample_size_means(10, 1, variance_reduction=0.5) < 0.51 * n
