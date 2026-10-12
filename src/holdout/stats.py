"""Experiment statistics, written out explicitly so every number can be explained.

Covers the checks and estimators a modern experimentation platform runs:
health checks (SRM, covariate balance), effect estimates with delta-method
confidence intervals, multiple-testing correction, variance reduction
(CUPED), Bayesian decision metrics, always-valid sequential tests (mSPRT)
and power / sample-size planning.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

Z = stats.norm.ppf


# ----------------------------------------------------------- health checks
def srm_test(counts, expected_share=None) -> dict:
    """Sample ratio mismatch: are arm sizes consistent with the planned split?
    A tiny p-value (convention: < 0.001) means randomisation or logging broke
    and no result from the test should be trusted."""
    counts = np.asarray(counts, float)
    share = np.full(len(counts), 1 / len(counts)) if expected_share is None else np.asarray(expected_share, float)
    expected = counts.sum() * share / share.sum()
    chi2, p = stats.chisquare(counts, expected)
    return {"chi2": float(chi2), "p": float(p), "observed_share": (counts / counts.sum()).tolist()}


def smd(x_t: np.ndarray, x_c: np.ndarray) -> float:
    """Standardised mean difference; |SMD| < 0.1 is the usual 'balanced' bar."""
    pooled = np.sqrt((np.var(x_t, ddof=1) + np.var(x_c, ddof=1)) / 2)
    return float((np.mean(x_t) - np.mean(x_c)) / pooled) if pooled > 0 else 0.0


# ----------------------------------------------------------- effect estimates
@dataclass
class Effect:
    mean_t: float
    mean_c: float
    diff: float
    diff_lo: float
    diff_hi: float
    lift: float          # relative: mean_t / mean_c - 1
    lift_lo: float
    lift_hi: float
    p: float
    se: float


def compare_means(y_t, y_c, alpha: float = 0.05) -> Effect:
    """Welch difference in means plus a delta-method CI for relative lift.

    Relative lift r = m_t / m_c - 1. Its variance by the delta method is
        Var(r) ~ Var(m_t)/m_c^2 + m_t^2 Var(m_c)/m_c^4
    which is what platforms such as Optimizely, Eppo and Statsig report.
    """
    y_t, y_c = np.asarray(y_t, float), np.asarray(y_c, float)
    mt, mc = y_t.mean(), y_c.mean()
    vt, vc = y_t.var(ddof=1) / len(y_t), y_c.var(ddof=1) / len(y_c)
    se = np.sqrt(vt + vc)
    z = Z(1 - alpha / 2)
    diff = mt - mc
    p = 2 * stats.norm.sf(abs(diff / se)) if se > 0 else 1.0
    lift = mt / mc - 1
    se_lift = np.sqrt(vt / mc**2 + mt**2 * vc / mc**4)
    return Effect(mt, mc, diff, diff - z * se, diff + z * se, lift,
                  lift - z * se_lift, lift + z * se_lift, float(p), float(se))


def ratio_metric(num_t, den_t, num_c, den_c, alpha: float = 0.05) -> Effect:
    """Metrics like revenue per visitor where the denominator is itself random.
    Randomisation is per customer, so a naive per-visit t-test understates
    variance. The delta method gives the correct per-arm variance:
        Var(N/D) ~ (1/n mu_D^2) [Var(N) - 2R Cov(N,D) + R^2 Var(D)]
    """
    def arm(num, den):
        num, den = np.asarray(num, float), np.asarray(den, float)
        n, mn, md = len(num), num.mean(), den.mean()
        r = mn / md
        cov = np.cov(num, den, ddof=1)
        var = (cov[0, 0] - 2 * r * cov[0, 1] + r**2 * cov[1, 1]) / (n * md**2)
        return r, var

    rt, vt = arm(num_t, den_t)
    rc, vc = arm(num_c, den_c)
    se = np.sqrt(vt + vc)
    z = Z(1 - alpha / 2)
    diff = rt - rc
    p = 2 * stats.norm.sf(abs(diff / se))
    se_lift = np.sqrt(vt / rc**2 + rt**2 * vc / rc**4)
    lift = rt / rc - 1
    return Effect(rt, rc, diff, diff - z * se, diff + z * se, lift,
                  lift - z * se_lift, lift + z * se_lift, float(p), float(se))


def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (controls family-wise error)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ----------------------------------------------------------- variance reduction
@dataclass
class CupedResult:
    raw: Effect
    adjusted: Effect
    theta: float
    variance_reduction: float   # 1 - Var(adjusted diff) / Var(raw diff)
    equivalent_traffic: float   # how much more traffic the raw test would need


def cuped(y_t, y_c, x_t, x_c, alpha: float = 0.05) -> CupedResult:
    """CUPED (Deng et al., Microsoft 2013).

    Subtract the part of the outcome predicted by a pre-experiment covariate x:
        y_adj = y - theta * (x - mean(x)),  theta = Cov(y, x) / Var(x)
    x is unaffected by treatment, so the estimate stays unbiased while variance
    falls by roughly corr(y, x)^2. x can be a raw pre-period metric or an ML
    prediction from pre-treatment features (CUPAC, DoorDash 2020).
    """
    y = np.concatenate([y_t, y_c]).astype(float)
    x = np.concatenate([x_t, x_c]).astype(float)
    theta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    mx = x.mean()
    adj_t = np.asarray(y_t, float) - theta * (np.asarray(x_t, float) - mx)
    adj_c = np.asarray(y_c, float) - theta * (np.asarray(x_c, float) - mx)
    raw = compare_means(y_t, y_c, alpha)
    adj = compare_means(adj_t, adj_c, alpha)
    # report lift relative to the raw control mean so the two are comparable
    adj.lift = adj.diff / raw.mean_c
    adj.lift_lo, adj.lift_hi = adj.diff_lo / raw.mean_c, adj.diff_hi / raw.mean_c
    vr = 1 - adj.se**2 / raw.se**2
    return CupedResult(raw, adj, float(theta), float(vr), float(1 / (1 - vr)))


# ----------------------------------------------------------- Bayesian
def bayes_binary(conv_t, n_t, conv_c, n_c, draws: int = 200_000, seed: int = 0) -> dict:
    """Beta-Binomial with a flat prior. Returns the probability treatment beats
    control and the expected loss of shipping each arm: the average amount of
    conversion rate you give up if you pick it and it is actually worse."""
    rng = np.random.default_rng(seed)
    pt = rng.beta(1 + conv_t, 1 + n_t - conv_t, draws)
    pc = rng.beta(1 + conv_c, 1 + n_c - conv_c, draws)
    lift = pt / pc - 1
    return {
        "p_better": float((pt > pc).mean()),
        "loss_ship_t": float(np.maximum(pc - pt, 0).mean()),
        "loss_ship_c": float(np.maximum(pt - pc, 0).mean()),
        "lift_mean": float(lift.mean()),
        "lift_lo": float(np.percentile(lift, 2.5)),
        "lift_hi": float(np.percentile(lift, 97.5)),
    }


# ----------------------------------------------------------- sequential testing
def msprt_lambda(diff, var_diff, tau2):
    """Mixture SPRT statistic for a difference in means (Johari et al. 2017,
    as used by Optimizely). With a N(0, tau2) mixture over the true effect:
        Lambda = sqrt(V / (V + tau2)) * exp(diff^2 tau2 / (2 V (V + tau2)))
    where V is the variance of the observed difference. Stopping the first
    time Lambda >= 1/alpha keeps the false-positive rate <= alpha no matter
    how often you look."""
    v = np.asarray(var_diff, float)
    return np.sqrt(v / (v + tau2)) * np.exp(np.asarray(diff) ** 2 * tau2 / (2 * v * (v + tau2)))


def always_valid_p(diffs, var_diffs, tau2) -> np.ndarray:
    """Running always-valid p-value: min over time of 1 / Lambda (capped at 1)."""
    lam = msprt_lambda(diffs, var_diffs, tau2)
    return np.minimum.accumulate(np.minimum(1.0, 1.0 / np.maximum(lam, 1e-300)), axis=-1)


def simulate_peeking(p: float = 0.15, n_per_arm: int = 20_000, looks: int = 20,
                     sims: int = 4000, alpha: float = 0.05, tau2: float | None = None,
                     seed: int = 0) -> dict:
    """A/A simulation (no true effect). Compare the false-positive rate of
    (a) one look at the end, (b) a classic z-test checked at every look and
    stopped at the first 'significant' result, (c) mSPRT checked at every look."""
    rng = np.random.default_rng(seed)
    checkpoints = np.linspace(n_per_arm / looks, n_per_arm, looks).astype(int)
    batch = np.diff(np.concatenate([[0], checkpoints]))
    ct = np.cumsum(rng.binomial(batch, p, size=(sims, looks)), axis=1)
    cc = np.cumsum(rng.binomial(batch, p, size=(sims, looks)), axis=1)
    n = checkpoints[None, :]
    pt, pc = ct / n, cc / n
    pooled = (ct + cc) / (2 * n)
    var = 2 * pooled * (1 - pooled) / n
    var = np.maximum(var, 1e-12)
    z = (pt - pc) / np.sqrt(var)
    sig = np.abs(z) > Z(1 - alpha / 2)
    if tau2 is None:
        # tune the mixture to the noise level at the planned sample size, which
        # makes the test most sensitive around the end of the planned run
        tau2 = 2 * p * (1 - p) / n_per_arm
    avp = always_valid_p(pt - pc, var, tau2)
    peek_fpr_by_look = sig.cumsum(axis=1).astype(bool).mean(axis=0)
    return {
        "fixed_horizon_fpr": float(sig[:, -1].mean()),
        "peeking_fpr": float(sig.any(axis=1).mean()),
        "msprt_fpr": float((avp[:, -1] <= alpha).mean()),
        "peeking_fpr_by_look": peek_fpr_by_look.tolist(),
        "msprt_fpr_by_look": (avp <= alpha).mean(axis=0).tolist(),
        "looks": looks, "sims": sims, "p": p, "n_per_arm": n_per_arm,
    }


# ----------------------------------------------------------- planning
def sample_size_means(sd: float, mde_abs: float, alpha: float = 0.05, power: float = 0.8,
                      variance_reduction: float = 0.0) -> int:
    """Users per arm to detect an absolute difference `mde_abs` (two-sided)."""
    var = sd**2 * (1 - variance_reduction)
    return int(np.ceil(2 * var * (Z(1 - alpha / 2) + Z(power)) ** 2 / mde_abs**2))


def sample_size_proportion(p: float, rel_mde: float, alpha: float = 0.05, power: float = 0.8,
                           variance_reduction: float = 0.0) -> int:
    return sample_size_means(np.sqrt(p * (1 - p)), p * rel_mde, alpha, power, variance_reduction)


def mde_for_n(sd: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.8,
              variance_reduction: float = 0.0) -> float:
    """Smallest absolute effect detectable with n users per arm."""
    var = sd**2 * (1 - variance_reduction)
    return float((Z(1 - alpha / 2) + Z(power)) * np.sqrt(2 * var / n_per_arm))
