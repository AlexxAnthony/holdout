"""Who should get which email? Uplift modelling and off-policy evaluation.

An average treatment effect hides who actually responds. A T-learner fits
one outcome model per arm; the predicted uplift for a customer is the
difference between the treated-arm and control-arm predictions.

Models are trained on 70% of customers and judged only on the other 30%.
Because the experiment was randomised (1/3 per arm), the value of any
targeting policy can be estimated on that holdout without bias using
inverse propensity weighting:
    value(policy) = mean( y * 1[arm == policy(x)] / P(arm) )
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold


def _model(binary: bool, seed: int):
    kw = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=200,
              l2_regularization=1.0, random_state=seed)
    return HistGradientBoostingClassifier(**kw) if binary else HistGradientBoostingRegressor(**kw)


def _predict(m, X, binary):
    return m.predict_proba(X)[:, 1] if binary else m.predict(X)


def t_learner(X_tr, arm_tr, y_tr, X_te, arms, binary=True, seed=0) -> dict[str, np.ndarray]:
    """Predicted outcome under each arm for every test customer."""
    out = {}
    for a in arms:
        mask = arm_tr == a
        m = _model(binary, seed).fit(X_tr[mask], y_tr[mask])
        out[a] = _predict(m, X_te, binary)
    return out


def cross_fit_prediction(X, y, binary: bool, folds: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold predictions of an outcome from pre-treatment features.
    Used as the CUPAC covariate: each customer's prediction comes from a model
    that never saw that customer, and the features ignore treatment, so the
    adjustment cannot leak the treatment effect."""
    pred = np.zeros(len(y))
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X):
        m = _model(binary, seed).fit(X.iloc[tr], y[tr])
        pred[te] = _predict(m, X.iloc[te], binary)
    return pred


@dataclass
class PolicyCurve:
    share_targeted: np.ndarray
    value: np.ndarray        # IPW estimate of mean outcome per customer
    lo: np.ndarray
    hi: np.ndarray


def ipw_value(y, arm, chosen, propensity) -> np.ndarray:
    """Per-customer IPW contributions for a policy that picks `chosen`."""
    return y * (arm == chosen) / propensity


def policy_curve(y, arm, best_arm, best_uplift, control, propensity, shares, n_boot=300, seed=0) -> PolicyCurve:
    """Email the top k% of customers (by predicted uplift) with their best
    email, nobody else. Returns IPW value per customer with bootstrap CIs."""
    order = np.argsort(-best_uplift)
    rank = np.empty(len(order)); rank[order] = np.arange(len(order)) / len(order)
    contribs = []
    for s in shares:
        chosen = np.where(rank < s, best_arm, control)
        contribs.append(ipw_value(y, arm, chosen, propensity))
    contribs = np.array(contribs)  # (S, N)
    rng = np.random.default_rng(seed)
    boots = np.array([contribs[:, rng.integers(0, contribs.shape[1], contribs.shape[1])].mean(1)
                      for _ in range(n_boot)])
    return PolicyCurve(np.asarray(shares), contribs.mean(1),
                       np.percentile(boots, 5, 0), np.percentile(boots, 95, 0))


def uplift_by_bucket(y, arm, treat, control, score, buckets=5):
    """Observed uplift (treat minus control mean) within quantiles of the
    predicted uplift score - the honest test of whether the model ranks
    customers better than chance."""
    q = pd.qcut(score, buckets, labels=False, duplicates="drop")
    rows = []
    for b in range(int(q.max()) + 1):
        m = q == b
        yt, yc = y[m & (arm == treat)], y[m & (arm == control)]
        diff = yt.mean() - yc.mean()
        se = np.sqrt(yt.var(ddof=1) / len(yt) + yc.var(ddof=1) / len(yc))
        rows.append({"bucket": b + 1, "predicted": float(score[m].mean()), "observed": float(diff),
                     "lo": float(diff - 1.645 * se), "hi": float(diff + 1.645 * se), "n": int(m.sum())})
    return pd.DataFrame(rows)
