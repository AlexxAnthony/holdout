"""Run the full experiment analysis and save artifacts for the dashboard.

    python scripts/analyse.py

Steps: health checks -> effects with multiple-testing correction -> Bayesian
read -> CUPED/CUPAC variance reduction -> peeking simulation and sequential
replay -> uplift targeting evaluated on a 30% holdout.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from holdout import stats as S  # noqa: E402
from holdout.data import (ARMS, CONTROL, METRICS, arm, balance_frame, feature_matrix,  # noqa: E402
                          load, rng_order)
from holdout.uplift import (cross_fit_prediction, ipw_value, policy_curve, t_learner,
                             uplift_by_bucket)  # noqa: E402

ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)
TREATMENTS = [a for a in ARMS if a != CONTROL]


def save(name, obj):
    (ART / name).write_text(json.dumps(obj, indent=2, default=float))


def main():
    t0 = time.time()
    df = load()
    print(f"loaded {len(df):,} rows")
    c = arm(df, CONTROL)

    # ---------------------------------------------------------------- 1. health
    counts = [len(arm(df, a)) for a in ARMS]
    srm = S.srm_test(counts)
    bf = balance_frame(df)
    covs = [col for col in bf.columns if col != "segment"]
    bal = []
    for t in TREATMENTS:
        for cov in covs:
            bal.append({"arm": t, "covariate": cov,
                        "smd": S.smd(bf.loc[bf.segment == t, cov].to_numpy(),
                                     bf.loc[bf.segment == CONTROL, cov].to_numpy())})
    bal = pd.DataFrame(bal)
    bal.to_csv(ART / "balance.csv", index=False)
    save("health.json", {"arms": ARMS, "counts": counts, "srm": srm,
                         "max_abs_smd": float(bal.smd.abs().max())})
    print("SRM p", round(srm["p"], 3), "max |SMD|", round(bal.smd.abs().max(), 4))

    # ---------------------------------------------------------------- 2. effects
    rows = []
    for t in TREATMENTS:
        tr = arm(df, t)
        for m in METRICS:
            e = S.compare_means(tr[m], c[m])
            rows.append({"arm": t, "metric": m, "family": "primary", **e.__dict__})
        # ratio metrics: per-visitor spend and average order value
        for name, num, den in [("spend_per_visitor", "spend", "visit"), ("order_value", "spend", "conversion")]:
            e = S.ratio_metric(tr[num], tr[den], c[num], c[den])
            rows.append({"arm": t, "metric": name, "family": "diagnostic", **e.__dict__})
    # head to head: men's vs women's
    for m in METRICS:
        e = S.compare_means(arm(df, "Mens E-Mail")[m], arm(df, "Womens E-Mail")[m])
        rows.append({"arm": "Mens vs Womens", "metric": m, "family": "head_to_head", **e.__dict__})
    res = pd.DataFrame(rows)
    prim = res.family == "primary"
    res["p_holm"] = np.nan
    res.loc[prim, "p_holm"] = S.holm(res.loc[prim, "p"])
    res.loc[res.family == "head_to_head", "p_holm"] = S.holm(res.loc[res.family == "head_to_head", "p"])
    res.to_csv(ART / "results.csv", index=False)
    print(res[["arm", "metric", "mean_t", "mean_c", "lift", "p", "p_holm"]].round(4).to_string())

    # ---------------------------------------------------------------- 3. Bayesian
    bayes = {}
    for t in TREATMENTS + ["Mens vs Womens"]:
        a, b = (arm(df, "Mens E-Mail"), arm(df, "Womens E-Mail")) if t == "Mens vs Womens" else (arm(df, t), c)
        bayes[t] = {m: S.bayes_binary(int(a[m].sum()), len(a), int(b[m].sum()), len(b))
                    for m in ["visit", "conversion"]}
    save("bayes.json", bayes)

    # ---------------------------------------------------------------- 4. CUPED / CUPAC
    X = feature_matrix(df)
    cup = []
    for m in ["visit", "spend"]:
        y = df[m].to_numpy(float)
        cupac_x = cross_fit_prediction(X, y, binary=(m == "visit"))
        for t in TREATMENTS:
            mt, mc = (df.segment == t).to_numpy(), (df.segment == CONTROL).to_numpy()
            for method, x in [("CUPED (past spend)", df["history"].to_numpy(float)), ("CUPAC (ML prediction)", cupac_x)]:
                r = S.cuped(y[mt], y[mc], x[mt], x[mc])
                cup.append({"metric": m, "arm": t, "method": method, "theta": r.theta,
                            "variance_reduction": r.variance_reduction,
                            "equivalent_traffic": r.equivalent_traffic,
                            "raw_diff": r.raw.diff, "raw_lo": r.raw.diff_lo, "raw_hi": r.raw.diff_hi, "raw_p": r.raw.p,
                            "adj_diff": r.adjusted.diff, "adj_lo": r.adjusted.diff_lo,
                            "adj_hi": r.adjusted.diff_hi, "adj_p": r.adjusted.p})
    cup = pd.DataFrame(cup)
    cup.to_csv(ART / "cuped.csv", index=False)
    print(cup[["metric", "arm", "method", "variance_reduction", "raw_p", "adj_p"]].round(4).to_string())

    # ---------------------------------------------------------------- 5. peeking
    p_ctrl = float(c["visit"].mean())
    peek = S.simulate_peeking(p=p_ctrl, n_per_arm=len(c), looks=20, sims=5000, seed=3)
    save("peeking.json", peek)
    print("peeking FPR", peek["fixed_horizon_fpr"], peek["peeking_fpr"], peek["msprt_fpr"])

    # sequential replay of the real test, customers in random arrival order
    replay = []
    for t, m in [("Womens E-Mail", "visit"), ("Womens E-Mail", "spend")]:
        sub = df[df.segment.isin([t, CONTROL])]
        sub = sub.iloc[rng_order(len(sub), seed=11)]
        is_t = (sub.segment == t).to_numpy()
        y = sub[m].to_numpy(float)
        n_t, n_c = np.cumsum(is_t), np.cumsum(~is_t)
        s_t, s_c = np.cumsum(y * is_t), np.cumsum(y * ~is_t)
        q_t, q_c = np.cumsum(y**2 * is_t), np.cumsum(y**2 * ~is_t)
        idx = np.unique(np.linspace(500, len(sub) - 1, 120).astype(int))
        mt, mc = s_t[idx] / n_t[idx], s_c[idx] / n_c[idx]
        vt = (q_t[idx] / n_t[idx] - mt**2) / n_t[idx]
        vc = (q_c[idx] / n_c[idx] - mc**2) / n_c[idx]
        diff, var = mt - mc, vt + vc
        from scipy.stats import norm
        p_classic = 2 * norm.sf(np.abs(diff) / np.sqrt(var))
        tau2 = float(var[-1])  # tuned to the noise level at the planned (full) sample size
        p_av = S.always_valid_p(diff, var, tau2)
        for k, i in enumerate(idx):
            replay.append({"arm": t, "metric": m, "customers": int(i + 1), "diff": diff[k],
                           "p_classic": p_classic[k], "p_always_valid": p_av[k]})
    pd.DataFrame(replay).to_csv(ART / "replay.csv", index=False)

    # ---------------------------------------------------------------- 6. who should get which email?
    # 6a. segment-level uplift on the full experiment (readable evidence)
    seg_rows = []
    hist_group = np.select([(df.mens == 1) & (df.womens == 1), df.mens == 1, df.womens == 1],
                           ["Bought both", "Men's only", "Women's only"], "Neither")
    for g in ["Men's only", "Women's only", "Bought both"]:
        sub = df[hist_group == g]
        for t in TREATMENTS:
            for m in ["visit", "spend"]:
                e = S.compare_means(arm(sub, t)[m], arm(sub, CONTROL)[m])
                seg_rows.append({"group": g, "arm": t, "metric": m, "n": int(len(sub)), "diff": e.diff,
                                 "lo": e.diff_lo, "hi": e.diff_hi, "p": e.p})
    pd.DataFrame(seg_rows).to_csv(ART / "segments.csv", index=False)

    # 6b. T-learner trained on 70%, every policy judged on the untouched 30%
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.3, random_state=7,
                                      stratify=df["segment"])
    arm_all = df["segment"].to_numpy()
    a_te = arm_all[idx_te]
    propensity = 1 / 3
    shares = np.linspace(0, 1, 21)
    te_df = df.iloc[idx_te]
    simple_rule = np.where((te_df.womens == 1) & (te_df.mens == 0), "Womens E-Mail", "Mens E-Mail")
    up = {"shares": shares.tolist(), "n_test": int(len(idx_te)), "policies": {}, "curves": {}}
    buckets = []
    rng = np.random.default_rng(5)
    boot_idx = [rng.integers(0, len(idx_te), len(idx_te)) for _ in range(1000)]
    for m in ["visit", "spend"]:
        y = df[m].to_numpy(float)
        y_te = y[idx_te]
        preds = t_learner(X.iloc[idx_tr], arm_all[idx_tr], y[idx_tr], X.iloc[idx_te], ARMS,
                          binary=(m == "visit"))
        upl = np.column_stack([preds[t] - preds[CONTROL] for t in TREATMENTS])
        model_choice = np.array(TREATMENTS)[upl.argmax(1)]
        policies = {
            "Email nobody": np.full(len(idx_te), CONTROL),
            "Everyone gets the women's email": np.full(len(idx_te), "Womens E-Mail"),
            "Everyone gets the men's email": np.full(len(idx_te), "Mens E-Mail"),
            "Simple rule: women's email to women-only buyers": simple_rule,
            "ML model picks each customer's email": model_choice,
        }
        contrib = {k: ipw_value(y_te, a_te, v, propensity) for k, v in policies.items()}
        ref = contrib["Everyone gets the men's email"]
        out = {}
        for k, v in contrib.items():
            boots = np.array([v[b].mean() for b in boot_idx])
            dboots = np.array([(v[b] - ref[b]).mean() for b in boot_idx])
            out[k] = {"value": float(v.mean()), "lo": float(np.percentile(boots, 5)),
                      "hi": float(np.percentile(boots, 95)), "vs_mens": float((v - ref).mean()),
                      "vs_mens_lo": float(np.percentile(dboots, 5)), "vs_mens_hi": float(np.percentile(dboots, 95))}
        up["policies"][m] = out
        # 6c. if you can only email k% of the list: rank by predicted men's-email uplift
        mens_up = upl[:, TREATMENTS.index("Mens E-Mail")]
        mens_all = np.full(len(idx_te), "Mens E-Mail")
        curve = policy_curve(y_te, a_te, mens_all, mens_up, CONTROL, propensity, shares)
        rand = policy_curve(y_te, a_te, mens_all, rng.random(len(idx_te)), CONTROL, propensity, shares)
        up["curves"][m] = {"model": curve.value.tolist(), "lo": curve.lo.tolist(), "hi": curve.hi.tolist(),
                           "random": rand.value.tolist()}
        for k, t in enumerate(TREATMENTS):
            b = uplift_by_bucket(y_te, a_te, t, CONTROL, upl[:, k])
            b["metric"], b["arm"] = m, t
            buckets.append(b)
        print(m, {k: (round(v["value"], 4), round(v["vs_mens"], 4), round(v["vs_mens_lo"], 4),
                      round(v["vs_mens_hi"], 4)) for k, v in out.items()})
    save("uplift.json", up)
    pd.concat(buckets).to_csv(ART / "uplift_buckets.csv", index=False)

    # ---------------------------------------------------------------- planner inputs
    save("meta.json", {
        "n_customers": int(len(df)), "arm_counts": dict(zip(ARMS, counts)),
        "control": {m: {"mean": float(c[m].mean()), "sd": float(c[m].std())} for m in METRICS},
        "cupac_variance_reduction": {m: float(cup[(cup.metric == m) & cup.method.str.startswith("CUPAC")]
                                              .variance_reduction.mean()) for m in ["visit", "spend"]},
        "runtime_s": round(time.time() - t0, 1),
    })
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
