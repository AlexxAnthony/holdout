# Holdout - A/B Testing and Experimentation Analysis

A full experiment analysis of a real randomised email test on 64,000 customers, covering the steps
a modern experimentation team runs: health checks, effect estimation, multiple-testing correction,
Bayesian decision metrics, variance reduction (CUPED/CUPAC), sequential testing for safe peeking,
uplift targeting, and test planning. All of it is shown in a Streamlit dashboard that reads like a decision memo.

**Live dashboard:** _add Streamlit Cloud link_ | **Code:** https://github.com/AlexxAnthony/holdout

## The question

A retailer emailed recent customers at random: a third got a men's merchandise email, a third a women's
email, and a third nothing. **Which email should it send, to whom, and how sure can we be?**

Data: Kevin Hillstrom's MineThatData E-Mail Analytics challenge (2008), a real randomised experiment
with visits, conversions and spend tracked for two weeks.

## Decision

**Send the men's email to the whole list. Do not deploy a targeting model.**

| Finding | Evidence |
|---|---|
| The men's email more than doubles spend per customer | $0.65 to $1.42 (+118%, 95% CI +54% to +181%), Holm-adjusted p < 0.001 |
| It beats the women's email head to head | +$0.35 per customer, p = 0.03; also higher visits and conversion |
| The gain comes from more buyers, not bigger baskets | Average order value unchanged ($113.53 vs $114.00, p = 0.97) |
| The women's email only works on women's-item buyers | Visit lift of about +1 point for men-only buyers vs +7 points for the men's email |
| **Targeting doesn't beat the simple answer** | On a 30% holdout, an ML model choosing each customer's email earned -$0.05 per customer vs "everyone gets the men's email" (90% CI -$0.23 to +$0.12) |

## What the project covers

| Step | Method | Result here |
|---|---|---|
| Sample ratio mismatch | Chi-square test on arm sizes | p = 0.90, healthy |
| Covariate balance | Standardised mean differences | max \|SMD\| = 0.014 |
| Effect estimates | Welch tests, **delta-method** CIs for relative lift | See decision table |
| Ratio metrics | Delta-method variance for spend per visitor and order value | Neither moved |
| Multiple testing | **Holm-Bonferroni** across 6 primary comparisons | All remain significant |
| Bayesian read | Beta-Binomial, P(better), **expected loss** | Agrees with the frequentist result |
| Variance reduction | **CUPED** (past spend) and **CUPAC** (cross-fitted ML prediction) | Only 2.6% variance removed. Honest result: the pre-period covariate is weak |
| Peeking | A/A simulation, 5,000 tests × 20 looks | Peeking inflates false positives from 4% to 24% |
| Sequential testing | **mSPRT** always-valid p-values | False-positive rate stays under 5% (0.04%), but needs about 5× more data to call the spend result |
| Heterogeneous effects | T-learner (gradient boosting), uplift by quintile | Ranks women's-email responders; men's-email effect is broad |
| Policy evaluation | Inverse propensity weighting with paired bootstrap on a 30% holdout | No policy beats sending everyone the men's email |
| Test planning | Sample size / MDE with multi-arm and CUPED adjustments | Interactive planner |

## Honest caveats

- **Two weeks of data.** It can't show whether sales were new or pulled forward, or whether unsubscribes rose.
  No guardrail metrics (unsubscribes, complaints) were recorded.
- **No timestamps.** The sequential replay assumes customers arrived in random order.
- **2008 data.** The methods transfer to current experiments; the effect sizes are specific to this retailer.
- **CUPED barely helped**, because the only pre-period covariate (12-month spend) barely predicts two-week
  behaviour. The method is verified in unit tests where it removes the expected variance.

## Project structure

```
src/holdout/
  stats.py      SRM, SMD, delta method, ratio metrics, Holm, CUPED, Bayesian, mSPRT, power
  uplift.py     T-learner, cross-fitted CUPAC covariate, IPW policy evaluation
  data.py       loading and validation of the Hillstrom file
scripts/
  analyse.py    full analysis -> artifacts/
app/            Streamlit dashboard (decision memo + 6 pages)
tests/          9 tests, including simulation checks of CI coverage, CUPED unbiasedness
                and mSPRT error control
```

## Run it

```bash
pip install -r requirements.txt
streamlit run app/Home.py

# to rerun the analysis (about 10 seconds)
pip install -r requirements-analysis.txt
python scripts/analyse.py
pytest tests
```
