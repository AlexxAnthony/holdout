import numpy as np
import streamlit as st

from common import LABELS, footer, load, page, pct, pp, usd

page("Holdout - email experiment decision memo",
     "A retailer emailed 64,000 recent customers at random: a third got a **men's** email, a third a "
     "**women's** email, and a third **nothing**. Which email should it send, to whom, and how sure "
     "can we be? This dashboard is the full analysis, from health checks to targeting.")
d = load()
r = d["results"].set_index(["arm", "metric"])
mens_spend = r.loc[("Mens E-Mail", "spend")]
mens_visit = r.loc[("Mens E-Mail", "visit")]
wom_spend = r.loc[("Womens E-Mail", "spend")]
h2h = r.loc[("Mens vs Womens", "spend")]
aov = r.loc[("Mens E-Mail", "order_value")]
pol = d["uplift"]["policies"]["spend"]
model = pol["ML model picks each customer's email"]
srm_p = d["health"]["srm"]["p"]

st.success("**Recommendation: send the men's email to the whole list. Do not deploy the targeting model.**")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Spend per customer, men's email", usd(mens_spend.mean_t), f"{pct(mens_spend.lift, 0, True)} vs no email")
c2.metric("Visit rate, men's email", pct(mens_visit.mean_t), f"{pp(mens_visit["diff"])} vs no email")
c3.metric("Extra spend per 100k customers emailed", usd(mens_spend["diff"] * 100_000, 0),
          help=f"Two-week window. 95% CI {usd(mens_spend.diff_lo * 100_000, 0)} to {usd(mens_spend.diff_hi * 100_000, 0)}")
c4.metric("Experiment health", "Passed", help=f"Sample ratio p = {srm_p:.2f}; all covariates balanced")

st.subheader("Why")
st.markdown((f"""
1. **Both emails work, and the men's email works better.** It lifted spend per customer by
   {pct(mens_spend.lift, 0)} (95% CI {pct(mens_spend.lift_lo, 0)} to {pct(mens_spend.lift_hi, 0)}),
   against {pct(wom_spend.lift, 0)} for the women's email. Head to head, the men's email earned
   {usd(h2h['diff'], 2)} more per customer (Holm-adjusted p = {h2h.p_holm:.3f}).
2. **The gain comes from more buyers, not bigger baskets.** Average order value barely moved
   ({usd(aov.mean_t)} vs {usd(aov.mean_c)}, p = {aov.p:.2f}). The email brings people in; it
   doesn't change what they buy once there.
3. **Targeting doesn't beat the simple answer.** The women's email only works on customers who already
   buy women's items, but even for them it is no better than the men's email. A machine-learning model
   choosing an email per customer earned {usd(model['vs_mens'], 2, True)} per customer versus sending
   everyone the men's email (90% CI {usd(model['vs_mens_lo'], 2, True)} to {usd(model['vs_mens_hi'], 2, True)}),
   on customers it never trained on. Extra complexity, no measurable gain.
4. **The result is trustworthy.** Group sizes match the planned 1/3 split, every customer characteristic is
   balanced across arms, and the conclusions survive correction for testing several metrics at once.
""").replace("$", "\\$"))

st.subheader("What I would test next")
st.markdown("""
- **A men's-email variant against the current men's email**, now that it is the champion. With this
  traffic, the *Test planner* page shows how big a lift such a test could detect.
- **Longer-term effects:** the data covers two weeks only. A holdout group kept out of email for a quarter
  would show whether these sales are new or just pulled forward, and whether unsubscribes rise.
- **Guardrail metrics:** unsubscribes and spam complaints were not recorded here; a production test
  should track them alongside revenue.
""")

with st.expander("What is on each page"):
    st.markdown("""
| Page | Question it answers |
|---|---|
| Experiment health | Can we trust the randomisation at all? (sample ratio mismatch, covariate balance) |
| Results | How big is each effect, with confidence intervals, multiple-testing correction and a Bayesian view? |
| Variance reduction | Could CUPED / CUPAC have reached the same answer with less traffic? |
| Peeking and sequential testing | What goes wrong if you check results daily, and how to do it safely |
| Targeting | Who responds to which email, and is a model worth deploying? |
| Test planner | How many customers and days the next test needs |
""")
footer()
