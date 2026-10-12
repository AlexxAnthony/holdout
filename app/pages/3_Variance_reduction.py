import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import LABELS, PLOT, footer, load, page, pct

page("Variance reduction: CUPED and CUPAC",
     "Most of the noise in an A/B test comes from customers simply being different from each other. "
     "**CUPED** (Microsoft, 2013) removes the part of each customer's outcome that was predictable *before* "
     "the test from a pre-period metric. **CUPAC** (DoorDash, 2020) uses a machine-learning prediction "
     "instead. Less noise means narrower confidence intervals, or the same answer with less traffic.")
d = load()
cup = d["cuped"]

st.subheader("What it achieved here")
rows = []
for x in cup.itertuples():
    rows.append({"Metric": "Visit rate" if x.metric == "visit" else "Spend per customer",
                 "Email": LABELS[x.arm], "Method": x.method,
                 "Variance removed": pct(x.variance_reduction, 1),
                 "Traffic saved": pct(1 - 1 / x.equivalent_traffic, 1),
                 "CI width before → after": f"{(x.raw_hi - x.raw_lo):.4f} → {(x.adj_hi - x.adj_lo):.4f}"})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

best = cup.variance_reduction.max()
st.warning(f"""
**Honest result: very little.** The best adjustment removed only {pct(best)} of the variance. CUPED works
when the pre-period metric strongly predicts the outcome, and here it doesn't. Total spend over the past year
says little about whether someone visits in a particular two-week window, and even an ML model using every
pre-experiment feature does only slightly better.

The lesson for a real programme: **the covariate matters more than the method.** The right input would be
the *same metric in the weeks before the test* (pre-period visits or pre-period spend), which this dataset
doesn't include. The method itself is verified in the project's unit tests, where it removes exactly the
expected share of variance and stays unbiased.
""")

st.subheader("What it would be worth with a better covariate")
st.markdown("Variance falls by roughly ρ², where ρ is the correlation between the covariate and the "
            "outcome. Move the slider to see what a stronger pre-period metric would save.")
rho = st.slider("Correlation between pre-period covariate and outcome (ρ)", 0.0, 0.9, 0.5, 0.05)
grid = np.linspace(0, 0.9, 50)
fig = go.Figure()
fig.add_trace(go.Scatter(x=grid, y=grid**2, name="Share of traffic saved", line=dict(color="#1f4e79", width=3)))
fig.add_vline(x=rho, line_dash="dot")
achieved = float(np.sqrt(max(best, 0)))
fig.add_trace(go.Scatter(x=[achieved], y=[best], mode="markers+text", text=["this dataset"],
                         textposition="top right", marker=dict(size=12, color="#c55a11"), name="Achieved here"))
fig.update_layout(height=340, xaxis_title="ρ", yaxis_title="Traffic saved", yaxis_tickformat=".0%", **PLOT)
st.plotly_chart(fig, width="stretch")
st.info(f"At ρ = {rho:.2f}, a test that needs 4 weeks without CUPED would need about "
        f"**{4 * (1 - rho**2):.1f} weeks** with it ({pct(rho**2, 0)} less traffic).")

with st.expander("How it works"):
    st.markdown(r"""
For each customer, with outcome $y$ and a pre-experiment covariate $x$:

$$y_{adj} = y - \theta\,(x - \bar{x}), \qquad \theta = \frac{\mathrm{Cov}(y, x)}{\mathrm{Var}(x)}$$

Because $x$ was measured before randomisation, it is independent of treatment. The adjustment changes
each customer's outcome but not the expected difference between arms, so the estimate stays unbiased while
its variance falls by a factor of $1 - \rho^2$.

For CUPAC, $x$ is an out-of-fold prediction of $y$ from pre-treatment features (recency, purchase history,
channel, area). Cross-fitting means no customer's prediction comes from a model that saw their own outcome.
""")
footer()
