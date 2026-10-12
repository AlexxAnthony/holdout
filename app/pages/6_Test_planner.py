import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import PLOT, footer, load, page, pct
from holdout.stats import mde_for_n, sample_size_means

page("Test planner",
     "Plan the next test before launching it: how many customers it needs, how long it will run, and the "
     "smallest effect it can reliably detect. Under-powered tests are the most common way experimentation "
     "programmes waste time. Defaults come from this experiment's control group.")
d = load()
ctrl = d["meta"]["control"]

c1, c2 = st.columns(2)
with c1:
    metric = st.selectbox("Primary metric", ["visit", "conversion", "spend"],
                          format_func={"visit": "Visit rate", "conversion": "Conversion rate",
                                       "spend": "Spend per customer"}.get)
    base, sd = ctrl[metric]["mean"], ctrl[metric]["sd"]
    rel_mde = st.slider("Smallest relative lift worth detecting", 0.02, 0.50, 0.10, 0.01, format="%.2f")
    arms = st.number_input("Number of arms (including control)", 2, 6, 2)
with c2:
    power = st.slider("Power", 0.70, 0.95, 0.80, 0.05)
    alpha = st.select_slider("Significance level", [0.01, 0.025, 0.05, 0.10], 0.05)
    traffic = st.number_input("Eligible customers per day", 1_000, 2_000_000, 20_000, 1_000)
    vr = st.slider("Variance reduction from CUPED", 0.0, 0.6, 0.0, 0.05,
                   help="0 if you have no good pre-period covariate. 0.3-0.5 is common with a same-metric pre-period.")

alpha_adj = alpha / (arms - 1)  # Bonferroni across treatment-vs-control comparisons
n = sample_size_means(sd, base * rel_mde, alpha_adj, power, vr)
total = n * arms
days = int(np.ceil(total / traffic))
weeks = int(np.ceil(days / 7))

m1, m2, m3 = st.columns(3)
m1.metric("Customers per arm", f"{n:,}")
m2.metric("Total customers", f"{total:,}")
m3.metric("Run time", f"{days} days", help="Round up to whole weeks so every weekday is covered equally.")
st.markdown((f"Baseline {('$%.2f' % base) if metric == 'spend' else pct(base, 2)}; detecting a "
            f"{pct(rel_mde, 0)} lift means telling it apart from noise at a standard deviation of "
            f"{('$%.2f' % sd) if metric == 'spend' else pct(sd, 1)} per customer. "
            f"**Recommend running {weeks} full week{'s' if weeks > 1 else ''}** to cover weekly cycles."
            + (f" With {arms} arms, alpha is split across {arms - 1} comparisons." if arms > 2 else "")
            ).replace("$", "\\$"))
if metric == "spend":
    st.warning("Spend is very noisy (most customers spend nothing), so revenue tests need far more traffic than "
               "visit or conversion tests. A common approach is a conversion primary metric with revenue as a "
               "secondary metric.")

ns = np.logspace(3, 6.5, 80)
mdes = [mde_for_n(sd, int(x), alpha_adj, power, vr) / base for x in ns]
fig = go.Figure(go.Scatter(x=ns, y=mdes, line=dict(color="#1f4e79", width=3), name="Detectable lift"))
fig.add_vline(x=n, line_dash="dot", annotation_text="your plan")
fig.update_layout(height=360, xaxis_type="log", xaxis_title="Customers per arm",
                  yaxis_title="Minimum detectable relative lift", yaxis_tickformat=".0%",
                  yaxis_range=[0, min(1.0, max(mdes))], **PLOT)
st.plotly_chart(fig, width="stretch")

with st.expander("Formula"):
    st.markdown(r"""
$$n_{\text{per arm}} = \frac{2\,\sigma^2 (1 - VR)\,(z_{1-\alpha/2} + z_{1-\beta})^2}{\delta^2}$$

where $\sigma$ is the metric's standard deviation, $\delta$ the absolute effect to detect, and $VR$ the
share of variance removed by CUPED.
""")
footer()
