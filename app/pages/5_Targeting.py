import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import COLOURS, LABELS, PLOT, footer, load, page, pct, pp, usd

page("Targeting: who should get which email?",
     "An average effect can hide very different responses. This page checks whether some customers "
     "should get the women's email, or no email at all, and whether a machine-learning targeting model "
     "beats a simple policy. Every policy is judged on 30% of customers the model never saw.")
d = load()
metric = st.radio("Outcome", ["spend", "visit"], horizontal=True,
                  format_func=lambda m: "Spend per customer" if m == "spend" else "Visit rate")
fmt = (lambda v: usd(v, 2, True)) if metric == "spend" else (lambda v: pp(v, 2))

# ---- segments
st.subheader("1. Response by purchase history (all 64,000 customers)")
seg = d["segments"][d["segments"].metric == metric]
fig = go.Figure()
for a in ["Mens E-Mail", "Womens E-Mail"]:
    s = seg[seg.arm == a]
    fig.add_trace(go.Bar(x=s.group, y=s["diff"], name=LABELS[a], marker_color=COLOURS[a],
                         error_y=dict(type="data", symmetric=False, array=s.hi - s["diff"],
                                      arrayminus=s["diff"] - s.lo)))
fig.update_layout(barmode="group", height=340, yaxis_title="Lift vs no email",
                  yaxis_tickformat="$.2f" if metric == "spend" else ".0%", **PLOT)
st.plotly_chart(fig, width="stretch")
st.markdown("**The women's email only works on customers who already buy women's items**, and does little for "
            "men-only buyers. The men's email works across all three groups, and is at least as good as the "
            "women's email even for women-only buyers.")

# ---- policies
st.subheader("2. Policy comparison on the 30% holdout")
pol = d["uplift"]["policies"][metric]
names = list(pol)
fig = go.Figure(go.Bar(
    y=names, x=[pol[k]["value"] for k in names], orientation="h",
    marker_color=["#8c8c8c", COLOURS["Womens E-Mail"], COLOURS["Mens E-Mail"], "#7f6000", "#548235"],
    error_x=dict(type="data", symmetric=False, array=[pol[k]["hi"] - pol[k]["value"] for k in names],
                 arrayminus=[pol[k]["value"] - pol[k]["lo"] for k in names])))
fig.update_layout(height=320, xaxis_title=("Spend per customer ($)" if metric == "spend" else "Visit rate"),
                  xaxis_tickformat="$.2f" if metric == "spend" else ".0%", **PLOT)
st.plotly_chart(fig, width="stretch")
st.dataframe(pd.DataFrame([{
    "Policy": k, "Value per customer": usd(v["value"]) if metric == "spend" else pct(v["value"], 2),
    "vs everyone gets men's email": fmt(v["vs_mens"]),
    "90% CI": f"{fmt(v['vs_mens_lo'])} to {fmt(v['vs_mens_hi'])}"} for k, v in pol.items()]),
    hide_index=True, width="stretch")
st.caption("Values estimated by inverse propensity weighting, valid because customers were randomised "
           "1/3 to each arm. Paired bootstrap (1,000 resamples) for the differences.")
st.info("**Verdict: neither the simple rule nor the ML model beats sending everyone the men's email**: both "
        "intervals include zero. A model that adds complexity without measurable gain should not be "
        "deployed.")

# ---- calibration
st.subheader("3. Can the model rank customers at all?")
b = d["buckets"][d["buckets"].metric == metric]
c1, c2 = st.columns(2)
for col, a in zip((c1, c2), ["Mens E-Mail", "Womens E-Mail"]):
    s = b[b.arm == a]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=s.bucket, y=s.observed, marker_color=COLOURS[a], name="Observed uplift",
                         error_y=dict(type="data", symmetric=False, array=s.hi - s.observed,
                                      arrayminus=s.observed - s.lo)))
    fig.add_trace(go.Scatter(x=s.bucket, y=s.predicted, mode="lines+markers", name="Predicted uplift",
                             line=dict(color="black", dash="dot")))
    fig.update_layout(height=320, title=LABELS[a], xaxis_title="Predicted-uplift quintile (1 = lowest)",
                      yaxis_tickformat="$.2f" if metric == "spend" else ".0%", **PLOT)
    col.plotly_chart(fig, width="stretch")
st.markdown("If the model ranks well, observed uplift rises from quintile 1 to 5. For visits it does for the "
            "**women's** email (it learned who buys women's items), but not for the men's email, whose effect "
            "is broad. **Spend is too sparse** (under 1% of customers buy in two weeks) for reliable "
            "customer-level uplift.")

# ---- budget curve
st.subheader("4. If you could only email part of the list")
cv = d["uplift"]["curves"][metric]
shares = np.array(d["uplift"]["shares"])
fig = go.Figure()
fig.add_trace(go.Scatter(x=shares, y=cv["hi"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=shares, y=cv["lo"], mode="lines", line=dict(width=0), fill="tonexty",
                         fillcolor="rgba(31,78,121,0.15)", name="90% interval", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=shares, y=cv["model"], name="Email the customers the model ranks highest",
                         line=dict(color="#1f4e79", width=3)))
fig.add_trace(go.Scatter(x=shares, y=cv["random"], name="Email a random selection",
                         line=dict(color="grey", dash="dash")))
fig.update_layout(height=360, xaxis_title="Share of list emailed (men's email)", xaxis_tickformat=".0%",
                  yaxis_title="Value per customer", yaxis_tickformat="$.2f" if metric == "spend" else ".1%", **PLOT)
st.plotly_chart(fig, width="stretch")
st.caption("Where the blue line sits above the grey one, the model picks better-than-random customers. "
           "This matters only when sends are costly or capped; at near-zero email cost, emailing everyone wins.")
footer()
