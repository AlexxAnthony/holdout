import plotly.graph_objects as go
import streamlit as st

from common import COLOURS, LABELS, PLOT, footer, load, page, pct

page("Experiment health",
     "Before reading any result, check the experiment itself. Broken randomisation or logging produces "
     "confident-looking numbers that are simply wrong. Teams at Microsoft and LinkedIn report that a "
     "noticeable share of their experiments fail these checks.")
d = load()
h = d["health"]

st.subheader("1. Sample ratio mismatch (SRM)")
c1, c2 = st.columns([1, 1.4])
with c1:
    srm = h["srm"]
    ok = srm["p"] > 0.001
    st.metric("Chi-square test p-value", f"{srm['p']:.3f}", "Pass" if ok else "FAIL - stop here",
              delta_color="normal" if ok else "inverse")
    st.markdown("The plan was an even three-way split. If the observed counts were far from that "
                "(p < 0.001 by convention), something dropped or duplicated users, and the test should be "
                "investigated rather than analysed.")
with c2:
    fig = go.Figure(go.Bar(x=[LABELS[a] for a in h["arms"]], y=h["counts"],
                           marker_color=[COLOURS[a] for a in h["arms"]],
                           text=[f"{n:,} ({pct(s)})" for n, s in zip(h["counts"], srm["observed_share"])],
                           textposition="outside"))
    fig.add_hline(y=sum(h["counts"]) / 3, line_dash="dash", annotation_text="expected")
    fig.update_layout(height=300, yaxis_title="Customers", yaxis_range=[0, max(h["counts"]) * 1.2], **PLOT)
    st.plotly_chart(fig, width="stretch")

st.subheader("2. Covariate balance")
st.markdown("Every pre-experiment characteristic should look the same in each arm. The standardised mean "
            "difference (SMD) measures the gap in standard-deviation units; |SMD| < 0.1 is the usual bar.")
b = d["balance"]
fig = go.Figure()
for a in b.arm.unique():
    s = b[b.arm == a]
    fig.add_trace(go.Scatter(x=s.smd, y=s.covariate, mode="markers", name=f"{LABELS[a]} vs no email",
                             marker=dict(size=10, color=COLOURS[a])))
for x in (-0.1, 0.1):
    fig.add_vline(x=x, line_dash="dash", line_color="grey")
fig.update_layout(height=420, xaxis_title="Standardised mean difference", xaxis_range=[-0.15, 0.15], **PLOT)
st.plotly_chart(fig, width="stretch")
st.caption(f"Largest |SMD| across all covariates: {h['max_abs_smd']:.3f}. The randomisation worked.")
footer()
