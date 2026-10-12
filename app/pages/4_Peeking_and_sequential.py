import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import PLOT, footer, load, page, pct

page("Peeking and sequential testing",
     "Stakeholders want to check results every day and stop as soon as something looks significant. "
     "With a classic test that quietly multiplies the false-positive rate. Sequential tests such as the "
     "**mixture SPRT** (used by Optimizely) give *always-valid* p-values, so you can look as often as you "
     "like without breaking the 5% error guarantee.")
d = load()
pk = d["peeking"]

st.subheader("1. Simulation: A/A tests where there is no real effect")
c1, c2, c3 = st.columns(3)
c1.metric("Look once at the end", pct(pk["fixed_horizon_fpr"]), help="Should be about 5%")
c2.metric("Check 20 times, stop when significant", pct(pk["peeking_fpr"]),
          f"{pk['peeking_fpr'] / 0.05:.1f}× the promised 5%", delta_color="inverse")
c3.metric("mSPRT, check 20 times", pct(pk["msprt_fpr"], 2), "under 5%, on the cautious side", delta_color="off",
          help="mSPRT guarantees at most 5%. In practice it is conservative; the price is slower detection.")

looks = np.arange(1, pk["looks"] + 1)
fig = go.Figure()
fig.add_trace(go.Scatter(x=looks, y=pk["peeking_fpr_by_look"], name="Classic test with peeking",
                         line=dict(color="#c00000", width=3)))
fig.add_trace(go.Scatter(x=looks, y=pk["msprt_fpr_by_look"], name="mSPRT (always valid)",
                         line=dict(color="#1f4e79", width=3)))
fig.add_hline(y=0.05, line_dash="dash", annotation_text="promised 5%")
fig.update_layout(height=360, xaxis_title="Number of looks so far", yaxis_title="Chance of a false 'winner'",
                  yaxis_tickformat=".0%", **PLOT)
st.plotly_chart(fig, width="stretch")
st.caption(f"{pk['sims']:,} simulated A/A tests at this experiment's real control visit rate "
           f"({pct(pk['p'])}) and size ({pk['n_per_arm']:,} customers per arm).")

st.subheader("2. Replay of the real test as if customers arrived one by one")
rp = d["replay"]
metric = st.radio("Metric (women's email vs no email)", ["spend", "visit"], horizontal=True,
                  format_func=lambda m: "Visit rate" if m == "visit" else "Spend per customer")
s = rp[rp.metric == metric]
fig = go.Figure()
FLOOR = 1e-6
fig.add_trace(go.Scatter(x=s.customers, y=s.p_classic.clip(lower=FLOOR), name="Classic p-value",
                         line=dict(color="#c00000")))
fig.add_trace(go.Scatter(x=s.customers, y=s.p_always_valid.clip(lower=FLOOR), name="Always-valid p-value",
                         line=dict(color="#1f4e79", width=3)))
fig.add_hline(y=0.05, line_dash="dash")
fig.update_layout(height=380, yaxis_type="log", xaxis_title="Customers observed so far",
                  yaxis_title="p-value (log scale)", yaxis_range=[-6.1, 0.05],
                  yaxis_tickvals=[1, 0.05, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                  yaxis_ticktext=["1", "0.05", "0.01", "0.001", "0.0001", "0.00001", "≤0.000001"], **PLOT)
st.plotly_chart(fig, width="stretch")


def first_cross(col):
    hit = s[s[col] <= 0.05]
    return int(hit.customers.iloc[0]) if len(hit) else None


fc, fa = first_cross("p_classic"), first_cross("p_always_valid")
final_p = float(s.p_classic.iloc[-1])
lines = []
if fc:
    lines.append(f"- The classic p-value first dipped below 0.05 after **{fc:,} customers**. It can bounce back "
                 "above, which is exactly the trap: stop at a lucky moment and you report a false win.")
else:
    lines.append("- The classic p-value never went below 0.05.")
if fa:
    lines.append(f"- The always-valid p-value only ever goes down, and crossed 0.05 after **{fa:,} customers**. "
                 "It is slower; that is the price of being allowed to look at any time.")
else:
    lines.append(f"- The always-valid p-value **never crossed 0.05 within this test**, even though the final classic "
                 f"p-value was {final_p:.4f}. That is the price of being allowed to look at any time: a sequential "
                 "test needs more data to declare a winner, so plan for it when sizing the test.")
st.markdown("\n".join(lines))
st.caption("The dataset has no timestamps, so arrival order is randomised. mSPRT mixture prior N(0, τ²), "
           "with τ² set to the variance of the difference at the planned sample size, so the test is tuned to "
           "be most sensitive around the end of the planned run.")
footer()
