import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import COLOURS, LABELS, METRIC_LABELS, PLOT, footer, load, page, pct, usd

page("Results",
     "Effects of each email against no email, with 95% confidence intervals. Relative lifts use the "
     "**delta method**, the same approach commercial experimentation platforms use. Because six "
     "primary comparisons are tested (two emails × three metrics), p-values are **Holm-adjusted** so the "
     "chance of any false alarm stays at 5%.")
d = load()
r = d["results"]

prim = r[r.family == "primary"]
fig = go.Figure()
for a in ["Mens E-Mail", "Womens E-Mail"]:
    s = prim[prim.arm == a]
    fig.add_trace(go.Scatter(
        x=s.lift, y=[METRIC_LABELS[m] for m in s.metric], mode="markers", name=LABELS[a],
        marker=dict(size=12, color=COLOURS[a]),
        error_x=dict(type="data", symmetric=False, array=s.lift_hi - s.lift, arrayminus=s.lift - s.lift_lo),
        hovertemplate="%{y}: %{x:.0%}<extra></extra>"))
fig.add_vline(x=0, line_color="grey")
fig.update_layout(height=320, title="Relative lift vs no email (95% CI)", xaxis_tickformat=".0%",
                  scattermode="group", **PLOT)
st.plotly_chart(fig, width="stretch")


def fmt(row):
    money = row.metric in ("spend", "spend_per_visitor", "order_value")
    f = (lambda v: usd(v)) if money else (lambda v: pct(v, 2))
    return {
        "Comparison": LABELS[row.arm] + ("" if row.arm == "Mens vs Womens" else " vs no email"),
        "Metric": METRIC_LABELS[row.metric],
        "Treatment": f(row.mean_t), "Comparison group": f(row.mean_c),
        "Lift": f"{pct(row.lift, 1, True)} ({pct(row.lift_lo, 0)} to {pct(row.lift_hi, 0)})",
        "p": f"{row.p:.4f}" if row.p >= 1e-4 else "<0.0001",
        "Holm p": "-" if pd.isna(row.p_holm) else (f"{row.p_holm:.4f}" if row.p_holm >= 1e-4 else "<0.0001"),
    }


st.subheader("Primary metrics")
st.dataframe(pd.DataFrame([fmt(x) for x in prim.itertuples()]), hide_index=True, width="stretch")

st.subheader("Men's vs women's email, head to head")
st.dataframe(pd.DataFrame([fmt(x) for x in r[r.family == "head_to_head"].itertuples()]),
             hide_index=True, width="stretch")
st.caption("The relative-lift interval also carries the uncertainty in the comparison group's mean, so for a "
           "result near the 5% line (spend here) it can just touch zero while the test on the absolute "
           "difference is significant. The honest reading: the men's email very likely earns more, by an "
           "amount that is still imprecise.")

st.subheader("Diagnostic ratio metrics")
st.markdown("Spend per visitor and average order value have a random denominator (only some customers "
            "visit or buy), so a naive t-test would understate the uncertainty. These use the delta-method "
            "variance for ratio metrics. **Neither moved significantly:** the emails bring more people in, "
            "but don't change how much each buyer spends.")
st.dataframe(pd.DataFrame([fmt(x) for x in r[r.family == "diagnostic"].itertuples()]),
             hide_index=True, width="stretch")

st.subheader("Bayesian view")
st.markdown("The same data read as probabilities a manager can act on: *how likely is it that this email is "
            "better*, and *how much conversion we give up on average* by each decision (expected loss). "
            "Shipping is the right call when its expected loss is tiny next to the loss of not shipping.")
def loss(v):
    return "≈ 0" if v < 5e-6 else f"{v * 100:.3f} pts"


rows = []
for comp, vals in d["bayes"].items():
    for m, b in vals.items():
        rows.append({"Comparison": LABELS[comp] + ("" if comp == "Mens vs Womens" else " vs no email"),
                     "Metric": METRIC_LABELS[m],
                     "P(better)": f"{b['p_better']:.4f}" if b["p_better"] < 0.9999 else ">0.9999",
                     "Lift (95% credible interval)": f"{pct(b['lift_mean'], 0, True)} "
                                                     f"({pct(b['lift_lo'], 0)} to {pct(b['lift_hi'], 0)})",
                     "Expected loss if shipped": loss(b["loss_ship_t"]),
                     "Expected loss if not shipped": loss(b["loss_ship_c"])})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.caption("Flat Beta(1,1) priors on each rate. With 21,000 customers per arm the prior hardly matters, "
           "so the Bayesian and frequentist answers agree, as they should.")
footer()
