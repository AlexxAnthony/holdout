"""Shared loading, styling and formatting for the dashboard."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
ART = ROOT / "artifacts"
GITHUB = "https://github.com/AlexxAnthony/holdout"

COLOURS = {"No E-Mail": "#8c8c8c", "Mens E-Mail": "#1f4e79", "Womens E-Mail": "#c55a11"}
LABELS = {"No E-Mail": "No email", "Mens E-Mail": "Men's email", "Womens E-Mail": "Women's email",
          "Mens vs Womens": "Men's vs women's"}
METRIC_LABELS = {"visit": "Visit rate", "conversion": "Conversion rate", "spend": "Spend per customer",
                 "spend_per_visitor": "Spend per visitor", "order_value": "Average order value"}
PLOT = dict(margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.25))


@st.cache_data
def load():
    j = lambda f: json.loads((ART / f).read_text())
    c = lambda f: pd.read_csv(ART / f)
    return {
        "meta": j("meta.json"), "health": j("health.json"), "bayes": j("bayes.json"),
        "peeking": j("peeking.json"), "uplift": j("uplift.json"),
        "balance": c("balance.csv"), "results": c("results.csv"), "cuped": c("cuped.csv"),
        "replay": c("replay.csv"), "buckets": c("uplift_buckets.csv"), "segments": c("segments.csv"),
    }


def page(title: str, intro: str | None = None):
    st.set_page_config(page_title=f"{title} | Holdout", page_icon="🧪", layout="wide")
    st.title(title)
    if intro:
        st.markdown(intro)


def pct(x: float, dp: int = 1, sign: bool = False) -> str:
    return f"{x * 100:+.{dp}f}%" if sign else f"{x * 100:.{dp}f}%"


def pp(x: float, dp: int = 1) -> str:
    """Percentage points."""
    return f"{x * 100:+.{dp}f} pts"


def usd(x: float, dp: int = 2, sign: bool = False) -> str:
    return f"{'+' if sign and x >= 0 else ''}{'-' if x < 0 else ''}${abs(x):,.{dp}f}"


def footer():
    st.divider()
    st.caption(
        "Data: Kevin Hillstrom's MineThatData E-Mail Analytics challenge (2008), a real randomised test on "
        "64,000 customers over two weeks. Spend is in US dollars, as in the original data. "
        f"Code: [{GITHUB}]({GITHUB})"
    )
