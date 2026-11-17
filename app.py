"""Trendline Breakout · Meta-Label ML — Streamlit terminal. Built by Kate Wong.

Method ported from neurotrader888/TrendlineBreakoutMetaLabel (MIT).
"""
import io
import json
import threading
import time
import uuid
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from market_data import make_synthetic_pair
from trendlines import fit_trendlines_single, trendline_pass
from dataset import FEATURES
from meta_model import threshold_sweep, confusion
from worker import (TIMEFRAMES, TF_LABEL, TF_UNIT, bars_per_year, tf_defaults,
                    resample_ohlc, fit_windows, run_job)

st.set_page_config(page_title="Trendline Breakout · Meta-Label ML",
                   page_icon="📐", layout="wide", initial_sidebar_state="expanded")

AUTHOR = "Kate Wong"
REF_SEED = 7
MODEL_SEED = 69420
REF_SOURCE = "Reference BTC/ETH pair (seeded)"

C = dict(ink="#05070d", panel="#0b101c", panel2="#101728",
         btc="#f7931a", btc2="#ffb347", cyan="#22d3ee", magenta="#ff2ea6",
         green="#2ce5a7", red="#ff4d5e", violet="#8b7bff",
         text="#e8ecf6", muted="#8b94ab")
PLOTLY_CFG = {"displayModeBar": False, "responsive": True}

CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
:root{
  --ink:#05070d; --panel:#0b101c; --panel2:#101728;
  --btc:#f7931a; --btc2:#ffb347; --cyan:#22d3ee; --magenta:#ff2ea6;
  --green:#2ce5a7; --red:#ff4d5e; --violet:#8b7bff;
  --text:#e8ecf6; --muted:#8b94ab; --line:rgba(247,147,26,.16);
}
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"]{
  font-family:'Space Grotesk',system-ui,sans-serif;}
p, li, label, .stMarkdown, [data-testid="stWidgetLabel"]{font-family:'Space Grotesk',system-ui,sans-serif;}
code, pre, .mono{font-family:'IBM Plex Mono',monospace !important;}
[data-testid="stIconMaterial"], span[data-testid="stIconMaterial"],
.material-symbols-rounded, .material-symbols-outlined, .material-icons,
[data-baseweb="icon"] > svg, [data-testid="stExpander"] summary span{
  font-family:'Material Symbols Rounded','Material Symbols Outlined','Material Icons' !important;
  font-feature-settings:'liga' !important; -webkit-font-smoothing:antialiased;}
[data-testid="stSidebarCollapseButton"] button span,
[data-testid="stSidebarCollapsedControl"] button span,
[data-testid="stExpandSidebarButton"] span{font-size:0 !important;line-height:0 !important;}
[data-testid="stSidebarCollapseButton"] button::after{content:"‹‹";}
[data-testid="stSidebarCollapsedControl"] button::after{content:"››";}
[data-testid="stExpandSidebarButton"]::after{content:"››";}
[data-testid="stSidebarCollapseButton"] button::after,
[data-testid="stSidebarCollapsedControl"] button::after,
[data-testid="stExpandSidebarButton"]::after{
  font-family:'IBM Plex Mono',monospace;font-size:1rem;color:var(--muted);
  letter-spacing:-1px;line-height:1;transition:color .15s ease;}
[data-testid="stSidebarCollapseButton"] button:hover::after,
[data-testid="stSidebarCollapsedControl"] button:hover::after,
[data-testid="stExpandSidebarButton"]:hover::after{color:var(--btc2);}
[data-testid="stHeaderActionElements"], .stMarkdown a.anchor-link,
.stMarkdown h1>a,.stMarkdown h2>a,.stMarkdown h3>a,.stMarkdown h4>a{display:none !important;}
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1100px 700px at 88% -12%, rgba(34,211,238,.10), transparent 60%),
    radial-gradient(950px 640px at -12% 12%, rgba(247,147,26,.12), transparent 55%),
    radial-gradient(820px 820px at 52% 112%, rgba(255,46,166,.07), transparent 62%),
    linear-gradient(180deg,#05070d 0%,#070b14 100%);
  color:var(--text);}
[data-testid="stAppViewContainer"]::before{
  content:""; position:fixed; inset:0; z-index:0; pointer-events:none;
  background:
    repeating-linear-gradient(0deg, rgba(139,148,171,.04) 0 1px, transparent 1px 56px),
    repeating-linear-gradient(90deg, rgba(139,148,171,.04) 0 1px, transparent 1px 56px);}
[data-testid="stHeader"]{background:transparent;}
#MainMenu, footer, [data-testid="stDecoration"]{visibility:hidden; height:0;}
.block-container{padding-top:1.1rem; padding-bottom:3rem; max-width:1480px;}
*::-webkit-scrollbar{width:10px;height:10px}
*::-webkit-scrollbar-track{background:#0a0f1a}
*::-webkit-scrollbar-thumb{background:linear-gradient(180deg,var(--btc),var(--magenta));border-radius:8px;border:2px solid #0a0f1a}
*{scrollbar-width:thin; scrollbar-color:#f7931a #0a0f1a;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0a0f1a 0%,#0b1120 100%);
  border-right:1px solid var(--line); min-width:326px; max-width:326px;}
.sb-sec{font-family:'IBM Plex Mono',monospace;font-size:.66rem;letter-spacing:.2em;
  color:var(--btc);text-transform:uppercase;margin:1.1rem 0 .45rem;display:flex;align-items:center;gap:.5rem;}
.sb-sec::after{content:"";flex:1;height:1px;background:linear-gradient(90deg,var(--line),transparent);}
.sb-logo{font-weight:700;font-size:1.06rem;line-height:1.2;color:var(--text);}
.sb-logo em{display:block;font-style:normal;font-size:1.22rem;
  background:linear-gradient(90deg,var(--btc),var(--magenta),var(--cyan));
  -webkit-background-clip:text;background-clip:text;color:transparent;}
.sb-tag{font-family:'IBM Plex Mono',monospace;font-size:.6rem;color:var(--muted);letter-spacing:.16em;margin-top:.35rem;}
.hero{position:relative;padding:.6rem 0 .2rem;}
.hero h1{font-size:clamp(1.9rem,3.6vw,2.9rem);font-weight:700;letter-spacing:-.025em;
  margin:.3rem 0 .1rem;line-height:1.06;color:var(--text);}
.hero h1 .grad{display:block;background:linear-gradient(92deg,var(--btc) 5%,var(--magenta) 52%,var(--cyan) 95%);
  -webkit-background-clip:text;background-clip:text;color:transparent;}
.hero .sub{color:var(--muted);max-width:860px;font-size:.94rem;margin:.35rem 0 .6rem;}
.hero .sub b{color:var(--btc2);font-weight:600;}
.hairline{height:3px;border-radius:3px;margin:.5rem 0 .9rem;
  background:linear-gradient(90deg,var(--btc),var(--magenta),var(--cyan),var(--btc));
  background-size:300% 100%;animation:shimmer 8s linear infinite;opacity:.8;}
@keyframes shimmer{0%{background-position:0% 0}100%{background-position:300% 0}}
.chips{display:flex;flex-wrap:wrap;gap:.4rem;margin-bottom:.3rem;}
.chip{font-family:'IBM Plex Mono',monospace;font-size:.65rem;letter-spacing:.07em;
  padding:.26rem .58rem;border-radius:999px;border:1px solid var(--line);
  background:rgba(16,23,40,.75);color:var(--muted);}
.chip b{color:var(--text);font-weight:600;}
.chip.hot{border-color:rgba(247,147,26,.5);color:var(--btc2);}
.chip.cool{border-color:rgba(34,211,238,.4);color:var(--cyan);}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:.7rem;margin:.5rem 0 .9rem;}
.card{position:relative;overflow:hidden;background:linear-gradient(160deg,#0d1322,#0a0f1a);
  border:1px solid var(--line);border-radius:14px;padding:.75rem .95rem .65rem;}
.card::before{content:"";position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--btc),var(--magenta));opacity:.8;}
.card.cyan::before{background:linear-gradient(90deg,var(--cyan),var(--violet));}
.card.green::before{background:linear-gradient(90deg,var(--green),var(--cyan));}
.card.red::before{background:linear-gradient(90deg,var(--red),var(--magenta));}
.card .k{font-family:'IBM Plex Mono',monospace;font-size:.6rem;letter-spacing:.17em;
  text-transform:uppercase;color:var(--muted);}
.card .v{font-family:'IBM Plex Mono',monospace;font-size:1.5rem;font-weight:600;color:var(--text);margin-top:.12rem;}
.card .d{font-size:.7rem;color:var(--muted);margin-top:.08rem;}
.card .d.up{color:var(--green);} .card .d.dn{color:var(--red);} .card .d.warn{color:var(--btc2);}
.verdict{border-radius:16px;padding:.95rem 1.15rem;margin:.3rem 0 .9rem;border:1px solid;
  display:flex;align-items:center;gap:1rem;flex-wrap:wrap;}
.verdict .big{font-weight:700;font-size:1.2rem;letter-spacing:.02em;}
.verdict .txt{color:var(--muted);font-size:.85rem;flex:1;min-width:260px;line-height:1.55;}
.verdict.sig{border-color:rgba(44,229,167,.5);background:linear-gradient(120deg,rgba(44,229,167,.12),rgba(34,211,238,.06));}
.verdict.sig .big{color:var(--green);text-shadow:0 0 18px rgba(44,229,167,.45);}
.verdict.weak{border-color:rgba(247,147,26,.5);background:linear-gradient(120deg,rgba(247,147,26,.13),rgba(255,46,166,.05));}
.verdict.weak .big{color:var(--btc2);}
.verdict.noise{border-color:rgba(255,77,94,.5);background:linear-gradient(120deg,rgba(255,77,94,.13),rgba(255,46,166,.06));}
.verdict.noise .big{color:var(--red);}
.sec{display:flex;align-items:center;gap:.6rem;margin:1.15rem 0 .5rem;}
.sec .idx{font-family:'IBM Plex Mono',monospace;font-size:.6rem;color:var(--ink);
  background:linear-gradient(135deg,var(--btc),var(--magenta));border-radius:6px;padding:.14rem .4rem;font-weight:600;}
.sec .t{font-weight:600;font-size:1.02rem;color:var(--text);}
.sec::after{content:"";flex:1;height:1px;background:linear-gradient(90deg,var(--line),transparent);}
.panel{background:linear-gradient(150deg,#0d1424,#0a0f1a);border:1px solid var(--line);
  border-radius:16px;padding:1.05rem 1.2rem;margin:.5rem 0;}
.panel .h3{margin:.1rem 0 .45rem;font-size:1.1rem;font-weight:600;color:var(--text);}
.panel .h4{margin:.7rem 0 .3rem;font-size:.95rem;font-weight:600;color:var(--btc2);}
.panel p{color:var(--muted);font-size:.87rem;line-height:1.65;margin:.2rem 0;}
.panel ul,.panel ol{color:var(--muted);font-size:.85rem;line-height:1.7;margin:.25rem 0;padding-left:1.15rem;}
.panel li b{color:var(--text);font-weight:600;}
.steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:.75rem;margin-top:.7rem;}
.step{border:1px solid rgba(139,148,171,.18);border-radius:12px;padding:.75rem .85rem;background:rgba(11,16,28,.6);}
.step .n{font-family:'IBM Plex Mono',monospace;font-size:1.25rem;font-weight:600;
  background:linear-gradient(90deg,var(--btc),var(--magenta));-webkit-background-clip:text;background-clip:text;color:transparent;}
.step .h{font-weight:600;color:var(--text);margin:.15rem 0;}
.step .b{font-size:.79rem;color:var(--muted);line-height:1.5;}
.pd-grid{display:grid;grid-template-columns:1fr 1fr;gap:.75rem;}
.pd{border-radius:14px;padding:.9rem 1rem;border:1px solid;}
.pd.keep{border-color:rgba(44,229,167,.35);background:rgba(44,229,167,.06);}
.pd.kill{border-color:rgba(255,77,94,.35);background:rgba(255,77,94,.06);}
.pd .h4{margin:0 0 .45rem;font-size:.92rem;font-weight:600;}
.pd.keep .h4{color:var(--green);} .pd.kill .h4{color:var(--red);}
.pd ul{margin:0;padding-left:1.05rem;color:var(--muted);font-size:.82rem;line-height:1.6;}
.pd li b{color:var(--text);font-weight:600;}
.callout{border-left:3px solid var(--btc);background:rgba(247,147,26,.07);
  border-radius:0 12px 12px 0;padding:.75rem .95rem;color:var(--muted);font-size:.85rem;margin:.75rem 0;line-height:1.6;}
.callout b{color:var(--btc2);}
.callout.cool{border-left-color:var(--cyan);background:rgba(34,211,238,.06);}
.callout.cool b{color:var(--cyan);}
.strat-block{border:1px solid var(--line);border-radius:16px;padding:1rem 1.15rem;margin:.7rem 0;
  background:linear-gradient(155deg,#0d1424,#0a0f1a);}
.strat-block .hd{display:flex;align-items:baseline;gap:.6rem;flex-wrap:wrap;margin-bottom:.45rem;}
.strat-block .nm{font-weight:700;font-size:1.05rem;color:var(--text);}
.strat-block .tagline{font-family:'IBM Plex Mono',monospace;font-size:.66rem;color:var(--cyan);letter-spacing:.08em;}
.strat-block ol{margin:.25rem 0 .7rem;padding-left:1.15rem;color:var(--muted);font-size:.84rem;line-height:1.65;}
.strat-block ol b{color:var(--text);font-weight:600;}
.strat-block .ask{font-size:.8rem;color:var(--btc2);border-top:1px dashed rgba(247,147,26,.25);padding-top:.55rem;margin-top:.6rem;}
button[data-baseweb="tab"]{font-family:'IBM Plex Mono',monospace;font-size:.76rem;letter-spacing:.05em;color:var(--muted);}
button[data-baseweb="tab"][aria-selected="true"]{color:var(--btc2);}
div[data-baseweb="tab-highlight"]{background:linear-gradient(90deg,var(--btc),var(--magenta));height:3px;}
div[data-baseweb="tab-border"]{background:var(--line);}
[data-testid="stBaseButton-primary"]{background:linear-gradient(135deg,var(--btc),var(--magenta));
  color:#0a0710;border:none;font-weight:700;letter-spacing:.04em;box-shadow:0 4px 20px rgba(247,147,26,.32);}
[data-testid="stBaseButton-primary"]:hover{transform:translateY(-1px);box-shadow:0 6px 26px rgba(255,46,166,.42);color:#0a0710;}
[data-testid="stBaseButton-secondary"]{background:var(--panel2);border:1px solid rgba(34,211,238,.32);color:var(--text);}
[data-baseweb="slider"] div[role="slider"]{background:var(--btc);box-shadow:0 0 10px rgba(247,147,26,.55);}
[data-baseweb="select"]>div{background:var(--panel2);border-color:var(--line);}
[data-baseweb="checkbox"] [aria-checked="true"]{background-color:var(--btc) !important;}
[data-testid="stFileUploaderDropzone"]{background:var(--panel2);border:1px dashed rgba(247,147,26,.4);border-radius:12px;}
[data-testid="stFileUploaderDropzoneInstructions"] span{color:var(--muted);}
[data-testid="stFileUploaderDropzoneInstructions"] small{font-size:0 !important;line-height:0 !important;display:block;}
[data-testid="stFileUploaderDropzoneInstructions"] small::after{
  content:"max 200 MB per file · CSV, PQ, PARQUET";display:block;font-size:.72rem;
  line-height:1.45;color:var(--muted);font-family:'IBM Plex Mono',monospace;}
[data-testid="stProgress"] div[role="progressbar"]{background:rgba(139,148,171,.15);}
[data-testid="stProgress"] div[role="progressbar"]>div{background:linear-gradient(90deg,var(--btc),var(--magenta),var(--cyan));}
[data-testid="stExpander"] details{background:var(--panel);border:1px solid var(--line);border-radius:12px;}
[data-testid="stCodeBlock"] pre{max-height:440px;overflow:auto;background:#0a0f1a !important;
  border:1px solid var(--line);border-radius:12px;}
[data-testid="stTable"] table{background:transparent;color:var(--text);font-family:'IBM Plex Mono',monospace;font-size:.78rem;}
[data-testid="stTable"] th{color:var(--btc2) !important;border-bottom:1px solid var(--line) !important;}
[data-testid="stTable"] td{border-bottom:1px solid rgba(139,148,171,.12) !important;}
[data-testid="stAlert"]{border-radius:12px;}
.hist-row{display:flex;align-items:center;gap:.6rem;padding:.42rem .6rem;border:1px solid rgba(139,148,171,.14);
  border-radius:10px;margin-bottom:.35rem;background:rgba(11,16,28,.55);
  font-family:'IBM Plex Mono',monospace;font-size:.71rem;flex-wrap:wrap;}
.hist-row .lbl{flex:1;color:var(--text);min-width:180px;}
.hist-row .m{color:var(--muted);}
.badge{padding:.12rem .5rem;border-radius:999px;font-size:.61rem;font-weight:600;}
.badge.sig{background:rgba(44,229,167,.15);color:var(--green);}
.badge.weak{background:rgba(247,147,26,.15);color:var(--btc2);}
.badge.noise{background:rgba(255,77,94,.15);color:var(--red);}
.foot{margin-top:2rem;padding-top:.85rem;border-top:1px solid var(--line);
  font-family:'IBM Plex Mono',monospace;font-size:.63rem;letter-spacing:.13em;color:var(--muted);}
.foot b{color:var(--btc2);font-weight:600;}
@media (max-width:760px){[data-testid="stSidebar"]{min-width:0;max-width:none;}
  .pd-grid{grid-template-columns:1fr;} .card .v{font-size:1.22rem;}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------- notices: popup only, never inline ----------------
def notify(text, hints=None, technical=None):
    sig = hash((text, tuple(hints or ())))
    seen = st.session_state.setdefault("_notices_seen", set())
    if sig in seen:
        return
    seen.add(sig)
    if technical:
        print("[TLB]", technical)
    st.session_state["_popup"] = {"text": text, "hints": list(hints or ())}


def _popup_body():
    p = st.session_state.get("_popup") or {}
    st.markdown(f'<div class="panel"><div class="h3">{p.get("text","")}</div></div>',
                unsafe_allow_html=True)
    if p.get("hints"):
        st.markdown('<div class="panel"><div class="h4">What to try</div><ul>'
                    + "".join(f"<li>{h}</li>" for h in p["hints"]) + "</ul></div>",
                    unsafe_allow_html=True)
    if st.button("Dismiss", type="primary", use_container_width=True, key="popup_dismiss"):
        st.session_state.pop("_popup", None)
        st.rerun()


_popup_dialog = st.dialog("Notice")(_popup_body) if hasattr(st, "dialog") else None


def flush_popup():
    p = st.session_state.get("_popup")
    if not p:
        return
    if _popup_dialog is not None:
        try:
            _popup_dialog(); return
        except Exception:
            pass
    try:
        st.toast(p["text"], icon="⚠️")
    except Exception:
        pass
    st.session_state.pop("_popup", None)


def reset_notices():
    st.session_state.pop("_notices_seen", None)
    st.session_state.pop("_popup", None)


# ---------------- helpers ----------------
def show_fig(fig, key=None):
    try:
        st.plotly_chart(fig, width="stretch", key=key, config=PLOTLY_CFG)
    except Exception:
        st.plotly_chart(fig, use_container_width=True, key=key, config=PLOTLY_CFG)


def base_fig(height=340, title=None, top=None):
    fig = go.Figure()
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,15,26,.55)",
        font=dict(family="IBM Plex Mono, monospace", size=11, color="#aab3c8"),
        margin=dict(l=54, r=16, t=top or (62 if title else 26), b=38),
        title=dict(text=title, x=0.0, xanchor="left", y=.98, yanchor="top",
                   font=dict(size=12.5, color=C["text"], family="Space Grotesk")),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=C["panel2"], bordercolor=C["btc"],
                        font=dict(family="IBM Plex Mono", size=11)))
    fig.update_xaxes(gridcolor="rgba(139,148,171,.10)", zerolinecolor="rgba(139,148,171,.22)")
    fig.update_yaxes(gridcolor="rgba(139,148,171,.10)", zerolinecolor="rgba(139,148,171,.22)")
    return fig


def fmt_n(x, spec="{:.3f}", dash="—"):
    try:
        return spec.format(x) if np.isfinite(x) else dash
    except Exception:
        return dash


def sec(idx, title):
    st.markdown(f'<div class="sec"><span class="idx">{idx}</span><span class="t">{title}</span></div>',
                unsafe_allow_html=True)


def cards_html(items):
    h = '<div class="cards">'
    for it in items:
        h += f'<div class="card {it.get("cls","")}"><div class="k">{it["k"]}</div><div class="v">{it["v"]}</div>'
        if it.get("d"):
            h += f'<div class="d {it.get("dc","")}">{it["d"]}</div>'
        h += "</div>"
    return h + "</div>"


def num_slider(label, lo, hi, default, key):
    lo = int(lo); hi = int(max(int(hi), lo + 1))
    v = st.session_state.get(key)
    if v is not None and not (lo <= int(v) <= hi):
        st.session_state.pop(key, None)
    return int(st.slider(label, lo, hi, int(min(max(int(default), lo), hi)), key=key))


def year_bounds(df):
    now = int(pd.Timestamp.now(tz="UTC").year)
    if df is None or len(df) == 0:
        return now - 1, now
    return int(df.index.year.min()), int(max(int(df.index.year.max()), now, int(df.index.year.min())))


def year_defaults(df):
    lo, hi = year_bounds(df)
    return (lo, hi)


def clamp_years(v, lo, hi, fb):
    try:
        a, b = int(v[0]), int(v[1])
    except Exception:
        return fb
    a = max(lo, min(a, hi)); b = max(a, min(b, hi))
    return a, b


def year_range(label, lo, hi, default, key):
    lo, hi = int(lo), int(hi)
    if hi <= lo:
        st.caption(f"{label}: {lo}")
        return lo, lo
    v = st.session_state.get(key)
    if isinstance(v, (tuple, list)) and len(v) == 2:
        a, b = int(v[0]), int(v[1])
        if not (lo <= a <= b <= hi):
            st.session_state.pop(key, None)
    elif v is not None:
        st.session_state.pop(key, None)
    a, b = st.slider(label, lo, hi, (int(default[0]), int(default[1])), key=key)
    return int(a), int(b)


def verdict_of(lift):
    if lift >= 0.15:
        return "FILTER ADDS VALUE", "sig"
    if lift >= 0.0:
        return "MARGINAL IMPROVEMENT", "weak"
    return "FILTER HURTS", "noise"


# ---------------- plots ----------------
def equity3_fig(curves, title="Equity · out-of-sample"):
    fig = base_fig(370, title)
    x = curves["x"]
    for key, name, col, w in (("bh", "Buy & hold", C["violet"], 1.4),
                              ("all", "All breakouts", C["cyan"], 1.8),
                              ("meta", "Meta-labeled", C["btc"], 2.8)):
        y = np.asarray(curves[key], float)
        fig.add_trace(go.Scatter(x=x[:len(y)], y=y, mode="lines", name=name,
                                 line=dict(color=col, width=w, dash="dot" if key == "bh" else "solid"),
                                 hovertemplate="%{y:+.3f}<extra>" + name + "</extra>"))
    fig.add_hline(y=0, line_color="rgba(139,148,171,.45)", line_width=1)
    fig.update_yaxes(title="Σ log return")
    return fig


def price_lines_fig(df, lines, lookback, tail=420):
    n = len(df)
    a = max(0, n - int(tail))
    d = df.iloc[a:]
    fig = base_fig(430, "Price · projected trendlines", top=70)
    fig.add_trace(go.Scatter(x=d.index, y=d["close"], mode="lines", name="Close",
                             line=dict(color=C["text"], width=1.5)))
    for key, name, col in (("resist", "Resistance", C["red"]), ("support", "Support", C["green"])):
        y = np.asarray(lines[key][a:], float)
        fig.add_trace(go.Scatter(x=d.index[:len(y)], y=np.exp(y), mode="lines", name=name,
                                 line=dict(color=col, width=1.6, dash="dash")))
    fig.update_yaxes(title="price")
    fig.update_layout(hovermode="x")
    return fig


def slope_fig(df, lines):
    fig = base_fig(230, "Trendline slopes")
    fig.add_trace(go.Scatter(x=df.index, y=lines["r_slope"], name="Resistance",
                             line=dict(color=C["red"], width=1.4)))
    fig.add_trace(go.Scatter(x=df.index, y=lines["s_slope"], name="Support",
                             line=dict(color=C["green"], width=1.4)))
    fig.add_hline(y=0, line_color="rgba(139,148,171,.4)", line_width=1)
    fig.update_yaxes(title="log slope / bar")
    return fig


def position_fig(x, sig):
    fig = base_fig(150, "Position", top=34)
    fig.add_trace(go.Scatter(x=x, y=sig, mode="lines", line=dict(color=C["cyan"], width=1.4, shape="hv"),
                             fill="tozeroy", fillcolor="rgba(34,211,238,.18)"))
    fig.update_yaxes(range=[-.2, 1.2], tickvals=[0, 1], ticktext=["flat", "long"], title=None)
    fig.update_layout(showlegend=False, margin=dict(l=54, r=16, t=34, b=26))
    return fig


def trade_fig(df, trade, lookback):
    e, x = int(trade["entry_i"]), int(trade["exit_i"])
    a = max(0, e - lookback)
    d = df.iloc[a:x + 1]
    fig = base_fig(400, f"Trade #{int(trade.name)} · {trade['exit_reason']}", top=70)
    fig.add_trace(go.Candlestick(x=d.index, open=d["open"], high=d["high"], low=d["low"], close=d["close"],
                                 increasing_line_color=C["green"], decreasing_line_color=C["red"],
                                 increasing_fillcolor="rgba(44,229,167,.5)",
                                 decreasing_fillcolor="rgba(255,77,94,.5)", name="OHLC"))
    xs = [df.index[a], df.index[min(e, len(df) - 1)]]
    ys = [float(trade["intercept"]), float(trade["intercept"] + trade["slope"] * (e - a))]
    fig.add_trace(go.Scatter(x=xs, y=np.exp(ys), mode="lines", name="Resistance",
                             line=dict(color=C["text"], width=1.6, dash="dash")))
    for key, name, col in (("tp", "Take profit", C["green"]), ("sl", "Stop loss", C["red"])):
        fig.add_trace(go.Scatter(x=[df.index[e], df.index[x]], y=[np.exp(trade[key])] * 2,
                                 mode="lines", name=name, line=dict(color=col, width=1.3, dash="dot")))
    fig.add_trace(go.Scatter(x=[df.index[e]], y=[np.exp(trade["entry_p"])], mode="markers",
                             name="Entry", marker=dict(symbol="triangle-up", size=12, color=C["btc"])))
    fig.add_trace(go.Scatter(x=[df.index[x]], y=[np.exp(trade["exit_p"])], mode="markers",
                             name="Exit", marker=dict(symbol="x", size=11, color=C["magenta"])))
    fig.update_layout(xaxis_rangeslider_visible=False, hovermode="x")
    fig.update_yaxes(title="price")
    return fig


def prob_fig(trades, threshold):
    s = trades[np.isfinite(trades["model_prob"])]
    fig = base_fig(300, "Model probability at entry")
    taken = s[s["model_prob"] > threshold]["model_prob"]
    skipped = s[s["model_prob"] <= threshold]["model_prob"]
    fig.add_trace(go.Histogram(x=skipped, nbinsx=28, name="Skipped", marker_color=C["muted"], opacity=.7))
    fig.add_trace(go.Histogram(x=taken, nbinsx=28, name="Taken", marker_color=C["btc"], opacity=.85))
    fig.add_vline(x=threshold, line_color=C["cyan"], line_width=2, line_dash="dash",
                  annotation_text=f"threshold {threshold:.2f}", annotation_position="top",
                  annotation_font=dict(color=C["cyan"], size=10))
    fig.update_layout(barmode="overlay", bargap=.05)
    fig.update_xaxes(title="P(profit)")
    fig.update_yaxes(title="trades")
    return fig


def imp_fig(importances):
    names = {"resist_s": "resistance slope / ATR", "tl_err": "mean line error / ATR",
             "max_dist": "max line distance / ATR", "vol": "volume vs median",
             "adx": "ADX (trend strength)"}
    k = list(importances.keys())
    v = [importances[i] for i in k]
    fig = base_fig(240, "Feature importances")
    fig.add_trace(go.Bar(y=[names.get(i, i) for i in k], x=v, orientation="h",
                         marker_color=[C["btc"], C["cyan"], C["magenta"], C["violet"], C["green"]],
                         opacity=.88, text=[f"{q:.1%}" for q in v], textposition="outside",
                         textfont=dict(color=C["text"])))
    fig.update_layout(showlegend=False, margin=dict(l=150, r=16, t=44, b=34))
    fig.update_xaxes(title="mean Gini importance across refits", range=[0, max(v) * 1.3 + 1e-6])
    return fig


def sweep_fig(sweep, threshold):
    fig = base_fig(300, "Threshold sweep")
    fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["pf"], name="Profit factor",
                             mode="lines+markers", line=dict(color=C["btc"], width=2.2)))
    fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep["win_rate"], name="Win rate",
                             mode="lines+markers", line=dict(color=C["cyan"], width=1.6, dash="dot"),
                             yaxis="y2"))
    fig.add_vline(x=threshold, line_color=C["magenta"], line_width=1.6, line_dash="dash")
    fig.add_hline(y=1.0, line_color="rgba(139,148,171,.4)", line_width=1)
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", tickformat=".0%",
                                  gridcolor="rgba(0,0,0,0)", title="win rate"))
    fig.update_xaxes(title="probability threshold")
    fig.update_yaxes(title="profit factor")
    return fig


def confusion_fig(cm):
    z = [[cm["tp"], cm["fp"]], [cm["fn"], cm["tn"]]]
    txt = [[f"TP {cm['tp']}", f"FP {cm['fp']}"], [f"FN {cm['fn']}", f"TN {cm['tn']}"]]
    fig = base_fig(260, "Confusion · taken vs profitable")
    fig.add_trace(go.Heatmap(z=z, text=txt, texttemplate="%{text}", showscale=False,
                             x=["predicted profit", "predicted loss"],
                             y=["actually profit", "actually loss"],
                             colorscale=[[0, "#0b101c"], [1, "#f7931a"]],
                             textfont=dict(color=C["text"], size=13)))
    fig.update_layout(hovermode="closest")
    return fig


def scatter_fig(trades):
    s = trades[np.isfinite(trades["model_prob"])]
    fig = base_fig(300, "Resistance slope vs trade return")
    for mask, name, col in ((s["return"] > 0, "winner", C["green"]), (s["return"] <= 0, "loser", C["red"])):
        d = s[mask]
        fig.add_trace(go.Scattergl(x=d["resist_s"], y=d["return"], mode="markers", name=name,
                                   marker=dict(size=5, color=col, opacity=.55)))
    fig.add_hline(y=0, line_color="rgba(139,148,171,.45)", line_width=1)
    fig.update_xaxes(title="resist_s (slope / ATR)")
    fig.update_yaxes(title="log return")
    fig.update_layout(hovermode="closest")
    return fig


# ---------------- data ----------------
@st.cache_data(show_spinner="Synthesizing reference feed…")
def synth_pair(seed=REF_SEED):
    return make_synthetic_pair(seed)


@st.cache_data(show_spinner=False)
def parse_bytes(name, data):
    bio = io.BytesIO(data)
    if name.lower().endswith(".csv"):
        df = pd.read_csv(bio, float_precision="round_trip")
        c0 = df.columns[0]
        df[c0] = pd.to_datetime(df[c0])
        df = df.set_index(c0)
    else:
        df = pd.read_parquet(bio)
    df.columns = [str(c).lower() for c in df.columns]
    df.index = pd.to_datetime(df.index).astype("datetime64[s]")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    need = {"open", "high", "low", "close"}
    if not need.issubset(df.columns):
        raise ValueError("needs columns open / high / low / close")
    keep = ["open", "high", "low", "close"] + (["volume"] if "volume" in df.columns else [])
    df = df[keep].astype(float).dropna()
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    if len(df) < 12:
        raise ValueError("needs at least 12 usable rows")
    return df


@st.cache_data(show_spinner=False)
def to_tf(df, tf):
    return resample_ohlc(df, tf)


@st.cache_data(show_spinner=False)
def preview_lines(win, lookback, step):
    """Cached trendline fit for the Trendlines tab — no refit on every widget change."""
    return trendline_pass(np.log(win["close"].to_numpy(dtype=float)), int(lookback), int(step))


@st.cache_data(show_spinner="Preparing reference CSV…")
def reference_csv(symbol):
    b, e = synth_pair()
    df = (b if symbol == "BTC" else e).copy()
    df.index.name = "timestamp"
    return df.to_csv().encode("utf-8")


@st.cache_data(show_spinner="Packaging reference pair…")
def reference_zip():
    readme = (
        "Trendline Breakout · Meta-Label ML — reference pair\r\n"
        f"Seeded synthetic feed, make_synthetic_pair(seed={REF_SEED}). NOT exchange history.\r\n"
        "Hourly OHLCV, 2016-01-01 → 2020-12-31, 43,848 bars each.\r\n"
        "Schema: timestamp,open,high,low,close,volume (UTC-naive, ascending).\r\n"
        f"Built by {AUTHOR}. Method after neurotrader888/TrendlineBreakoutMetaLabel (MIT).\r\n"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("BTCUSD_reference_hourly.csv", reference_csv("BTC"))
        z.writestr("ETHUSD_reference_hourly.csv", reference_csv("ETH"))
        z.writestr("README.txt", readme)
    return buf.getvalue()


def use_reference_pair():
    reset_notices()
    st.session_state["w_source"] = REF_SOURCE


def resolve_data(source, up1):
    if source == "Upload files":
        if up1 is None:
            b, _ = synth_pair()
            return b, "DEMO FEED", "No file uploaded yet — running on the seeded reference feed."
        return parse_bytes(up1.name, up1.getvalue()), up1.name, None
    b, _ = synth_pair()
    return b, "REFERENCE PAIR", None


# ---------------- config & callbacks ----------------
def collect_cfg():
    g = st.session_state.get
    tf = g("w_tf", "1D")
    d = tf_defaults(tf)
    df = st.session_state.get("ohlcv")
    lo, hi = year_bounds(df)
    yrs = clamp_years(g("w_years", year_defaults(df)), lo, hi, year_defaults(df))
    depth = g("w_depth", 3)
    return dict(
        tf=tf, lookback=int(g("w_lb", d["lookback"])), line_step=int(g("w_step_line", d["line_step"])),
        hold=int(g("w_hold", d["hold"])), tp=float(g("w_tp", 3.0)), sl=float(g("w_sl", 3.0)),
        atr=int(g("w_atr", d["atr"])), train=int(g("w_train", d["train"])),
        step=int(g("w_step", d["step"])), threshold=float(g("w_thr", 0.50)),
        trees=int(g("w_trees", 300)), depth=None if depth == "Unlimited" else int(depth),
        y0=int(yrs[0]), y1=int(yrs[1]),
    )


def start_run():
    active = st.session_state.get("active")
    if active and active.get("status") == "running":
        return
    reset_notices()
    df = st.session_state.get("ohlcv")
    if df is None or len(df) < 12:
        notify("Not enough bars at this bar size to run the strategy.",
               ["Pick a finer bar size — Hourly or Daily.", "Load a longer file."])
        return
    cfg = collect_cfg()
    win = df[(df.index.year >= cfg["y0"]) & (df.index.year <= cfg["y1"])]
    if len(win) < 12:
        notify("No bars in the selected years.", ["Widen the year range.", "Check the feed's last year in the sidebar."])
        return
    try:
        fit_windows(len(win), cfg["train"], cfg["step"])
    except ValueError as ex:
        notify(str(ex), ["Widen the years.", "Shorten the training window."], technical=str(ex))
        return
    job = dict(id=uuid.uuid4().hex[:10], status="running", phase="Initializing…", progress=0.0,
               config=cfg, start=time.time(), elapsed=0.0, error=None, stop=threading.Event(),
               trades=None, curves=None, metrics=None, importances=None, sweep=None,
               confusion=None, lines=None, train_log=None, window_desc="", _finalized=False)
    th = threading.Thread(target=run_job, args=(job, win.copy(), cfg), daemon=True)
    job["thread"] = th
    st.session_state["active"] = job
    st.session_state["win"] = win
    th.start()


def request_stop():
    a = st.session_state.get("active")
    if a and "stop" in a:
        a["stop"].set()


def clear_active():
    a = st.session_state.get("active")
    if a and a.get("status") == "running":
        return
    reset_notices()
    st.session_state["active"] = None


def finalize(job):
    m = job["metrics"]
    lift = m["meta"]["pf"] - m["all"]["pf"]
    job["verdict"] = verdict_of(lift if np.isfinite(lift) else -1)
    job["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    job["_finalized"] = True
    saved = {k: v for k, v in job.items() if k not in ("stop", "thread")}
    hist = st.session_state.setdefault("history", [])
    hist.insert(0, saved)
    del hist[16:]


def load_run(rid):
    for r in st.session_state.get("history", []):
        if r["id"] == rid:
            st.session_state["active"] = r
            return


def delete_run(rid):
    st.session_state["history"] = [r for r in st.session_state.get("history", []) if r["id"] != rid]


def json_safe(o):
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def results_body(job, live=False):
    m = job["metrics"]
    meta, allx, bh = m["meta"], m["all"], m["bh"]
    if not live:
        lift = meta["pf"] - allx["pf"]
        label, cls = job.get("verdict", verdict_of(lift if np.isfinite(lift) else -1))
        txt = (f"The meta-label filter took {m['n_taken']} of {m['n_scored']} scored breakouts. "
               f"Profit factor {fmt_n(meta['pf'],'{:.3f}')} versus {fmt_n(allx['pf'],'{:.3f}')} unfiltered "
               f"and {fmt_n(bh['pf'],'{:.3f}')} for buy & hold, on out-of-sample bars only.")
        st.markdown(f'<div class="verdict {cls}"><span class="big">{label}</span>'
                    f'<span class="txt">{txt}</span></div>', unsafe_allow_html=True)
        st.markdown(cards_html([
            dict(k="Meta PF", v=fmt_n(meta["pf"], "{:.3f}"),
                 cls="green" if meta["pf"] > allx["pf"] else "red",
                 d=f"unfiltered {fmt_n(allx['pf'],'{:.3f}')}", dc="warn"),
            dict(k="Win rate", v=fmt_n(m["meta_trades"]["win"], "{:.1%}"), cls="cyan",
                 d=f"all trades {fmt_n(m['all_trades']['win'],'{:.1%}')}"),
            dict(k="Avg trade", v=fmt_n(m["meta_trades"]["avg"], "{:+.4f}"),
                 d=f"all {fmt_n(m['all_trades']['avg'],'{:+.4f}')}",
                 dc="up" if m["meta_trades"]["avg"] > 0 else "dn"),
            dict(k="Time in market", v=fmt_n(meta["exposure"], "{:.0%}"), cls="cyan",
                 d=f"unfiltered {fmt_n(allx['exposure'],'{:.0%}')}"),
            dict(k="Max drawdown", v=fmt_n(meta["mdd"], "{:.1%}"), cls="red"),
        ]), unsafe_allow_html=True)
        st.caption(f"{job.get('window_desc','')} · elapsed {int(job.get('elapsed',0))}s · "
                   f"{job.get('finished_at','')}")
    show_fig(equity3_fig(job["curves"]), key=f"eq_{job['id']}_{live}")
    st.caption("Amber = meta-labeled · cyan = every breakout · violet = buy & hold. Gross of costs.")


def onboarding():
    st.markdown("""
<div class="panel">
  <div class="h3">Draw the line. Break the line. Let the model decide whether to take it.</div>
  <p>A trendline is fitted to every rolling window by rotating a least-squares line around its most extreme
  pivot until it touches price without cutting through it. A close beyond the projected resistance is a
  <b>primary breakout</b>. A random forest then answers the only question that matters:
  <b>will this particular breakout make money?</b> That second model is the meta-label.</p>
  <div class="steps">
    <div class="step"><div class="n">01</div><div class="h">Fit trendlines</div><div class="b">Pivot-anchored slope optimisation on log prices, window excludes the current bar, line projected forward.</div></div>
    <div class="step"><div class="n">02</div><div class="h">Label the breakouts</div><div class="b">Triple barrier — take profit, stop loss, hold period — sized in ATR units. Label 1 if the trade paid.</div></div>
    <div class="step"><div class="n">03</div><div class="h">Walk forward</div><div class="b">Re-fit the forest on trailing closed trades only, gate each new entry on P(profit) &gt; threshold.</div></div>
  </div>
  <div class="callout" style="margin-top:.85rem"><b>→ Load a feed, pick a bar size, press RUN STRATEGY.</b>
  Progress streams live and you can stop anytime. The reference pair and the full method are in Method &amp; Guide.</div>
</div>
""", unsafe_allow_html=True)


@st.fragment(run_every=0.6)
def live_fragment():
    job = st.session_state.get("active")
    if job is None or job.get("status") != "running":
        if job is not None and job.get("status") == "done" and not job.get("_finalized"):
            finalize(job)
        st.rerun(scope="app")
        return
    st.progress(min(float(job.get("progress", 0.0)), 1.0))
    c1, c2 = st.columns([4, 1])
    with c1:
        st.markdown(cards_html([
            dict(k="Phase", v="⚙ LIVE", cls="cyan", d=job.get("phase", "")),
            dict(k="Progress", v=f"{job.get('progress',0)*100:.0f}%",
                 d=f"{int(time.time()-job['start'])}s elapsed"),
            dict(k="Bar size", v=TF_LABEL[job["config"]["tf"]], cls="cyan"),
            dict(k="Lookback", v=str(job["config"]["lookback"]), d=f"refit every {job['config']['line_step']} bar(s)"),
        ]), unsafe_allow_html=True)
    with c2:
        st.markdown("<div style='height:.55rem'></div>", unsafe_allow_html=True)
        st.button("■ STOP", on_click=request_stop, key="stop_btn", use_container_width=True)
    if job.get("metrics"):
        results_body(job, live=True)
    else:
        st.info("Trendlines are fitted first, then the dataset, then the walk-forward model.")
    if job["status"] != "running":
        if job.get("status") == "done" and not job.get("_finalized"):
            finalize(job)
        st.rerun(scope="app")


def history_block():
    hist = st.session_state.get("history", [])
    if not hist:
        return
    sec("H", "Run history")
    for r in hist:
        label, cls = r.get("verdict", ("—", "weak"))
        c1, c2, c3 = st.columns([6, 1, 1])
        with c1:
            m = r["metrics"]
            st.markdown(
                f'<div class="hist-row"><span class="lbl">{TF_LABEL[r["config"]["tf"]]} · '
                f'L={r["config"]["lookback"]} · thr {r["config"]["threshold"]:.2f}</span>'
                f'<span class="m">PF {fmt_n(m["meta"]["pf"],"{:.3f}")} vs {fmt_n(m["all"]["pf"],"{:.3f}")}</span>'
                f'<span class="badge {cls}">{label}</span>'
                f'<span class="m">{r.get("finished_at","")}</span></div>', unsafe_allow_html=True)
        c2.button("view", key=f"v_{r['id']}", on_click=load_run, args=(r["id"],))
        c3.button("✕", key=f"d_{r['id']}", on_click=delete_run, args=(r["id"],))


# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown('<div class="sb-logo">Trendline Breakout<em>Meta-Label ML</em></div>'
                f'<div class="sb-tag">BUILT BY {AUTHOR.upper()}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-sec">01 · Data</div>', unsafe_allow_html=True)
    opts = ["Upload files", REF_SOURCE]
    source = st.selectbox("Source", opts, index=0, key="w_source")
    up1 = None
    if source == "Upload files":
        up1 = st.file_uploader("OHLCV (.csv / .pq / .parquet)", type=["csv", "pq", "parquet"],
                               key="up_1", help="timestamp + open/high/low/close (+volume) · max 200 MB per file")
        st.caption("max 200 MB per file. No file? Download the reference pair in Method & Guide → Data.")
    try:
        raw, src_label, notice = resolve_data(source, up1)
    except Exception as e:
        raw, src_label, notice = synth_pair()[0], "DEMO FEED", None
        notify("That file could not be read, so the reference feed is in use.",
               ["Columns: timestamp, open, high, low, close (volume optional).",
                "Positive prices, no empty rows.", "Keep it under max 200 MB."],
               technical=f"{type(e).__name__}: {e}")
    if notice:
        st.caption(notice)

    st.markdown('<div class="sb-sec">02 · Bar size</div>', unsafe_allow_html=True)
    tf = st.select_slider("Timeframe", options=list(TIMEFRAMES), value="1D",
                          format_func=lambda k: TF_LABEL[k], key="w_tf")
    if st.session_state.get("tf_prev") != tf:
        st.session_state["tf_prev"] = tf
        reset_notices()
        for k in ("w_lb", "w_step_line", "w_hold", "w_atr", "w_train", "w_step"):
            st.session_state.pop(k, None)
    df = to_tf(raw, tf)
    st.session_state["ohlcv"] = df
    d = tf_defaults(tf)
    unit = TF_UNIT[tf]
    n_bars = len(df)
    ylo, yhi = year_bounds(df)
    ydef = year_defaults(df)
    st.caption(f"{src_label} · {n_bars:,} {unit} bars · data {ylo}–{int(df.index.year.max())}"
               + (" · volume present" if "volume" in df.columns else " · no volume column"))
    if yhi > int(df.index.year.max()):
        st.caption(f"Year sliders run to {yhi} (current year); this feed ends in {int(df.index.year.max())}.")
    if n_bars < 60:
        st.caption(f"⚠ {n_bars} {unit} bars — indicative only at this sample size.")
    y0, y1 = year_range("Years (inclusive)", ylo, yhi, ydef, "w_years")
    n_win = int(((df.index.year >= y0) & (df.index.year <= y1)).sum())

    st.markdown('<div class="sb-sec">03 · Trendline</div>', unsafe_allow_html=True)
    lb = num_slider(f"Lookback ({unit} bars)", 5, max(6, min(240, max(6, n_win - 2))), d["lookback"], "w_lb")
    lst = num_slider("Refit every (bars)", 1, 12, d["line_step"], "w_step_line")
    st.caption("1 = exact source behaviour. Higher reuses the fit and projects it forward — faster on hourly.")

    st.markdown('<div class="sb-sec">04 · Meta-label model</div>', unsafe_allow_html=True)
    hold = num_slider(f"Hold period ({unit} bars)", 1, max(2, min(120, max(2, n_win // 4))), d["hold"], "w_hold")
    ca, cb = st.columns(2)
    with ca:
        tp = st.slider("TP × ATR", 0.5, 8.0, 3.0, step=0.5, key="w_tp")
    with cb:
        sl = st.slider("SL × ATR", 0.5, 8.0, 3.0, step=0.5, key="w_sl")
    atrn = num_slider("ATR / volume lookback", max(lb, 3), max(lb + 1, min(400, max(4, n_win - 1))),
                      max(d["atr"], lb), "w_atr")
    tr = num_slider(f"Training window ({unit} bars)", 10, max(11, n_win - 2), d["train"], "w_train")
    stp = num_slider(f"Re-train every ({unit} bars)", 1, max(2, n_win // 2), d["step"], "w_step")
    thr = st.slider("Probability threshold", 0.30, 0.80, 0.50, step=0.01, key="w_thr")
    trees = num_slider("Trees per refit", 50, 1000, 300, "w_trees")
    depth = st.selectbox("Max depth", [2, 3, 4, 6, "Unlimited"], index=1, format_func=str, key="w_depth")
    if n_win >= 12:
        t_, s_ = fit_windows(n_win, tr, stp)
        st.caption(f"auto-fit → train {t_:,} · step {s_:,} · OOS {n_win - t_:,} bars")

    st.markdown('<div class="sb-sec">05 · Execute</div>', unsafe_allow_html=True)
    active = st.session_state.get("active")
    running = bool(active and active.get("status") == "running")
    st.button("📐 RUN STRATEGY", type="primary", on_click=start_run, disabled=running, use_container_width=True)
    st.button("Clear results", on_click=clear_active, disabled=running, use_container_width=True)
    if running:
        st.caption(f"● running — {active.get('progress',0)*100:.0f}%")

cfg_now = collect_cfg()
tf_now = cfg_now["tf"]
unit_now = TF_UNIT[tf_now]
st.markdown(f"""
<div class="hero">
  <h1>Trendline Breakout<span class="grad">Meta-Label ML</span></h1>
  <div class="sub">A geometric primary signal, filtered by a <b>machine-learning meta-label</b>. Trendlines are fitted by
  pivot-anchored slope optimisation, breakouts are labelled by an <b>ATR triple barrier</b>, and a walk-forward random forest
  decides which breakouts to take — at the bar size you choose.</div>
  <div class="chips">
    <span class="chip hot">FEED <b>{src_label}</b></span>
    <span class="chip cool">BARS <b>{TF_LABEL[tf_now]} · {n_bars:,}</b></span>
    <span class="chip">LOOKBACK <b>{cfg_now['lookback']}</b></span>
    <span class="chip">THRESHOLD <b>{cfg_now['threshold']:.2f}</b></span>
    <span class="chip">BUILT BY <b>{AUTHOR}</b></span>
  </div>
  <div class="hairline"></div>
</div>
""", unsafe_allow_html=True)

tab_lab, tab_tl, tab_model, tab_trades, tab_method = st.tabs(
    ["BREAKOUT LAB", "TRENDLINES", "MODEL", "TRADES", "METHOD & GUIDE"])

with tab_lab:
    active = st.session_state.get("active")
    if active and active.get("status") == "running":
        sec("L", "Live run")
        live_fragment()
    elif active is None:
        onboarding()
        history_block()
    elif active.get("status") == "error":
        notify("The strategy could not complete on this sample.",
               ["Shorten the lookback or pick a finer bar size.",
                "Widen the years so more breakouts exist.",
                "Lower the probability threshold if nothing is taken."],
               technical=active.get("error"))
        onboarding()
        history_block()
    elif active.get("status") == "stopped" or not active.get("metrics"):
        sec("R", "Run stopped")
        st.caption("Partial runs are not scored. Adjust the settings and run again.")
        history_block()
    else:
        sec("R", "Results")
        if not active.get("_finalized"):
            finalize(active)
        st.markdown(
            f'<div class="chips"><span class="chip hot">{TF_LABEL[active["config"]["tf"]]}</span>'
            f'<span class="chip">L={active["config"]["lookback"]} · refit {active["config"]["line_step"]}</span>'
            f'<span class="chip cool">{active.get("window_desc","")}</span>'
            f'<span class="chip">{active["status"].upper()}</span></div>', unsafe_allow_html=True)
        results_body(active)
        if active.get("signal") is not None:
            show_fig(position_fig(active["curves"]["x"], active["signal"]), key="lab_pos")
        history_block()

with tab_tl:
    sec("T", "Trendline geometry")
    st.caption("Live fit on the loaded feed — independent of any run. The window never includes the current bar.")
    tail = num_slider("Bars shown", 60, max(61, min(2000, n_bars)), min(420, n_bars), "w_tail")
    win = df[(df.index.year >= y0) & (df.index.year <= y1)]
    if len(win) < cfg_now["lookback"] + 5:
        st.caption("Widen the years or shorten the lookback to populate this panel.")
    else:
        ln = preview_lines(win, cfg_now["lookback"], cfg_now["line_step"])
        show_fig(price_lines_fig(win, ln, cfg_now["lookback"], tail), key="tl_price")
        show_fig(slope_fig(win, ln), key="tl_slope")
        st.caption("Resistance in red, support in green, both projected onto the current bar. "
                   "A close above resistance is the primary long breakout.")
        act = st.session_state.get("active")
        if act and act.get("trades") is not None and len(act["trades"]):
            tr = act["trades"]
            n_run = int((act.get("metrics") or {}).get("bars", -1))
            if n_run != len(win):
                st.caption("Re-run the strategy on the current window to inspect individual trades.")
            else:
                labels = {}
                for i in tr.index:
                    row = tr.loc[i]
                    entry_time = win.index[int(row["entry_i"])]
                    stamp = entry_time.strftime("%Y-%m-%d %H:%M")
                    prob = fmt_n(row["model_prob"], "{:.2f}")
                    labels[i] = f"#{i} · {stamp} · P={prob} · {row['exit_reason']}"
                k = st.selectbox("Inspect trade", list(tr.index), key="w_trade_i",
                                 format_func=lambda i: labels.get(i, str(i)))
                show_fig(trade_fig(win, tr.loc[k], act["config"]["lookback"]), key="tl_trade")

with tab_model:
    sec("M", "Meta-label diagnostics")
    act = st.session_state.get("active")
    if not act or act.get("metrics") is None:
        st.caption("Run the strategy in the Breakout Lab to populate this panel.")
    else:
        cm = act["confusion"]
        c1, c2 = st.columns([1.2, 1])
        with c1:
            show_fig(imp_fig(act["importances"]), key="md_imp")
        with c2:
            if cm:
                st.markdown(cards_html([
                    dict(k="Precision", v=fmt_n(cm["precision"], "{:.1%}"), cls="cyan",
                         d="of taken trades, share profitable"),
                    dict(k="Recall", v=fmt_n(cm["recall"], "{:.1%}"),
                         d="of winners, share captured"),
                    dict(k="F1", v=fmt_n(cm["f1"], "{:.3f}"), cls="cyan"),
                    dict(k="Accuracy", v=fmt_n(cm["accuracy"], "{:.1%}"),
                         d=f"{cm['n']} scored trades"),
                ]), unsafe_allow_html=True)
        show_fig(prob_fig(act["trades"], act["config"]["threshold"]), key="md_prob")
        cc1, cc2 = st.columns([1.3, 1])
        with cc1:
            show_fig(sweep_fig(act["sweep"], act["config"]["threshold"]), key="md_sweep")
        with cc2:
            if cm:
                show_fig(confusion_fig(cm), key="md_cm")
        show_fig(scatter_fig(act["trades"]), key="md_scatter")
        st.caption("Sweep and confusion use trade-level returns on out-of-sample trades only. "
                   "Raising the threshold trades quantity for quality — watch both curves.")
        with st.expander("Training log"):
            st.table(pd.DataFrame(act["train_log"]).rename(
                columns={"bar": "bar index", "cases": "training trades", "win_rate": "base rate"}))

with tab_trades:
    sec("B", "Trade blotter")
    act = st.session_state.get("active")
    if not act or act.get("trades") is None:
        st.caption("Run the strategy in the Breakout Lab to populate this panel.")
    else:
        tr = act["trades"].copy()
        win = st.session_state.get("win", df)
        tr["entry_time"] = [win.index[int(i)] for i in tr["entry_i"]]
        tr["exit_time"] = [win.index[int(i)] for i in tr["exit_i"]]
        tr["taken"] = tr["model_prob"] > act["config"]["threshold"]
        show = tr[["entry_time", "exit_time", "exit_reason", "bars_held", "return",
                   "model_prob", "taken", "resist_s", "tl_err", "max_dist", "vol", "adx"]]
        only = st.toggle("Taken trades only", value=False, key="w_only_taken")
        st.dataframe(show[show["taken"]] if only else show, use_container_width=True, height=420)
        d1, d2, d3 = st.columns(3)
        d1.download_button("⬇ Trades (.csv)", show.to_csv(index=False).encode(),
                           file_name=f"trades_{act['id']}.csv", mime="text/csv", key="dl_trades")
        payload = dict(id=act["id"], config=act["config"], metrics=act["metrics"],
                       importances=act["importances"], confusion=act["confusion"],
                       window=act.get("window_desc"), built_by=AUTHOR,
                       method="neurotrader888/TrendlineBreakoutMetaLabel (MIT)")
        d2.download_button("⬇ Results (.json)", json.dumps(payload, default=json_safe, indent=2),
                           file_name=f"results_{act['id']}.json", mime="application/json", key="dl_json")
        d3.button("↻ Re-run current settings", on_click=start_run, key="rerun_btn")

with tab_method:
    sec("M", "The method, end to end")
    st.markdown("""
<div class="strat-block">
  <div class="hd"><span class="nm">1 · Trendline automation</span><span class="tagline">GEOMETRY · PIVOT-ANCHORED FIT</span></div>
  <ol>
    <li>Fit a least-squares line through the window of <b>log closes</b> — logs make slopes comparable across price levels.</li>
    <li>Take the pivot: the bar furthest <b>above</b> the line for resistance, furthest <b>below</b> for support.</li>
    <li>Rotate the line around that pivot by numerical gradient descent with <b>step halving</b>, minimising squared
        distance under a hard constraint: the line may never cut through price.</li>
    <li>The window <b>excludes the current bar</b>; the fitted line is projected one step forward and compared with
        today's close. No look-ahead.</li>
  </ol>
  <div class="ask">Refit-every-N reuses a fit and projects it forward for N bars — a speed lever, not a different method.</div>
</div>

<div class="strat-block">
  <div class="hd"><span class="nm">2 · Primary signal &amp; triple barrier</span><span class="tagline">LONG BREAKOUTS · ATR SIZING</span></div>
  <ol>
    <li>Entry when the close breaks the projected <b>resistance</b>. One position at a time.</li>
    <li>Barriers in log-ATR units: <b>TP = entry + k·ATR</b>, <b>SL = entry − k·ATR</b>, plus a <b>hold period</b>.</li>
    <li>ATR is Wilder-smoothed true range on <b>log</b> high/low/close, so barriers scale with recent volatility.</li>
    <li>The trade's outcome — profit or not — becomes the <b>meta-label</b>. The primary signal is never optimised.</li>
  </ol>
  <div class="ask">Meta-labeling separates "where is the signal" from "should I take it" — the model only sizes the bet.</div>
</div>

<div class="strat-block">
  <div class="hd"><span class="nm">3 · Features &amp; walk-forward forest</span><span class="tagline">5 FEATURES · LEAKAGE-FREE</span></div>
  <ol>
    <li><b>resist_s</b> — resistance slope ÷ ATR: how steep the line is relative to noise.</li>
    <li><b>tl_err</b> — mean signed distance from the line ÷ ATR: how tightly price respects it.</li>
    <li><b>max_dist</b> — worst distance from the line ÷ ATR: how much the line was stretched.</li>
    <li><b>vol</b> — volume ÷ rolling median: participation on the breakout bar.</li>
    <li><b>adx</b> — Wilder ADX at the lookback: is this a trending regime at all?</li>
    <li>Every <b>step</b> bars, re-fit <code>RandomForestClassifier</code> on trades that <b>opened and closed</b> inside
        the trailing training window. Predict <code>P(profit)</code> at each new breakout; take it above the threshold.</li>
  </ol>
  <div class="ask">Skipped breakouts still occupy the slot until their barrier resolves — exactly as in the source.</div>
</div>

<div class="pd-grid">
  <div class="pd keep"><div class="h4">✓ What this controls</div><ul>
    <li><b>Selection bias</b> — the primary rule is fixed, never swept</li>
    <li><b>Look-ahead</b> — window excludes the current bar; training uses closed trades only</li>
    <li><b>Regime drift</b> — the forest is re-fitted on a rolling window</li>
    <li><b>Volatility scaling</b> — every barrier and feature is ATR-normalised</li>
  </ul></div>
  <div class="pd kill"><div class="h4">✕ What it does not</div><ul>
    <li><b>Costs</b> — gross of fees, slippage and funding</li>
    <li><b>Overlapping labels</b> — trades are sequential, but regimes repeat</li>
    <li><b>Small samples</b> — few breakouts means a noisy forest</li>
    <li><b>Shorts</b> — the meta-label side is long-only, as in the source</li>
  </ul></div>
</div>
""", unsafe_allow_html=True)

    sec("D", "Data — format and a reference pair to try")
    st.markdown("""
<div class="panel">
<div class="h3">File requirements</div>
<ul>
  <li><b>Format.</b> <code>.csv</code>, <code>.pq</code>, <code>.parquet</code> — <b>max 200 MB per file</b>.</li>
  <li><b>Columns.</b> timestamp first, then <code>open, high, low, close</code>; <code>volume</code> optional but
      recommended — without it the volume feature is constant and the model leans on the other four.</li>
  <li><b>Timestamps.</b> one timezone throughout (UTC safest); rows are sorted and de-duplicated for you.</li>
  <li><b>Values.</b> strictly positive prices; incomplete rows dropped. At least 12 usable rows.</li>
</ul>
<div class="h4">Example — first three rows</div>
</div>
""", unsafe_allow_html=True)
    st.code("timestamp,open,high,low,close,volume\n"
            "2016-01-01 00:00:00,430.00,431.82,428.15,430.44,1184.2\n"
            "2016-01-01 01:00:00,430.44,433.10,429.90,432.02,1620.7\n"
            "2016-01-01 02:00:00,432.02,432.55,427.40,428.11,2044.9", language="text")
    st.markdown(f"""
<div class="callout cool"><b>Reference pair — download and try.</b> The app's default seeded feed exported at full
precision: BTC-like and ETH-like hourly OHLCV, 2016-01-01 → 2020-12-31, 43,848 bars each (≈5 MB per file, far below the
max 200 MB cap). <b>Synthetic</b> — GARCH volatility, t(4) tails, trend regimes, jumps, volume reacting to absolute
returns — generated with seed {REF_SEED}. <b>Not exchange history.</b> Uploading it reproduces the default run exactly.</div>
""", unsafe_allow_html=True)
    r1, r2, r3, r4 = st.columns(4)
    r1.download_button("⬇ BTC reference (.csv)", reference_csv("BTC"), file_name="BTCUSD_reference_hourly.csv",
                       mime="text/csv", key="dl_btc")
    r2.download_button("⬇ ETH reference (.csv)", reference_csv("ETH"), file_name="ETHUSD_reference_hourly.csv",
                       mime="text/csv", key="dl_eth")
    r3.download_button("⬇ Both + README (.zip)", reference_zip(), file_name="reference_pair_hourly.zip",
                       mime="application/zip", key="dl_zip_ref")
    r4.button("▶ Use reference pair now", on_click=use_reference_pair, key="use_ref")
    st.markdown("""
<div class="panel">
<div class="h3">Loading real exchange history</div>
<ul>
  <li><b>Binance public data</b> — <code>data.binance.vision</code>, klines as daily/monthly ZIPs per symbol and interval.</li>
  <li><b>CryptoDataDownload</b> — free hourly and daily BTCUSD / ETHUSD CSVs.</li>
  <li><b>Kaggle</b> — long-running minute-level Bitcoin datasets and hourly Binance mirrors.</li>
  <li><b>The source repo</b> ships <code>BTCUSDT3600.csv</code>, the exact feed the original scripts use.</li>
</ul>
<p>Normalise to the schema above before uploading. Binance klines carry <code>open_time</code> in milliseconds plus
extra columns — convert, rename, keep the five price/volume columns.</p>
</div>
""", unsafe_allow_html=True)

    sec("Q", "Quick start")
    st.markdown("""
<div class="panel">
<ol>
  <li><b>Load data.</b> Reference pair, or your own OHLCV file.</li>
  <li><b>Bar size.</b> Start on <b>Daily</b> — the fit is exact (refit every 1) and fast. Hourly works too; raise
      "refit every" to 4–8 if you want it snappier.</li>
  <li><b>Run.</b> Defaults follow the source: lookback 30 daily / 72 hourly, 3×ATR barriers, forest depth 3.</li>
  <li><b>Read the verdict.</b> Compare meta-labeled PF against <i>every breakout</i> — that difference is the model's
      contribution. Buy & hold is the third benchmark.</li>
  <li><b>Inspect.</b> Trendlines tab for the geometry and a single trade; Model tab for importances, the probability
      split and the threshold sweep; Trades tab for the blotter and exports.</li>
  <li><b>Stress it.</b> Change one thing — threshold, hold period, barrier multiples — and re-run. A filter that only
      works at one setting is not a filter.</li>
</ol>
</div>
""", unsafe_allow_html=True)

    sec("S", "Recommended settings by bar size")
    st.table(pd.DataFrame({
        "Lookback": [72, 30, 12, 3],
        "Refit every": ["4 bars", "1 bar", "1 bar", "1 bar"],
        "Hold period": [12, 5, 3, 1],
        "ATR lookback": [168, 60, 24, 3],
        "Train / step": ["17,520 / 8,760", "365 / 90", "36 / 6", "5 / 1"],
        "Use": ["source setting", "recommended", "regime questions", "illustration only"],
    }, index=["Hourly", "Daily", "Monthly", "Yearly"]))
    st.caption("Defaults load when you move the Timeframe slider; every window auto-fits the bars you actually have.")

    sec("T", "Troubleshooting")
    st.table(pd.DataFrame({
        "Likely cause": [
            "Lookback too long for the sample, or a quiet range with no breaks",
            "ATR / ADX warm-up plus the training window consumed the data",
            "Model is conservative, or too few training trades",
            "No volume column in the file",
            "Hourly feed with refit-every-1",
            "Coarse bar size leaves a handful of trades",
        ],
        "What to do": [
            "Shorten the lookback, widen the years, or pick a finer bar size",
            "Shorten the ATR lookback or the training window; widen the years",
            "Lower the threshold, raise trees, or cap depth at 2–3",
            "Add a volume column — otherwise that feature is constant",
            "Raise 'refit every' to 4–8, or switch to Daily",
            "Treat results as indicative; report the trade count alongside PF",
        ],
    }, index=["No breakouts produced", "Too few bars at this size", "Nothing taken",
              "Volume feature flat", "Run feels slow", "Tiny trade count"]))
    st.caption("Any of these raises one dismissible popup — the page itself stays clean.")

    sec("F", "FAQ")
    st.markdown("""
<div class="panel">
<div class="h4">Why log prices?</div>
<p>Slopes and distances become scale-free, so a line fitted at $400 and one at $40,000 are comparable — and ATR
barriers become proportional rather than absolute.</p>
<div class="h4">Why is the model only allowed to say no?</div>
<p>That is meta-labeling. The primary rule decides <i>where</i> a trade exists; the model only decides <i>whether</i> to
take it. It cannot invent trades, so it cannot manufacture a backtest out of nothing.</p>
<div class="h4">Is the threshold a fitted parameter?</div>
<p>Treat it as one. The sweep panel shows how sensitive the result is; a threshold chosen after seeing the sweep is
selection bias, and should be validated on data the choice never saw.</p>
<div class="h4">Why do skipped breakouts still block new entries?</div>
<p>To match the source exactly: the slot stays occupied until the barrier resolves. It keeps the comparison between
filtered and unfiltered honest — both see the same opportunity set.</p>
<div class="h4">Are the seeds fixed?</div>
<p>Yes — the forest uses 69420 (as in the source) and the reference feed uses 7. Same inputs, same result.</p>
<div class="h4">Does a higher profit factor prove the filter works?</div>
<p>No. Compare win rate, average trade and time in market too, and check that the improvement survives a neighbouring
threshold and bar size. Everything here is gross of costs.</p>
</div>
""", unsafe_allow_html=True)

    sec("C", "Complete source")
    FILES = ["app.py", "worker.py", "meta_model.py", "dataset.py", "trendlines.py",
             "market_data.py", "requirements.txt", "README.md"]

    @st.cache_data(show_spinner=False)
    def project_zip():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in FILES:
                p = Path(f)
                if p.exists():
                    z.write(p, f)
        return buf.getvalue()

    pick = st.selectbox("Source file", FILES, key="w_file")
    p = Path(pick)
    if p.exists():
        st.code(p.read_text(encoding="utf-8"),
                language="python" if pick.endswith(".py") else "text", line_numbers=True)
        d1, d2 = st.columns(2)
        d1.download_button(f"⬇ {pick}", p.read_text(encoding="utf-8"), file_name=pick, key="dl_file")
        d2.download_button("⬇ Full project (.zip)", project_zip(),
                           file_name="trendline_breakout_metalabel.zip",
                           mime="application/zip", key="dl_proj")
    else:
        st.caption("That file is not present in this deployment.")
    st.markdown('<div class="callout"><b>Run it:</b> <code>pip install -r requirements.txt</code> → '
                '<code>streamlit run app.py</code>. Method after '
                '<code>neurotrader888/TrendlineBreakoutMetaLabel</code> (MIT).</div>', unsafe_allow_html=True)

st.markdown(f'<div class="foot">TRENDLINE BREAKOUT · META-LABEL ML · BUILT BY <b>{AUTHOR.upper()}</b> · '
            'GEOMETRY FINDS THE TRADE — THE MODEL DECIDES</div>', unsafe_allow_html=True)

flush_popup()
