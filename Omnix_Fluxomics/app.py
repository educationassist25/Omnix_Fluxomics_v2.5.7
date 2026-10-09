"""
Omnix™ Fluxomics — isotope-tracing analysis for unbiased LC-MS metabolomics.
Run:  streamlit run app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(page_title="Omnix Fluxomics", page_icon="🧪", layout="wide", initial_sidebar_state="expanded")

from core import __version__  # noqa: E402
from ui import common as C  # noqa: E402
from ui import (step1_upload, step2_design, step4_normalize, step5_incorporation,  # noqa: E402
                step6_boxplots, step7_stats, step8_heatmap, step9_pathways, step11_kinetics,
                step13_export)

CSS = """
<style>
.block-container{padding-top:3.4rem;max-width:1400px}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]{gap:0.1rem}
h1,h2,h3{letter-spacing:-0.01em}
.app-title{font-size:1.55rem;font-weight:700;margin:0;color:#000}
.app-sub{color:#000;font-size:0.92rem;margin:0 0 0.4rem}
.step-head{border-bottom:1px solid #e6e5ee;padding-bottom:0.5rem;margin-bottom:1rem}
.step-kicker{font-size:0.78rem;letter-spacing:.08em;text-transform:uppercase;color:#000;font-weight:700}
.step-title{font-size:1.45rem;font-weight:700;margin:0.1rem 0}
.step-desc{color:#000;max-width:72ch;margin:0}
.direction{background:#f4f2fc;border:1px solid #dcd6f5;border-radius:8px;padding:8px 12px;margin:6px 0 12px;font-size:0.92rem}
.direction .pos{color:#000;font-weight:700}
.note{background:#fff8e6;border-left:3px solid #e0a400;padding:8px 12px;border-radius:4px;font-size:0.9rem;margin:6px 0}
.okbox{background:#effaf3;border-left:3px solid #1f9d55;padding:8px 12px;border-radius:4px;font-size:0.9rem;margin:6px 0}
.ds-card{background:#ffffff;border:1px solid #e6e5ee;border-radius:8px;padding:8px 10px;font-size:0.82rem;color:#000;margin-bottom:12px}
.st-key-intbox{background:#f1eefc !important;border:1.5px solid #6f5fc7 !important;border-radius:8px;padding:8px 10px;margin:4px 0 12px}
.st-key-intbox label p{font-weight:600}
.int-head{font-size:0.8rem;font-weight:700;color:#000;line-height:1.3;padding-bottom:12px;display:block}
[data-testid="stMetricValue"]{font-variant-numeric:tabular-nums}
section[data-testid="stSidebar"] .stButton button{justify-content:flex-start;border:none;background:transparent;
  text-align:left;padding:0.32rem 0.6rem;width:100%;font-size:0.93rem}
section[data-testid="stSidebar"] .stButton button:hover{background:#ebe8f7}
/* v2.4.2: all text solid black (captions, labels, help text, metrics, tabs, expanders, notes) */
.stApp, .stApp p, .stApp li, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5,
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *, .stCaption, small,
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *, [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *,
[data-testid="stMetricValue"], [data-testid="stExpander"] summary *, [data-baseweb="tab"] p, .note, .okbox, .direction,
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] *:not(code),
[data-testid="stTooltipContent"], [data-baseweb="select"] *, input, textarea {color:#000000 !important; opacity:1 !important}
.stButton button[kind="primary"] *, .stDownloadButton button[kind="primary"] *, .stFormSubmitButton button[kind="primary"] * {color:#ffffff !important}
[data-testid="stAlertContentError"] *, [data-testid="stAlertContentWarning"] *, [data-testid="stAlertContentInfo"] *,
[data-testid="stAlertContentSuccess"] * {color:#000000 !important}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
C.init_state()

PAGES = {"upload": step1_upload, "design": step2_design, "normalize": step4_normalize,
         "incorporation": step5_incorporation, "boxplots": step6_boxplots, "stats": step7_stats,
         "pathways": step9_pathways, "heatmap": step8_heatmap,
         "kinetics": step11_kinetics, "export": step13_export}

with st.sidebar:
    st.markdown("<p class='app-title'>Omnix™ Fluxomics</p><p class='app-sub'>Stable-isotope metabolomics & flux "
                "analysis</p>", unsafe_allow_html=True)
    for key, label in C.STEPS:
        mark = "✓ " if C.done(key) else "○ "
        current = st.session_state.step == key
        if st.button(("▸ " if current else "") + mark + label, key=f"nav_{key}", type="secondary"):
            st.session_state.step = key
            st.rerun()
    st.divider()
    ds = st.session_state.ds
    if ds is not None:
        n_lab = int((ds.S.SampleType == "13C labeled").sum())
        st.markdown(f"<div class='ds-card'><b>Dataset</b><br>{'Compound Discoverer' if ds.source == 'cd' else 'Generic table'}"
                    f"<br>{ds.CT.Compound.nunique()} metabolites · {len(ds.CT)} isotopologue channels<br>"
                    f"{len(ds.S)} samples ({n_lab} labeled)</div>", unsafe_allow_html=True)
    if st.session_state.ds is not None:
        try:
            box = st.container(border=True, key="intbox")       # Streamlit ≥ 1.39: keyed, styled via CSS
        except TypeError:
            box = st.container()
        with box:
            st.markdown("<div class='int-head'>Intensity dataset for downstream steps</div>",
                        unsafe_allow_html=True)
            st.radio("Intensity dataset for downstream steps", ["log2", "log2_iqr"], key="intensity_choice",
                     format_func={"log2_iqr": "Log2 + IQR-normalized intensity", "log2": "Log2 intensity"}.get,
                     label_visibility="collapsed")
    with st.expander("Session"):
        if st.button("Reset all steps", key="reset"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
        st.caption(f"Version {__version__}")

page = PAGES[st.session_state.step]
page.render()
