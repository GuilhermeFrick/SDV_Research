"""
Dashboard — SOME/IP IDS Pipeline
Streamlit multi-página para análise, mistura de datasets e avaliação do multiclasse.

Uso:
    streamlit run detection/dashboard/app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent   # SDV_Research/
sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="SOME/IP IDS Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS extra ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  section[data-testid="stSidebar"] { min-width: 240px; }
  .metric-label { font-size: 12px !important; }
  .block-container { padding-top: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ SOME/IP IDS")
    st.caption("Pipeline de detecção de intrusão")
    st.divider()

    page = st.radio(
        "Módulo",
        options=[
            "📊  Visão Geral",
            "🔀  Mixer de Datasets",
            "⚙️  Pipeline",
            "📈  Resultados",
            "🔍  Inferência",
        ],
        label_visibility="collapsed",
    )

# ── Roteamento ─────────────────────────────────────────────────────────────────
if   "Visão"      in page: from pages import p1_overview  as pg
elif "Mixer"      in page: from pages import p2_mixer     as pg
elif "Pipeline"   in page: from pages import p3_pipeline  as pg
elif "Resultados" in page: from pages import p4_results   as pg
elif "Inferência" in page: from pages import p5_inference as pg

pg.render(ROOT)
