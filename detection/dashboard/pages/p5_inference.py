"""Página 5 — Inferência: classifica pacotes de um CSV."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

CLASS_NAMES = ["Benigno", "DoS", "Fuzzy", "MITM Multi", "MITM Single", "FakeClientID"]
CLASS_COLOR = {
    "Benigno":     "#2ca02c", "DoS":        "#d62728",
    "Fuzzy":       "#ff7f0e", "MITM Multi":  "#9467bd",
    "MITM Single": "#8c564b", "FakeClientID":"#1f77b4",
}

FEAT_COLS = [
    "f01_ip_time_interval", "f08_someip_payload_change",
    "f11_ip_length_change",  "f12_tcpudp_length_change",
    "f13_payload_repeat_rate","f15_someip_payload_len",
    "f16_tcpudp_len",         "f17_src_packet_rate",
    "f18_src_payload_diversity","f19_is_sd",
    "f20_src_service_diversity","f21_is_relay_service",
    "f22_src_clientid_diversity",
]


@st.cache_resource
def _load_model(model_path: Path):
    import xgboost as xgb
    m = xgb.XGBClassifier()
    m.load_model(str(model_path))
    return m


def _normalize(X: np.ndarray, norm_params: dict) -> np.ndarray:
    X = X.copy().astype(np.float32)
    for j, col in enumerate(FEAT_COLS):
        if col in norm_params:
            lo = norm_params[col]["min"]
            hi = norm_params[col]["max"]
            d  = hi - lo
            X[:, j] = np.clip((X[:, j] - lo) / d, 0, 1) if d > 0 else 0.0
    return X


def render(root: Path):
    st.title("🔍 Inferência por pacote")
    st.caption("Faça upload de um CSV com features extraídas e veja a classificação "
               "do modelo multiclasse para cada pacote.")

    model_dir = root / "detection" / "multiclass" / "model"
    model_json  = model_dir / "multiclass_classifier.json"
    norm_json   = model_dir / "norm_params.json"

    if not model_json.exists():
        st.error("Modelo não encontrado. Execute o treino na página ⚙️ Pipeline.")
        return

    with open(norm_json) as f:
        norm_params = json.load(f)

    # ── Opção: usar features.csv existente ou upload ───────────────────────────
    source = st.radio(
        "Fonte dos dados",
        ["📂 features.csv existente (multiclasse)",
         "📤 Upload de CSV"],
        horizontal=True,
    )

    df_input = None

    if "features.csv" in source:
        feat_csv = root / "detection" / "multiclass" / "data" / "features.csv"
        if feat_csv.exists():
            n_sample = st.slider("Amostras para classificar", 1000, 100_000, 10_000, 1000)
            with st.spinner("Carregando..."):
                df_input = pd.read_csv(feat_csv, nrows=n_sample * 5)
                if len(df_input) > n_sample:
                    df_input = df_input.sample(n_sample, random_state=42)
            st.success(f"{len(df_input):,} amostras carregadas.")
        else:
            st.warning("features.csv não encontrado.")

    else:
        uploaded = st.file_uploader(
            "CSV com colunas de features (f01…f22)",
            type=["csv"],
        )
        if uploaded:
            df_input = pd.read_csv(uploaded)
            st.success(f"{len(df_input):,} linhas carregadas.")

    if df_input is None:
        return

    # ── Verificar colunas ──────────────────────────────────────────────────────
    missing = [c for c in FEAT_COLS if c not in df_input.columns]
    if missing:
        # Adicionar f22 com default=1.0 se ausente (dados Kim sem FakeClientID)
        if missing == ["f22_src_clientid_diversity"]:
            df_input["f22_src_clientid_diversity"] = 1.0
            st.info("f22_src_clientid_diversity não encontrada — usando 1.0 (padrão Kim).")
        else:
            st.error(f"Colunas ausentes: {missing}")
            return

    # ── Inferência ─────────────────────────────────────────────────────────────
    if st.button("▶ Classificar pacotes", type="primary"):
        with st.spinner("Classificando..."):
            model = _load_model(model_json)
            X = df_input[FEAT_COLS].fillna(0).values
            X_norm = _normalize(X, norm_params)
            probs  = model.predict_proba(X_norm)
            preds  = probs.argmax(axis=1)
            conf   = probs.max(axis=1)

        df_res = df_input.copy()
        df_res["classe_pred"] = [CLASS_NAMES[p] for p in preds]
        df_res["confiança"]   = conf.round(4)
        if "label" in df_res.columns:
            df_res["classe_real"] = df_res["label"].map(
                {i: n for i, n in enumerate(CLASS_NAMES)}).fillna("Desconhecida")
            df_res["acerto"] = df_res["classe_pred"] == df_res["classe_real"]

        # ── Distribuição predita ───────────────────────────────────────────────
        st.subheader("Distribuição das predições")
        pred_counts = df_res["classe_pred"].value_counts().reset_index()
        pred_counts.columns = ["Classe", "Pacotes"]

        col1, col2 = st.columns([1, 1])
        with col1:
            fig_pie = px.pie(pred_counts, names="Classe", values="Pacotes",
                             color="Classe", color_discrete_map=CLASS_COLOR, hole=0.35)
            fig_pie.update_layout(height=300, margin=dict(t=10, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.dataframe(pred_counts, use_container_width=True, hide_index=True)
            st.metric("Ataques detectados",
                      f"{(df_res['classe_pred'] != 'Benigno').sum():,}",
                      f"{(df_res['classe_pred'] != 'Benigno').mean()*100:.1f}% do total")

        # ── Confiança por classe ───────────────────────────────────────────────
        st.subheader("Distribuição de confiança por classe predita")
        fig_box = px.box(
            df_res, x="classe_pred", y="confiança",
            color="classe_pred", color_discrete_map=CLASS_COLOR,
            points="outliers",
        )
        fig_box.update_layout(showlegend=False, height=320,
                              xaxis_title="Classe predita",
                              margin=dict(t=10, b=10))
        st.plotly_chart(fig_box, use_container_width=True)

        # ── Timeline se houver timestamp ───────────────────────────────────────
        if "timestamp" in df_res.columns:
            st.subheader("Linha do tempo das predições")
            df_ts = df_res[["timestamp", "classe_pred", "confiança"]].sort_values("timestamp")

            fig_tl = px.scatter(
                df_ts.sample(min(5000, len(df_ts))),
                x="timestamp", y="confiança",
                color="classe_pred", color_discrete_map=CLASS_COLOR,
                opacity=0.6, size_max=4,
                labels={"timestamp": "Timestamp (s)", "confiança": "Confiança"},
            )
            fig_tl.update_layout(height=320, margin=dict(t=10, b=30))
            st.plotly_chart(fig_tl, use_container_width=True)

        # ── Tabela de resultados ───────────────────────────────────────────────
        st.subheader("Amostra dos resultados")
        cols_show = ["classe_pred", "confiança"]
        if "classe_real" in df_res.columns:
            cols_show = ["classe_real", "classe_pred", "confiança", "acerto"]
        st.dataframe(
            df_res[cols_show].head(200),
            use_container_width=True, hide_index=True, height=300,
        )

        # Download
        csv_out = df_res[cols_show].to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Baixar resultados (.csv)", csv_out,
                           "predicoes.csv", "text/csv")
