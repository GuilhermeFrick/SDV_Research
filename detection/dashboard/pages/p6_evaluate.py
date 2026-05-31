"""Página 6 — Avaliação por Arquivo: seleciona quais arquivos testar no classificador."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import xgboost as xgb

CLASS_NAMES = ["Benigno", "DoS", "Fuzzy", "MITM_Multi", "MITM_Single", "FakeClientID"]
CLASS_COLOR = {
    "Benigno":    "#2ca02c", "DoS":       "#d62728",
    "Fuzzy":      "#ff7f0e", "MITM_Multi": "#9467bd",
    "MITM_Single":"#8c564b", "FakeClientID":"#1f77b4",
}

# Mapeamento: arquivo origem → label no features.csv
FILE_LABEL = {
    "benign_traffic.csv":             (0, "Benigno"),
    "dos_noti_flood.csv":             (1, "DoS"),
    "fuzzy_sd_offer_rand_noti1.csv":  (2, "Fuzzy"),
    "fuzzy_sd_offer_rand_noti2.csv":  (2, "Fuzzy"),
    "fuzzy_sd_offer_rand_noti3.csv":  (2, "Fuzzy"),
    "mitm_multi_attacker.csv":        (3, "MITM_Multi"),
    "mitm_single_attacker.csv":       (4, "MITM_Single"),
}

FEAT_COLS = [
    "f01_ip_time_interval",     "f08_someip_payload_change",
    "f11_ip_length_change",     "f12_tcpudp_length_change",
    "f13_payload_repeat_rate",  "f15_someip_payload_len",
    "f16_tcpudp_len",           "f17_src_packet_rate",
    "f18_src_payload_diversity","f19_is_sd",
    "f20_src_service_diversity","f21_is_relay_service",
    "f22_src_clientid_diversity",
]
CHUNK = 500_000


@st.cache_resource
def _load_model(path: str):
    m = xgb.XGBClassifier()
    m.load_model(path)
    return m


@st.cache_data(show_spinner=False)
def _load_features(feat_csv: str, labels: tuple[int, ...], n_max: int) -> pd.DataFrame:
    """Lê features.csv filtrando pelos labels selecionados."""
    parts = []
    for chunk in pd.read_csv(feat_csv, chunksize=CHUNK):
        sub = chunk[chunk["label"].isin(labels)]
        if len(sub):
            parts.append(sub)
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if n_max and len(df) > n_max:
        df = df.groupby("label", group_keys=False).apply(
            lambda g: g.sample(min(len(g), int(n_max * len(g) / len(df))), random_state=42)
        ).reset_index(drop=True)
    return df


def _normalize(X: np.ndarray, norm: dict) -> np.ndarray:
    X = X.copy().astype(np.float32)
    for j, col in enumerate(FEAT_COLS):
        p = norm.get(col, {})
        lo, hi = p.get("min", 0), p.get("max", 1)
        d = hi - lo
        X[:, j] = np.clip((X[:, j] - lo) / d, 0.0, 1.0) if d > 0 else 0.0
    return X


def render(root: Path):
    st.title("🎯 Avaliação por Arquivo")
    st.caption(
        "Selecione quais arquivos de origem deseja testar no classificador multiclasse. "
        "O modelo classifica cada pacote e mostra quantos foram detectados corretamente."
    )

    det       = root / "detection"
    model_dir = det / "multiclass" / "model"
    feat_csv  = det / "multiclass" / "data" / "features.csv"
    model_json= model_dir / "multiclass_classifier.json"
    norm_json = model_dir / "norm_params.json"

    # ── Verificar pré-requisitos ───────────────────────────────────────────────
    missing = []
    if not feat_csv.exists():   missing.append("`features.csv`")
    if not model_json.exists(): missing.append("`multiclass_classifier.json`")
    if not norm_json.exists():  missing.append("`norm_params.json`")
    if missing:
        st.error("Arquivos necessários não encontrados: " + ", ".join(missing))
        st.info("Execute o pipeline na página ⚙️ Pipeline primeiro.")
        return

    # ── Seleção de arquivos ────────────────────────────────────────────────────
    col_sel, col_res = st.columns([1, 2])

    with col_sel:
        st.subheader("📂 Selecione os arquivos")

        sel_ben    = st.checkbox("benign_traffic.csv",            value=True,  key="ben")
        sel_dos    = st.checkbox("dos_noti_flood.csv",            value=False, key="dos")
        sel_fuz    = st.checkbox("fuzzy_sd_offer_rand_noti*.csv", value=False, key="fuz")
        sel_mitm_m = st.checkbox("mitm_multi_attacker.csv",       value=False, key="mm")
        sel_mitm_s = st.checkbox("mitm_single_attacker.csv",      value=False, key="ms")

        # Labels selecionados
        selected_labels = set()
        if sel_ben:    selected_labels.add(0)
        if sel_dos:    selected_labels.add(1)
        if sel_fuz:    selected_labels.add(2)
        if sel_mitm_m: selected_labels.add(3)
        if sel_mitm_s: selected_labels.add(4)

        st.divider()
        n_max = st.number_input(
            "Limite de amostras (0 = todos)",
            min_value=0, max_value=5_000_000,
            value=0, step=100_000,
            help="Limitar acelera a avaliação. 0 = usa tudo.",
        )

        run = st.button("▶ Avaliar", type="primary",
                        disabled=len(selected_labels) == 0)

    # ── Resultado ──────────────────────────────────────────────────────────────
    with col_res:
        if not run:
            st.info("Selecione ao menos um arquivo e clique em **▶ Avaliar**.")
            return

        if not selected_labels:
            st.warning("Selecione pelo menos um arquivo.")
            return

        with st.spinner("Carregando features..."):
            df = _load_features(str(feat_csv), tuple(sorted(selected_labels)),
                                int(n_max))

        if df.empty:
            st.error("Nenhuma amostra encontrada para os arquivos selecionados.")
            return

        # Adicionar f22=1.0 se ausente
        if "f22_src_clientid_diversity" not in df.columns:
            df["f22_src_clientid_diversity"] = 1.0

        st.markdown(f"**{len(df):,} amostras** carregadas")

        # Distribuição real
        dist = df["label"].value_counts().sort_index()
        st.markdown("Distribuição real dos arquivos selecionados:")
        d_cols = st.columns(len(dist))
        for col, (lbl, cnt) in zip(d_cols, dist.items()):
            col.metric(CLASS_NAMES[lbl], f"{cnt:,}", f"{cnt/len(df)*100:.1f}%")

        # Classificar
        with st.spinner("Classificando..."):
            model = _load_model(str(model_json))
            with open(norm_json) as f:
                norm = json.load(f)

            X      = df[FEAT_COLS].fillna(0).values
            x_norm = _normalize(X, norm)
            y_true = df["label"].values.astype(int)
            y_pred = model.predict(x_norm)

        st.divider()

        # ── Taxa de detecção por classe ────────────────────────────────────────
        st.subheader("Taxa de detecção por classe")

        detection_rows = []
        for lbl in sorted(selected_labels):
            mask   = y_true == lbl
            n_real = mask.sum()
            if n_real == 0:
                continue
            n_correct = ((y_true == lbl) & (y_pred == lbl)).sum()
            n_miss    = n_real - n_correct
            recall    = n_correct / n_real
            detection_rows.append({
                "Classe":         CLASS_NAMES[lbl],
                "Total real":     int(n_real),
                "Detectados":     int(n_correct),
                "Não detectados": int(n_miss),
                "Taxa detecção":  recall,
            })

        df_det = pd.DataFrame(detection_rows)

        # Gráfico de barras — detectado vs não detectado
        fig_det = go.Figure()
        fig_det.add_trace(go.Bar(
            name="Detectados",
            x=df_det["Classe"], y=df_det["Detectados"],
            marker_color=[CLASS_COLOR.get(c, "#888") for c in df_det["Classe"]],
            text=df_det["Taxa detecção"].map(lambda x: f"{x:.1%}"),
            textposition="outside",
        ))
        fig_det.add_trace(go.Bar(
            name="Não detectados",
            x=df_det["Classe"], y=df_det["Não detectados"],
            marker_color="#cccccc", opacity=0.6,
        ))
        fig_det.update_layout(
            barmode="stack", height=340,
            yaxis_title="Nº de pacotes",
            margin={"t": 30, "b": 10},
            legend={"orientation": "h", "y": -0.25},
        )
        st.plotly_chart(fig_det, use_container_width=True)

        # Tabela
        df_show = df_det.copy()
        df_show["Taxa detecção"] = df_show["Taxa detecção"].map("{:.2%}".format)
        df_show["Total real"]    = df_show["Total real"].map("{:,}".format)
        df_show["Detectados"]    = df_show["Detectados"].map("{:,}".format)
        df_show["Não detectados"]= df_show["Não detectados"].map("{:,}".format)
        st.dataframe(df_show, use_container_width=True, hide_index=True)

        # ── O que o modelo previu para cada classe ─────────────────────────────
        st.divider()
        st.subheader("Como o modelo classificou cada classe")
        st.caption("Mostra para onde foram os pacotes que não foram detectados corretamente.")

        for lbl in sorted(selected_labels):
            mask = y_true == lbl
            if mask.sum() == 0:
                continue
            preds = y_pred[mask]
            counts = pd.Series(preds).value_counts().sort_index()
            total  = len(preds)

            with st.expander(f"**{CLASS_NAMES[lbl]}** — {total:,} pacotes"):
                rows = []
                for pred_lbl, cnt in counts.items():
                    correct = pred_lbl == lbl
                    rows.append({
                        "Previsto como":  CLASS_NAMES[pred_lbl],
                        "Nº pacotes":     int(cnt),
                        "% do total":     f"{cnt/total:.1%}",
                        "Correto?":       "✓" if correct else "✗",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True,
                             hide_index=True, height=min(200, 50 + 35*len(rows)))
