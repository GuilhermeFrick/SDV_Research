"""Página 4 — Resultados do Classificador Multiclasse."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

CLASS_NAMES = ["Benigno", "DoS", "Fuzzy", "MITM Multi", "MITM Single", "FakeClientID"]
CLASS_COLOR = {
    "Benigno":     "#2ca02c", "DoS":        "#d62728",
    "Fuzzy":       "#ff7f0e", "MITM Multi":  "#9467bd",
    "MITM Single": "#8c564b", "FakeClientID":"#1f77b4",
}

FEAT_NAMES = [
    "f01 ip_time_interval",    "f08 payload_hamming",
    "f11 ip_len_change",       "f12 tcpudp_len_change",
    "f13 payload_repeat_rate", "f15 payload_len",
    "f16 tcpudp_len",          "f17 src_packet_rate",
    "f18 payload_diversity",   "f19 is_sd",
    "f20 service_diversity",   "f21 relay_service",
    "f22 clientid_diversity",
]


def _load_results(model_dir: Path) -> dict | None:
    p = model_dir / "results.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def render(root: Path):
    st.title("📈 Resultados — Classificador Multiclasse")

    model_dir = root / "detection" / "multiclass" / "model"
    res = _load_results(model_dir)

    if res is None:
        st.error("Arquivo `results.json` não encontrado. "
                 "Execute o treino na página ⚙️ Pipeline.")
        return

    # ── Métricas globais ───────────────────────────────────────────────────────
    st.subheader("Métricas globais")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("F1 Macro",      f"{res['f1_macro']:.4f}")
    c2.metric("F1 Weighted",   f"{res['f1_weighted']:.4f}")
    c3.metric("Acurácia",      f"{res['accuracy']:.4f}")
    c4.metric("Latência/pkt",  f"{res['t_single_ms']:.2f} ms")
    c5.metric("Throughput",    f"{res['throughput_pkt_s']:,.0f} pkt/s")

    st.divider()

    col_left, col_right = st.columns([1.1, 1])

    # ── Matriz de confusão ─────────────────────────────────────────────────────
    with col_left:
        st.subheader("Matriz de confusão")

        cm = np.array(res["confusion_matrix"])
        n  = len(CLASS_NAMES)

        # Normalizar por linha (recall por classe)
        norm_mode = st.radio("Normalização", ["Absoluta", "Por linha (recall)"],
                             horizontal=True)
        if norm_mode == "Por linha (recall)":
            row_sum = cm.sum(axis=1, keepdims=True)
            cm_disp = np.where(row_sum > 0, cm / row_sum, 0)
            fmt_text = [[f"{cm_disp[i,j]:.3f}" for j in range(n)] for i in range(n)]
            colorscale = "Blues"
        else:
            cm_disp = cm
            fmt_text = [[f"{cm[i,j]:,}" for j in range(n)] for i in range(n)]
            colorscale = "Blues"

        fig_cm = go.Figure(go.Heatmap(
            z=cm_disp,
            x=CLASS_NAMES,
            y=CLASS_NAMES,
            text=fmt_text,
            texttemplate="%{text}",
            textfont=dict(size=11),
            colorscale=colorscale,
            showscale=True,
            hovertemplate=(
                "Real: %{y}<br>Previsto: %{x}<br>Valor: %{text}<extra></extra>"
            ),
        ))
        fig_cm.update_layout(
            xaxis_title="Previsto",
            yaxis_title="Real",
            yaxis=dict(autorange="reversed"),
            height=440,
            margin=dict(t=10, b=60, l=100, r=10),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # ── F1 por classe + comparativo binário ───────────────────────────────────
    with col_right:
        st.subheader("F1 por classe")

        f1_multi = res["f1_per_class"]
        f1_ovr   = res.get("ovr_f1", {})

        df_f1 = pd.DataFrame({
            "Classe":      CLASS_NAMES,
            "F1 Multi":    [f1_multi.get(c, 0) for c in CLASS_NAMES],
            "F1 Binário":  [f1_ovr.get(c, None) for c in CLASS_NAMES],
        })

        fig_f1 = go.Figure()
        fig_f1.add_trace(go.Bar(
            name="Multiclasse",
            x=df_f1["Classe"], y=df_f1["F1 Multi"],
            marker_color=[CLASS_COLOR[c] for c in CLASS_NAMES],
            text=[f"{v:.4f}" for v in df_f1["F1 Multi"]],
            textposition="outside",
        ))
        fig_f1.add_trace(go.Scatter(
            name="Binário (referência)",
            x=df_f1["Classe"],
            y=df_f1["F1 Binário"],
            mode="markers",
            marker=dict(symbol="diamond", size=10, color="#333"),
        ))
        fig_f1.update_layout(
            yaxis=dict(range=[0.99, 1.002], title="F1-Score"),
            height=340,
            margin=dict(t=10, b=60),
            legend=dict(orientation="h", y=-0.25),
            barmode="group",
        )
        st.plotly_chart(fig_f1, use_container_width=True)

        # Tabela detalhada
        df_f1["Δ vs Binário"] = df_f1.apply(
            lambda r: f"{r['F1 Multi'] - r['F1 Binário']:+.4f}"
            if r["F1 Binário"] is not None else "N/A", axis=1
        )
        df_f1["F1 Multi"]   = df_f1["F1 Multi"].map("{:.4f}".format)
        df_f1["F1 Binário"] = df_f1["F1 Binário"].map(
            lambda x: f"{x:.4f}" if x is not None else "—")
        st.dataframe(df_f1, use_container_width=True, hide_index=True, height=250)

    # ── Importância das features ───────────────────────────────────────────────
    st.divider()
    st.subheader("Importância das features")

    model_json = model_dir / "multiclass_classifier.json"
    if model_json.exists():
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(str(model_json))
        importances = model.feature_importances_

        n_feats = len(importances)
        feat_labels = FEAT_NAMES[:n_feats]

        df_imp = pd.DataFrame({"Feature": feat_labels, "Importância": importances})
        df_imp = df_imp.sort_values("Importância", ascending=True)

        fig_imp = px.bar(
            df_imp, x="Importância", y="Feature",
            orientation="h",
            color="Importância",
            color_continuous_scale="Blues",
            text=df_imp["Importância"].map("{:.3f}".format),
        )
        fig_imp.update_traces(textposition="outside")
        fig_imp.update_layout(
            coloraxis_showscale=False, height=420,
            margin=dict(t=10, b=10, l=180),
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Modelo `multiclass_classifier.json` não encontrado.")

    # ── Análise de erros ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("Análise dos erros (FP + FN por classe)")

    cm_abs = np.array(res["confusion_matrix"])
    error_data = []
    for i, cls in enumerate(CLASS_NAMES):
        tp = cm_abs[i, i]
        fn = cm_abs[i].sum() - tp          # erros de recall
        fp = cm_abs[:, i].sum() - tp       # erros de precisão
        n_real = cm_abs[i].sum()
        error_data.append({
            "Classe": cls,
            "TP": int(tp), "FP": int(fp), "FN": int(fn),
            "Total real": int(n_real),
            "Recall": f"{tp/n_real:.4f}" if n_real > 0 else "—",
        })
    st.dataframe(pd.DataFrame(error_data), use_container_width=True, hide_index=True)
