"""Página 2 — Mixer de Datasets."""
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

CLASS_MAP   = {0: "Benigno", 1: "DoS", 2: "Fuzzy",
               3: "MITM Multi", 4: "MITM Single", 5: "FakeClientID"}
CLASS_COLOR = {
    "Benigno":     "#2ca02c", "DoS":        "#d62728",
    "Fuzzy":       "#ff7f0e", "MITM Multi":  "#9467bd",
    "MITM Single": "#8c564b", "FakeClientID":"#1f77b4",
}

ATTACK_SOURCES = {
    "DoS":         ["dos_noti_flood.csv"],
    "Fuzzy":       ["fuzzy_sd_offer_rand_noti1.csv",
                    "fuzzy_sd_offer_rand_noti2.csv",
                    "fuzzy_sd_offer_rand_noti3.csv"],
    "MITM Multi":  ["mitm_multi_attacker.csv"],
    "MITM Single": ["mitm_single_attacker.csv"],
}
LABEL_MAP = {"DoS": 1, "Fuzzy": 2, "MITM Multi": 3, "MITM Single": 4}


@st.cache_data(show_spinner=False)
def _sample_timestamps(csv_path: Path, label_col: str,
                        n_sample: int = 50_000) -> pd.DataFrame:
    """Lê timestamps e labels de um CSV, amostrando se necessário."""
    cols = ["timestamp", label_col] if label_col in pd.read_csv(
        csv_path, nrows=1).columns else ["timestamp"]
    chunks = []
    for chunk in pd.read_csv(csv_path, usecols=cols, chunksize=200_000):
        chunks.append(chunk)
    df = pd.concat(chunks)
    if len(df) > n_sample:
        df = df.sample(n_sample, random_state=42)
    return df.sort_values("timestamp").reset_index(drop=True)


def _build_preview(parsed_dir: Path,
                   selected_attacks: list[str],
                   n_benigno: int,
                   strategy: str) -> pd.DataFrame | None:
    """Constrói uma amostra do dataset misto para preview."""
    parts = []

    # Benigno
    ben_csv = parsed_dir / "benign_traffic.csv"
    if ben_csv.exists():
        df_b = _sample_timestamps(ben_csv, "label", n_sample=min(n_benigno, 100_000))
        df_b["classe"] = "Benigno"
        df_b["label"]  = 0
        parts.append(df_b[["timestamp", "classe", "label"]])

    # Ataques selecionados
    for atk in selected_attacks:
        mc_label = LABEL_MAP[atk]
        for fname in ATTACK_SOURCES[atk]:
            p = parsed_dir / fname
            if not p.exists():
                continue
            df_a = _sample_timestamps(p, "label", n_sample=20_000)
            # manter apenas pacotes de ataque (label==1 no arquivo original)
            if "label" in df_a.columns:
                df_a = df_a[df_a["label"] == 1]
            df_a = df_a.copy()
            df_a["classe"] = atk
            df_a["label"]  = mc_label
            parts.append(df_a[["timestamp", "classe", "label"]])

    if not parts:
        return None

    df = pd.concat(parts, ignore_index=True)

    if strategy == "⏱️ Temporal (por timestamp)":
        df = df.sort_values("timestamp").reset_index(drop=True)
    elif strategy == "🎲 Aleatório (embaralhado)":
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    else:  # proporcional já está ok
        df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def render(root: Path):
    st.title("🔀 Mixer de Datasets")
    st.caption("Configure como os arquivos de ataque serão combinados com o tráfego benigno "
               "para gerar o dataset misto respeitando a realidade temporal.")

    parsed_dir = root / "data" / "parsed"

    # ── Configurações ──────────────────────────────────────────────────────────
    col_cfg, col_prev = st.columns([1, 2])

    with col_cfg:
        st.subheader("⚙️ Configuração")

        st.markdown("**Ataques a incluir:**")
        sel_dos   = st.checkbox("DoS",         value=True)
        sel_fuzzy = st.checkbox("Fuzzy",       value=True)
        sel_mitm_m= st.checkbox("MITM Multi",  value=True)
        sel_mitm_s= st.checkbox("MITM Single", value=True)

        selected = []
        if sel_dos:    selected.append("DoS")
        if sel_fuzzy:  selected.append("Fuzzy")
        if sel_mitm_m: selected.append("MITM Multi")
        if sel_mitm_s: selected.append("MITM Single")

        st.divider()

        strategy = st.radio(
            "Estratégia de mistura",
            ["⏱️ Temporal (por timestamp)",
             "🎲 Aleatório (embaralhado)",
             "📊 Proporcional (por classe)"],
            help=(
                "**Temporal**: ordena todos os pacotes pelo timestamp real — "
                "simula um cenário onde DoS e MITM ocorrem simultaneamente na rede.\n\n"
                "**Aleatório**: embaralha sem considerar tempo — "
                "garante que o modelo não aprende ordem temporal.\n\n"
                "**Proporcional**: mantém ordenação temporal mas ajusta "
                "a quantidade de cada classe para um ratio configurável."
            ),
        )

        ratio_benigno = st.slider(
            "% de pacotes benigno no dataset final",
            min_value=30, max_value=90, value=60, step=5,
            help="O restante é dividido igualmente entre os ataques selecionados.",
        )

        st.divider()
        run_preview = st.button("👁️ Gerar Preview", type="primary",
                                disabled=len(selected) == 0)

    # ── Preview visual ─────────────────────────────────────────────────────────
    with col_prev:
        st.subheader("📊 Preview do dataset misto")

        if run_preview:
            if not (parsed_dir / "benign_traffic.csv").exists():
                st.warning("Arquivos CSV não encontrados. Execute o parser primeiro.")
            else:
                with st.spinner("Amostrando dados..."):
                    n_ben = int(200_000 * ratio_benigno / 100)
                    df_mix = _build_preview(parsed_dir, selected, n_ben, strategy)

                if df_mix is None:
                    st.error("Nenhum dado carregado.")
                else:
                    # ── Distribuição de classes ────────────────────────────────
                    counts = df_mix["classe"].value_counts().reset_index()
                    counts.columns = ["Classe", "Amostras"]
                    counts["color"] = counts["Classe"].map(CLASS_COLOR)

                    fig_bar = px.bar(
                        counts, x="Classe", y="Amostras", color="Classe",
                        color_discrete_map=CLASS_COLOR,
                        text_auto=True, title="Distribuição das classes no mix",
                    )
                    fig_bar.update_layout(showlegend=False, height=280,
                                          margin=dict(t=30, b=10))
                    st.plotly_chart(fig_bar, use_container_width=True)

                    # ── Timeline — density de pacotes por classe ───────────────
                    st.markdown("**Distribuição temporal (densidade de pacotes)**")

                    n_bins = 200
                    t_min  = df_mix["timestamp"].min()
                    t_max  = df_mix["timestamp"].max()
                    bins   = np.linspace(t_min, t_max, n_bins + 1)
                    bin_centers = 0.5 * (bins[:-1] + bins[1:])

                    fig_tl = go.Figure()
                    for classe in df_mix["classe"].unique():
                        ts = df_mix.loc[df_mix["classe"] == classe, "timestamp"].values
                        counts_hist, _ = np.histogram(ts, bins=bins)
                        fig_tl.add_trace(go.Scatter(
                            x=bin_centers, y=counts_hist,
                            name=classe,
                            mode="lines",
                            line=dict(color=CLASS_COLOR.get(classe, "#888"), width=1.5),
                            fill="tozeroy",
                            fillcolor=CLASS_COLOR.get(classe, "#888"),
                            opacity=0.45,
                        ))
                    fig_tl.update_layout(
                        xaxis_title="Timestamp (s)",
                        yaxis_title="Pacotes por janela",
                        height=320,
                        margin=dict(t=10, b=30),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    )
                    st.plotly_chart(fig_tl, use_container_width=True)
                    st.caption(
                        "Cada cor representa uma classe. Sobreposição indica "
                        "coexistência temporal de ataque e tráfego legítimo — "
                        "esse é o cenário real a ser classificado."
                    )

                    # ── Tabela resumo ──────────────────────────────────────────
                    total = len(df_mix)
                    st.markdown(f"**Total amostrado: {total:,} pacotes** "
                                f"(preview — dataset completo será maior)")

        else:
            st.info("Configure os parâmetros ao lado e clique em **Gerar Preview**.")

    # ── Botão de exportação ────────────────────────────────────────────────────
    st.divider()
    st.subheader("💾 Gerar dataset misto completo")

    out_path = root / "detection" / "multiclass" / "data" / "features_misto.csv"

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown(
            f"O pipeline irá:\n"
            f"1. Ler todos os CSVs selecionados de `data/parsed/`\n"
            f"2. Ordenar por timestamp (estratégia: **{strategy}**)\n"
            f"3. Extrair as 13 features stateful em ordem cronológica\n"
            f"4. Salvar em `detection/multiclass/data/features_misto.csv`\n\n"
            f"⚠️ Isso pode levar **5–20 minutos** dependendo do hardware."
        )
    with col_b:
        if st.button("🚀 Gerar features_misto.csv", type="primary",
                     disabled=len(selected) == 0):
            st.warning(
                "Execução do pipeline completo disponível na página **⚙️ Pipeline**. "
                "O preview acima confirma a mistura antes de extrair."
            )
