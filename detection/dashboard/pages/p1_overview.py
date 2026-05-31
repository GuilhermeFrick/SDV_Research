"""Página 1 — Visão Geral dos Datasets."""
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

COL_BEN  = "Benigno (label=0)"
COL_ATK  = "Anomalia (label=1)"
COL_PCT  = "% Anomalia"
COL_DUR  = "Duração (s)"

SOURCES = [
    ("benign_traffic.csv",            "Benigno",    "#2ca02c"),
    ("dos_noti_flood.csv",            "DoS",        "#d62728"),
    ("fuzzy_sd_offer_rand_noti1.csv", "Fuzzy",      "#ff7f0e"),
    ("fuzzy_sd_offer_rand_noti2.csv", "Fuzzy",      "#ff7f0e"),
    ("fuzzy_sd_offer_rand_noti3.csv", "Fuzzy",      "#ff7f0e"),
    ("mitm_multi_attacker.csv",       "MITM Multi", "#9467bd"),
    ("mitm_single_attacker.csv",      "MITM Single","#8c564b"),
]


@st.cache_data(show_spinner=False)
def _missing_row(fname: str, classe: str, color: str) -> dict:
    return {
        "Arquivo": fname, "Classe": classe,
        "Total": 0, COL_BEN: 0, COL_ATK: 0,
        COL_PCT: 0.0, COL_DUR: 0,
        "t_min": None, "t_max": None,
        "color": color, "exists": False,
    }


def _read_file_stats(p: Path) -> dict:
    has_label = "label" in pd.read_csv(p, nrows=0).columns
    cols      = ["timestamp", "label"] if has_label else ["timestamp"]
    total = n_ben = n_atk = 0
    t_min = t_max = None
    for chunk in pd.read_csv(p, usecols=cols, chunksize=500_000):
        total += len(chunk)
        ts     = chunk["timestamp"]
        t_min  = ts.min() if t_min is None else min(t_min, ts.min())
        t_max  = ts.max() if t_max is None else max(t_max, ts.max())
        if has_label:
            n_ben += int((chunk["label"] == 0).sum())
            n_atk += int((chunk["label"] == 1).sum())
    return {
        "total": total,
        "n_ben": n_ben if has_label else total,
        "n_atk": n_atk if has_label else 0,
        "t_min": t_min, "t_max": t_max,
    }


@st.cache_data(show_spinner=False)
def _load_summary(parsed_dir_str: str) -> list[dict]:
    parsed_dir = Path(parsed_dir_str)
    rows = []
    for fname, classe, color in SOURCES:
        p = parsed_dir / fname
        if not p.exists():
            rows.append(_missing_row(fname, classe, color))
            continue
        s   = _read_file_stats(p)
        dur = round(s["t_max"] - s["t_min"], 1) if s["t_min"] is not None else 0
        pct = round(s["n_atk"] / s["total"] * 100, 2) if s["total"] > 0 else 0.0
        rows.append({
            "Arquivo": fname, "Classe": classe,
            "Total": s["total"],
            COL_BEN: s["n_ben"], COL_ATK: s["n_atk"],
            COL_PCT: pct, COL_DUR: dur,
            "t_min": s["t_min"], "t_max": s["t_max"],
            "color": color, "exists": True,
        })
    return rows


def render(root: Path):
    st.title("📊 Visão Geral dos Datasets")
    parsed_dir = root / "detection" / "data" / "parsed"
    multiclass_dir = root / "detection" / "multiclass" / "data"

    # ── Carregar metadados ─────────────────────────────────────────────────────
    st.subheader("Arquivos de origem (Kim 2026)")
    with st.spinner("Lendo labels e timestamps..."):
        rows = _load_summary(str(parsed_dir))

    existing = [r for r in rows if r["exists"]]
    miss = sum(1 for r in rows if not r["exists"])

    # métricas globais
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Arquivos encontrados", f"{len(existing)} / {len(rows)}")
    total_pkts = sum(r["Total"] for r in existing)
    total_atk  = sum(r[COL_ATK] for r in existing)
    total_ben  = sum(r[COL_BEN] for r in existing)
    c2.metric("Total de pacotes",  f"{total_pkts:,}")
    c3.metric("Pacotes benigno",   f"{total_ben:,}", f"{total_ben/total_pkts*100:.1f}%" if total_pkts else "")
    c4.metric("Pacotes anomalia",  f"{total_atk:,}", f"{total_atk/total_pkts*100:.1f}%" if total_pkts else "")

    # ── Tabela por arquivo ─────────────────────────────────────────────────────
    df_show = pd.DataFrame(rows)[[
        "Arquivo", "Classe", "Total",
        COL_BEN, COL_ATK, COL_PCT, COL_DUR
    ]].copy()
    df_show["Total"]              = df_show["Total"].apply(lambda x: f"{x:,}")
    df_show[COL_BEN]  = df_show[COL_BEN].apply(lambda x: f"{x:,}")
    df_show[COL_ATK] = df_show[COL_ATK].apply(lambda x: f"{x:,}")
    df_show[COL_PCT]         = df_show[COL_PCT].apply(lambda x: f"{x:.1f}%")
    df_show[COL_DUR] = df_show[COL_DUR].apply(lambda x: f"{x:,}")

    st.dataframe(df_show, use_container_width=True, hide_index=True, height=290)

    if miss:
        st.warning(f"{miss} arquivo(s) não encontrado(s) em `{parsed_dir}`. "
                   "Execute o parser (`src/01_parse.py`) primeiro.")

    # ── Gráfico: benigno vs anomalia por arquivo ───────────────────────────────
    st.subheader("Composição de cada arquivo — benigno vs anomalia")

    if existing:
        fig_comp = go.Figure()
        labels_x = [r["Arquivo"].replace("_rand_noti", "…noti") for r in existing]

        fig_comp.add_trace(go.Bar(
            name=COL_BEN,
            x=labels_x,
            y=[r[COL_BEN] for r in existing],
            marker_color="#2ca02c",
            hovertemplate="%{x}<br>Benigno: %{y:,}<extra></extra>",
        ))
        fig_comp.add_trace(go.Bar(
            name=COL_ATK,
            x=labels_x,
            y=[r[COL_ATK] for r in existing],
            marker_color="#d62728",
            hovertemplate="%{x}<br>Anomalia: %{y:,}<extra></extra>",
        ))
        fig_comp.update_layout(
            barmode="stack",
            height=360,
            margin={"t": 10, "b": 80},
            legend={"orientation": "h", "y": -0.3},
            xaxis_tickangle=-30,
            yaxis_title="Nº de pacotes",
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        st.caption(
            "Cada barra empilhada mostra a proporção de pacotes benigno (verde) "
            "e anomalia (vermelho) dentro do arquivo. "
            "O `benign_traffic.csv` é 100% benigno por definição."
        )

    # ── Timeline — range temporal ──────────────────────────────────────────────
    st.subheader("Range temporal dos arquivos")

    timeline_rows = [r for r in existing if r["t_min"] is not None]
    if timeline_rows:
        fig2 = go.Figure()
        for r in timeline_rows:
            fig2.add_trace(go.Bar(
                name=r["Arquivo"],
                x=[r["t_max"] - r["t_min"]],
                y=[r["Arquivo"]],
                base=[r["t_min"]],
                orientation="h",
                marker_color=r["color"],
                marker_opacity=0.8,
                hovertemplate=(
                    f"<b>{r['Classe']}</b><br>"
                    f"{r['Arquivo']}<br>"
                    f"Início: %{{base:.1f}}s · Fim: {r['t_max']:.1f}s<br>"
                    f"Anomalia: {r[COL_ATK]:,} pkts "
                    f"({r[COL_PCT]:.1f}%)<extra></extra>"
                ),
            ))
        fig2.update_layout(
            barmode="stack", showlegend=False, height=380,
            xaxis_title="Timestamp (s)",
            yaxis={"autorange": "reversed"},
            margin={"t": 10, "b": 30, "l": 230},
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(
            "Sobreposição temporal indica que diferentes cenários de ataque "
            "ocorreram em períodos similares — base para o dataset misto."
        )
    else:
        st.info("Arquivos não encontrados — execute o parser primeiro.")

    # ── Multiclasse features.csv ───────────────────────────────────────────────
    st.divider()
    st.subheader("Dataset multiclasse consolidado (features.csv)")
    feat_csv = multiclass_dir / "features.csv"
    if feat_csv.exists():
        label_counts = pd.concat(
            c["label"] for c in pd.read_csv(feat_csv, usecols=["label"], chunksize=500_000)
        ).value_counts().sort_index()

        CLASS_MAP = {0: "Benigno", 1: "DoS", 2: "Fuzzy",
                     3: "MITM Multi", 4: "MITM Single", 5: "FakeClientID"}
        total_mc = label_counts.sum()
        df_lbl = pd.DataFrame({
            "Label": label_counts.index,
            "Classe": [CLASS_MAP.get(i, str(i)) for i in label_counts.index],
            "Amostras": label_counts.values,
            "% do total": (label_counts.values / total_mc * 100).round(2),
        })

        col1, col2 = st.columns([1, 1])
        with col1:
            st.metric("Total de amostras", f"{total_mc:,}")
            st.dataframe(df_lbl, use_container_width=True, hide_index=True)
        with col2:
            fig3 = px.pie(
                df_lbl, names="Classe", values="Amostras",
                color_discrete_sequence=px.colors.qualitative.T10, hole=0.4,
            )
            fig3.update_traces(
                texttemplate="%{label}<br>%{value:,}<br>(%{percent})",
                textposition="outside",
            )
            fig3.update_layout(height=320, margin={"t": 10, "b": 10}, showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("features.csv não encontrado. Execute `multiclass/01_features.py`.")
