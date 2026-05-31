"""Página 1 — Visão Geral dos Datasets."""
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

SOURCES = [
    ("benign_traffic.csv",            "Benigno",    "#2ca02c"),
    ("dos_noti_flood.csv",            "DoS",        "#d62728"),
    ("fuzzy_sd_offer_rand_noti1.csv", "Fuzzy",      "#ff7f0e"),
    ("fuzzy_sd_offer_rand_noti2.csv", "Fuzzy",      "#ff7f0e"),
    ("fuzzy_sd_offer_rand_noti3.csv", "Fuzzy",      "#ff7f0e"),
    ("mitm_multi_attacker.csv",       "MITM Multi", "#9467bd"),
    ("mitm_single_attacker.csv",      "MITM Single","#8c564b"),
]

ALKHATIB = [
    ("fakeClientID.pcap→features",  "FakeClientID", "#1f77b4"),
    ("fakeClientID2.pcap→features", "FakeClientID", "#1f77b4"),
]


def _load_summary(parsed_dir: Path) -> list[dict]:
    rows = []
    for fname, classe, color in SOURCES:
        p = parsed_dir / fname
        if not p.exists():
            rows.append({"Arquivo": fname, "Classe": classe, "Pacotes": 0,
                         "t_min": None, "t_max": None, "color": color, "exists": False})
            continue
        counts = 0
        t_min = t_max = None
        for chunk in pd.read_csv(p, usecols=["timestamp"], chunksize=500_000):
            counts += len(chunk)
            ts = chunk["timestamp"]
            t_min = ts.min() if t_min is None else min(t_min, ts.min())
            t_max = ts.max() if t_max is None else max(t_max, ts.max())
        dur = round(t_max - t_min, 1) if t_min is not None else 0
        rows.append({"Arquivo": fname, "Classe": classe, "Pacotes": counts,
                     "Duração (s)": dur, "t_min": t_min, "t_max": t_max,
                     "color": color, "exists": True})
    return rows


def render(root: Path):
    st.title("📊 Visão Geral dos Datasets")
    parsed_dir = root / "data" / "parsed"
    multiclass_dir = root / "detection" / "multiclass" / "data"

    # ── Status dos arquivos ────────────────────────────────────────────────────
    st.subheader("Arquivos de origem (Kim 2026)")

    with st.spinner("Lendo metadados..."):
        rows = _load_summary(parsed_dir)

    df_meta = pd.DataFrame(rows)[["Arquivo", "Classe", "Pacotes", "Duração (s)", "exists"]]
    ok   = df_meta["exists"]
    miss = (~ok).sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Arquivos encontrados", f"{ok.sum()} / {len(rows)}")
    c2.metric("Arquivos ausentes",    str(miss), delta="-" + str(miss) if miss else None,
              delta_color="inverse")
    total_pkts = int(df_meta.loc[ok, "Pacotes"].sum())
    c3.metric("Total de pacotes",     f"{total_pkts:,}")

    st.dataframe(
        df_meta.drop(columns=["exists"])
               .style.apply(lambda r: ["background:#fff3cd" if not rows[i]["exists"] else ""
                                       for i, _ in enumerate(r)], axis=1),
        use_container_width=True, height=280,
    )

    # ── Gráfico de barras — pacotes por classe ─────────────────────────────────
    st.subheader("Distribuição de pacotes por classe")

    agg = {}
    for r in rows:
        if r["exists"]:
            agg[r["Classe"]] = agg.get(r["Classe"], 0) + r["Pacotes"]

    if agg:
        fig = px.bar(
            x=list(agg.keys()), y=list(agg.values()),
            color=list(agg.keys()),
            color_discrete_sequence=px.colors.qualitative.T10,
            labels={"x": "Classe", "y": "Pacotes"},
            text_auto=True,
        )
        fig.update_layout(showlegend=False, height=320,
                          margin=dict(t=10, b=10))
        fig.update_traces(texttemplate="%{y:,.0f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    # ── Timeline — range temporal de cada arquivo ──────────────────────────────
    st.subheader("Range temporal dos arquivos")

    timeline_rows = [r for r in rows if r["exists"] and r["t_min"] is not None]
    if timeline_rows:
        fig2 = go.Figure()
        for i, r in enumerate(timeline_rows):
            fig2.add_trace(go.Bar(
                name=r["Arquivo"],
                x=[r["t_max"] - r["t_min"]],
                y=[r["Arquivo"]],
                base=[r["t_min"]],
                orientation="h",
                marker_color=r["color"],
                marker_opacity=0.75,
                hovertemplate=(
                    f"<b>{r['Classe']}</b><br>"
                    f"Arquivo: {r['Arquivo']}<br>"
                    f"Início: %{{base:.1f}}s<br>"
                    f"Fim: %{{x:.1f}}s<br>"
                    f"Duração: {r.get('Duração (s)', 0)}s<extra></extra>"
                ),
            ))
        fig2.update_layout(
            barmode="stack", showlegend=False, height=380,
            xaxis_title="Timestamp (s)",
            yaxis=dict(autorange="reversed"),
            margin=dict(t=10, b=30, l=220),
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Cada barra representa o range de timestamps de um arquivo CSV. "
                   "Sobreposição indica que os cenários ocorrem em períodos similares.")
    else:
        st.info("Arquivos CSV não encontrados em `data/parsed/`. "
                "Execute o parser (`src/01_parse.py`) primeiro.")

    # ── Multiclasse features.csv ───────────────────────────────────────────────
    st.subheader("Dataset multiclasse (features.csv)")
    feat_csv = multiclass_dir / "features.csv"
    if feat_csv.exists():
        labels = pd.concat(
            c["label"] for c in pd.read_csv(feat_csv, usecols=["label"], chunksize=500_000)
        ).value_counts().sort_index()

        CLASS_MAP = {0: "Benigno", 1: "DoS", 2: "Fuzzy",
                     3: "MITM Multi", 4: "MITM Single", 5: "FakeClientID"}
        df_lbl = pd.DataFrame({
            "Classe": [CLASS_MAP.get(i, str(i)) for i in labels.index],
            "Amostras": labels.values,
            "% do total": (labels.values / labels.sum() * 100).round(2),
        })
        col1, col2 = st.columns([1, 1])
        col1.dataframe(df_lbl, use_container_width=True, hide_index=True)

        fig3 = px.pie(df_lbl, names="Classe", values="Amostras",
                      color_discrete_sequence=px.colors.qualitative.T10,
                      hole=0.4)
        fig3.update_layout(height=300, margin=dict(t=10, b=10))
        col2.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("features.csv não encontrado. Execute `multiclass/01_features.py`.")
