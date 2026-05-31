"""Página 3 — Pipeline: extração de features e treino."""
from pathlib import Path
import subprocess, sys
import streamlit as st


def _run_script(script: Path, label: str):
    placeholder = st.empty()
    placeholder.info(f"⏳ Executando `{script.name}`...")
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, cwd=str(script.parent.parent),
    )
    if result.returncode == 0:
        placeholder.success(f"✓ {label} concluído.")
        with st.expander("Saída do script"):
            st.code(result.stdout[-4000:] if len(result.stdout) > 4000
                    else result.stdout, language="text")
    else:
        placeholder.error(f"✗ {label} falhou.")
        st.code(result.stderr[-3000:], language="text")


def render(root: Path):
    st.title("⚙️ Pipeline de Extração e Treino")
    det = root / "detection"

    st.subheader("Etapas disponíveis")

    tab1, tab2, tab3 = st.tabs([
        "1 · Features multiclasse",
        "2 · Merge FakeClientID",
        "3 · Treino multiclasse",
    ])

    with tab1:
        st.markdown(
            "Lê todos os CSVs Kim 2026 em `data/parsed/`, extrai 12 features stateful "
            "e gera `multiclass/data/features.csv` com 5 classes."
        )
        script = det / "multiclass" / "01_features.py"
        st.code(f"python {script.relative_to(root)}", language="bash")
        if st.button("▶ Executar 01_features.py", type="primary"):
            _run_script(script, "Extração de features")

    with tab2:
        st.markdown(
            "Adiciona a classe FakeClientID (Alkhatib 2021), recalcula normalização "
            "combinada e salva `X/y_train/test.npy`."
        )
        script2 = det / "multiclass" / "01b_merge_fakeclientid.py"
        st.code(f"python {script2.relative_to(root)}", language="bash")
        if st.button("▶ Executar 01b_merge_fakeclientid.py", type="primary"):
            _run_script(script2, "Merge FakeClientID")

    with tab3:
        st.markdown(
            "Treina o XGBoost `multi:softprob` (6 classes, 200 estimadores) e salva "
            "`multiclass_classifier.json` + `results.json`."
        )
        script3 = det / "multiclass" / "02_train.py"
        st.code(f"python {script3.relative_to(root)}", language="bash")
        if st.button("▶ Executar 02_train.py", type="primary", key="train"):
            _run_script(script3, "Treino multiclasse")

    st.divider()
    st.subheader("Status dos arquivos gerados")

    files = {
        "multiclass/data/features.csv":              det / "multiclass/data/features.csv",
        "multiclass/data/X_train.npy":               det / "multiclass/data/X_train.npy",
        "multiclass/data/X_test.npy":                det / "multiclass/data/X_test.npy",
        "multiclass/model/multiclass_classifier.json": det / "multiclass/model/multiclass_classifier.json",
        "multiclass/model/results.json":             det / "multiclass/model/results.json",
        "multiclass/model/norm_params.json":         det / "multiclass/model/norm_params.json",
    }

    rows = []
    for label, p in files.items():
        exists = p.exists()
        size   = f"{p.stat().st_size / 1e6:.1f} MB" if exists else "—"
        rows.append({"Arquivo": label, "Status": "✓" if exists else "✗", "Tamanho": size})

    import pandas as pd
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
