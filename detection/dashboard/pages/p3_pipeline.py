"""Página 3 — Pipeline: extração de features, treino e avaliação."""
from pathlib import Path
import subprocess, sys
import pandas as pd
import streamlit as st


def _run_script(script: Path, label: str,
                extra_args: list[str] | None = None,
                cwd: Path | None = None):
    ph = st.empty()
    ph.info(f"⏳ Executando `{script.name}`...")
    result = subprocess.run(
        [sys.executable, str(script)] + (extra_args or []),
        capture_output=True, text=True,
        cwd=str(cwd or script.parent.parent),
    )
    if result.returncode == 0:
        ph.success(f"✓ {label} concluído.")
        with st.expander("Saída do script"):
            out = result.stdout
            st.code(out[-4000:] if len(out) > 4000 else out, language="text")
    else:
        ph.error(f"✗ {label} falhou.")
        st.code(result.stderr[-3000:], language="text")


def _pipeline_tabs(det: Path, root: Path):
    tab1, tab2, tab3 = st.tabs([
        "1 · Features multiclasse",
        "2 · Merge FakeClientID",
        "3 · Treino multiclasse",
    ])
    with tab1:
        st.markdown(
            "Lê todos os CSVs Kim 2026 em `detection/data/parsed/`, extrai 12 features "
            "stateful e gera `multiclass/data/features.csv` com 5 classes."
        )
        s = det / "multiclass" / "01_features.py"
        st.code(f"python {s.relative_to(root)}", language="bash")
        if st.button("▶ Executar 01_features.py", type="primary"):
            _run_script(s, "Extração de features")

    with tab2:
        st.markdown(
            "Adiciona a classe FakeClientID (Alkhatib 2021), recalcula normalização "
            "combinada e salva `X/y_train/test.npy`."
        )
        s = det / "multiclass" / "01b_merge_fakeclientid.py"
        st.code(f"python {s.relative_to(root)}", language="bash")
        if st.button("▶ Executar 01b_merge_fakeclientid.py", type="primary"):
            _run_script(s, "Merge FakeClientID")

    with tab3:
        st.markdown(
            "Treina XGBoost `multi:softprob` (6 classes, 200 estimadores) e salva "
            "`multiclass_classifier.json` + `results.json`."
        )
        s = det / "multiclass" / "02_train.py"
        st.code(f"python {s.relative_to(root)}", language="bash")
        if st.button("▶ Executar 02_train.py", type="primary", key="train_orig"):
            _run_script(s, "Treino multiclasse")


def _cenario_a(det: Path, root: Path):
    st.markdown(
        "Usa o **modelo já treinado** (dataset sequencial) e avalia sobre o "
        "`features_misto.csv` gerado pelo Mixer.  \n"
        "*Responde: o modelo generaliza para cenários onde ataques coexistem "
        "com tráfego legítimo?*"
    )
    misto_csv     = det / "multiclass" / "data"  / "features_misto.csv"
    model_json    = det / "multiclass" / "model" / "multiclass_classifier.json"
    results_misto = det / "multiclass" / "model" / "results_misto.json"

    col_status, col_btn = st.columns([3, 1])
    with col_status:
        for label, p in {"features_misto.csv": misto_csv,
                         "multiclass_classifier.json": model_json}.items():
            st.markdown(f"{'✅' if p.exists() else '❌'} `{label}`")

    with col_btn:
        ready = misto_csv.exists() and model_json.exists()
        if st.button("▶ Avaliar", type="primary",
                     disabled=not ready, key="eval_misto"):
            script_eval = det / "multiclass" / "03_evaluate_misto.py"
            _run_script(script_eval, "Avaliação Cenário A",
                        extra_args=["--csv", str(misto_csv)], cwd=root)
            if results_misto.exists():
                st.success("Salvo em `model/results_misto.json`")
                st.info("Vá para **📈 Resultados** para ver a comparação.")

    if not ready:
        missing = []
        if not misto_csv.exists():
            missing.append("`features_misto.csv` — gere no **🔀 Mixer**")
        if not model_json.exists():
            missing.append("`multiclass_classifier.json` — execute o treino acima")
        st.warning("Pré-requisitos ausentes:\n" + "\n".join(f"- {m}" for m in missing))


def _status_table(det: Path):
    files = {
        "data/features.csv":                det / "multiclass/data/features.csv",
        "data/X_train.npy":                 det / "multiclass/data/X_train.npy",
        "data/X_test.npy":                  det / "multiclass/data/X_test.npy",
        "data/features_misto.csv":          det / "multiclass/data/features_misto.csv",
        "model/multiclass_classifier.json": det / "multiclass/model/multiclass_classifier.json",
        "model/norm_params.json":           det / "multiclass/model/norm_params.json",
        "model/results.json":               det / "multiclass/model/results.json",
        "model/results_misto.json":         det / "multiclass/model/results_misto.json",
    }
    rows = [{"Arquivo": lbl,
             "Status":  "✓" if p.exists() else "✗",
             "Tamanho": f"{p.stat().st_size/1e6:.1f} MB" if p.exists() else "—"}
            for lbl, p in files.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render(root: Path):
    st.title("⚙️ Pipeline de Extração e Treino")
    det = root / "detection"

    st.subheader("Pipeline sequencial (dataset original)")
    _pipeline_tabs(det, root)

    st.divider()
    st.subheader("Cenário A — Avaliação no dataset misto")
    _cenario_a(det, root)

    st.divider()
    st.subheader("Status dos arquivos")
    _status_table(det)
