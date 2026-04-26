"""
SOME/IP IDS - Etapa 3: Treinamento XGBoost + CTGAN + Avaliação
===============================================================
Reprodução de Kim et al. (2026) - Seções 5.4, 6.2, 6.3

Implementa:
  • Aumentação de dados com CTGAN (Seção 5.4.1)
  • Treinamento XGBoost com hiperparâmetros da Tabela 2
  • Otimização do limiar de decisão por F1 (Seção 5.4.2)
  • Avaliação nos cenários imbalanceado e balanceado (Seção 6.2)
  • Comparação com 10 baselines (Seção 6.3)
  • Curvas ROC, PR, F1×threshold, DET (Figura 10)

Entrada : data/train_features.csv, data/test_features.csv
Saída   : results/ (modelos, métricas, gráficos)
"""

import warnings
warnings.filterwarnings("ignore")

import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # sem display — funciona em servidores/Colab

from pathlib import Path
from sklearn.metrics import (
    precision_score, recall_score, f1_score, average_precision_score,
    roc_auc_score, roc_curve, precision_recall_curve, det_curve,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import train_test_split
import xgboost as xgb


# ── Colunas de features normalizadas (saída do script 02) ─────────────────────
NORM_FEATURES = [
    "f01_ip_time_interval_norm",
    "f02_someip_likelihood_norm",
    "f03_tcpudp_likelihood_norm",
    "f04_someip_entropy_norm",
    "f05_tcpudp_entropy_norm",
    "f06_someip_payload_changes_norm",
    "f07_tcpudp_payload_changes_norm",
    "f08_ip_length_changes_norm",
    "f09_tcpudp_length_changes_norm",
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. Aumentação com CTGAN
# ══════════════════════════════════════════════════════════════════════════════

def ctgan_augment(X_train: np.ndarray, y_train: np.ndarray,
                  target_ratio: float = 1.0, epochs: int = 100,
                  random_state: int = 42) -> tuple:
    """
    Gera amostras sintéticas de ataque com CTGAN para balancear o treino.

    Configuração do artigo (Seção 5.4.1):
      embedding_dim = 128, hidden_dim = 256, batch_size = 500, epochs = 100

    Retorna (X_aug, y_aug) com classes balanceadas.
    """
    try:
        from ctgan import CTGAN
    except ImportError:
        print("  [AVISO] ctgan não instalado. Pulando aumentação.")
        print("          Para instalar: pip install ctgan")
        return X_train, y_train

    attack_mask   = y_train == 1
    normal_mask   = y_train == 0
    n_normal      = int(normal_mask.sum())
    n_attack_real = int(attack_mask.sum())
    n_needed      = int(n_normal * target_ratio) - n_attack_real

    if n_needed <= 0:
        print(f"  Classes já balanceadas ({n_attack_real} ataques vs {n_normal} normais).")
        return X_train, y_train

    print(f"  Gerando {n_needed:,} amostras sintéticas de ataque com CTGAN...")
    print(f"  (épocas={epochs}, embedding=128, hidden=256, batch=500)")

    # Treina o CTGAN APENAS nos dados de ataque
    X_attack   = X_train[attack_mask]
    df_attack  = pd.DataFrame(X_attack, columns=[f"f{i}" for i in range(X_attack.shape[1])])

    model = CTGAN(
        embedding_dim=128,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        batch_size=500,
        epochs=epochs,
        verbose=False,
    )
    model.fit(df_attack)

    # Gera amostras sintéticas
    synthetic_df  = model.sample(n_needed)
    X_synthetic   = synthetic_df.values
    y_synthetic   = np.ones(n_needed, dtype=int)

    X_aug = np.vstack([X_train, X_synthetic])
    y_aug = np.concatenate([y_train, y_synthetic])

    print(f"  Dataset aumentado: {len(X_aug):,} amostras "
          f"({int((y_aug==0).sum()):,} normais | {int((y_aug==1).sum()):,} ataques)")
    return X_aug, y_aug


# ══════════════════════════════════════════════════════════════════════════════
# 2. Configuração XGBoost (Tabela 2 do artigo)
# ══════════════════════════════════════════════════════════════════════════════

XGBOOST_PARAMS = {
    "objective":        "binary:logistic",   # Logistic Loss (Tabela 2)
    "n_estimators":     1000,                # Number of trees
    "learning_rate":    0.05,                # Learning rate
    "max_depth":        6,                   # Maximum tree depth
    "subsample":        0.8,                 # Row subsampling ratio
    "colsample_bytree": 0.8,                 # Feature sampling ratio
    "min_child_weight": 1,                   # Minimum leaf instance weight
    "reg_lambda":       1.0,                 # L2 regularization (lambda)
    "min_split_loss":   0.0,                 # Minimum loss reduction (gamma)
    "use_label_encoder": False,
    "eval_metric":      "logloss",
    "random_state":     42,
    "n_jobs":           -1,
}


# ══════════════════════════════════════════════════════════════════════════════
# 3. Otimização do limiar de decisão
# ══════════════════════════════════════════════════════════════════════════════

def find_optimal_threshold(y_true, y_prob, metric="f1"):
    """
    Encontra o limiar que maximiza o F1-score (Seção 5.4.2).

    O artigo reporta threshold ótimo de 0.36 no treino.
    Retorna (threshold_ótimo, f1_máximo)
    """
    thresholds = np.arange(0.01, 0.99, 0.01)
    best_t, best_f1 = 0.5, 0.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t  = t

    return best_t, best_f1


# ══════════════════════════════════════════════════════════════════════════════
# 4. Avaliação completa
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, X, y_true, threshold, dataset_name=""):
    """Calcula todas as métricas reportadas no artigo."""
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    prauc = average_precision_score(y_true, y_prob)
    rocauc = roc_auc_score(y_true, y_prob)

    print(f"\n  [{dataset_name}] threshold={threshold:.4f}")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1-Score  : {f1:.4f}")
    print(f"    PR-AUC    : {prauc:.4f}")
    print(f"    ROC-AUC   : {rocauc:.4f}")
    print(f"\n  Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"    TN={cm[0,0]:>8,}  FP={cm[0,1]:>8,}")
    print(f"    FN={cm[1,0]:>8,}  TP={cm[1,1]:>8,}")

    return {
        "dataset":   dataset_name,
        "threshold": threshold,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "pr_auc":    prauc,
        "roc_auc":   rocauc,
        "y_prob":    y_prob,
        "y_true":    y_true,
        "y_pred":    y_pred,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. Geração dos gráficos (Figura 10 do artigo)
# ══════════════════════════════════════════════════════════════════════════════

def plot_performance_curves(res_imbal, res_bal, output_dir: Path):
    """
    Reproduz a Figura 10 do artigo:
      (a) ROC curve
      (b) Precision-Recall curve
      (c) F1 vs Threshold
      (d) Detection Error Tradeoff (DET)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Desempenho do Framework XGBoost para SOME/IP\n(Reprodução de Kim et al. 2026)",
                 fontsize=13, fontweight="bold")

    # ── (a) ROC curve ──────────────────────────────────────────────────────────
    ax = axes[0, 0]
    for res, color, name in [
        (res_imbal, "steelblue", f"Imbalanced (AUC={res_imbal['roc_auc']:.4f})"),
        (res_bal,   "tomato",    f"Balanced   (AUC={res_bal['roc_auc']:.4f})"),
    ]:
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_prob"])
        ax.plot(fpr, tpr, color=color, lw=2, label=name)
    ax.plot([0,1],[0,1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("(a) ROC curve"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ── (b) Precision-Recall curve ─────────────────────────────────────────────
    ax = axes[0, 1]
    for res, color, name in [
        (res_imbal, "steelblue", f"Imbalanced (AP={res_imbal['pr_auc']:.4f})"),
        (res_bal,   "tomato",    f"Balanced   (AP={res_bal['pr_auc']:.4f})"),
    ]:
        prec_c, rec_c, _ = precision_recall_curve(res["y_true"], res["y_prob"])
        ax.plot(rec_c, prec_c, color=color, lw=2, label=name)
    ax.axhline(0.5, color="gray", linestyle="--", lw=1)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("(b) Precision-Recall curve"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ── (c) F1 vs Threshold ────────────────────────────────────────────────────
    ax = axes[1, 0]
    thresholds = np.arange(0.01, 0.99, 0.005)
    for res, color, name in [
        (res_imbal, "steelblue", "Imbalanced"),
        (res_bal,   "tomato",    "Balanced"),
    ]:
        f1s = [f1_score(res["y_true"], (res["y_prob"] >= t).astype(int), zero_division=0)
               for t in thresholds]
        ax.plot(thresholds, f1s, color=color, lw=2, label=name)
    ax.axvline(0.36, color="orange", linestyle="--", lw=1.5, label="Threshold=0.36 (artigo)")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Weighted F1-score")
    ax.set_title("(c) F1-score vs Threshold"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # ── (d) Detection Error Tradeoff (DET) ────────────────────────────────────
    ax = axes[1, 1]
    for res, color, name in [
        (res_imbal, "steelblue", f"Imbalanced"),
        (res_bal,   "tomato",    f"Balanced"),
    ]:
        fpr_d, fnr_d, _ = det_curve(res["y_true"], res["y_prob"])
        ax.plot(fpr_d, fnr_d, color=color, lw=2, label=name)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("False Negative Rate")
    ax.set_title("(d) Detection Error Tradeoff (DET)"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_xlim([0, 0.7]); ax.set_ylim([0, 0.35])

    plt.tight_layout()
    path = output_dir / "figure10_performance_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Gráfico salvo: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Comparação com baselines (Tabela do artigo)
# ══════════════════════════════════════════════════════════════════════════════

def run_baseline_comparison(X_train, y_train, X_test_imbal, y_test_imbal,
                             output_dir: Path):
    """
    Treina e avalia 10 modelos baseline (Seção 6.3).
    Reproduz os gráficos de Precision, Recall, F1, PR-AUC (Figuras 11-12).
    """
    print("\n" + "="*60)
    print("COMPARAÇÃO COM BASELINES (Seção 6.3)")
    print("="*60)

    # Importações condicionais
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
    from sklearn.naive_bayes import GaussianNB

    try:
        import lightgbm as lgb
        LGB_OK = True
    except ImportError:
        LGB_OK = False
        print("  [AVISO] lightgbm não instalado. Pulando LGB.")

    # Define modelos — subset rápido (RF com 100 árvores para velocidade)
    baselines = {
        "XGB": xgb.XGBClassifier(**XGBOOST_PARAMS),
        "RF":  RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "DT":  DecisionTreeClassifier(random_state=42),
        "LR":  LogisticRegression(max_iter=1000, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "NB":  GaussianNB(),
    }
    if LGB_OK:
        import lightgbm as lgb
        baselines["LGB"] = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05,
                                               random_state=42, n_jobs=-1, verbose=-1)

    results = {}
    timing  = {}

    for name, model in baselines.items():
        print(f"\n  → Treinando {name}...")
        t0 = time.time()
        model.fit(X_train, y_train)
        t_train = time.time() - t0

        t0 = time.time()
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_imbal)[:, 1]
        else:
            y_prob = model.decision_function(X_test_imbal)
            # normaliza para [0,1]
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-12)
        t_infer = time.time() - t0

        # Usa threshold=0.36 fixo (do artigo) para comparação justa
        y_pred = (y_prob >= 0.36).astype(int)

        prec   = precision_score(y_test_imbal, y_pred, zero_division=0)
        rec    = recall_score(y_test_imbal, y_pred, zero_division=0)
        f1     = f1_score(y_test_imbal, y_pred, zero_division=0)
        prauc  = average_precision_score(y_test_imbal, y_prob)

        results[name] = {"precision": prec, "recall": rec, "f1": f1, "pr_auc": prauc}
        timing[name]  = {"train_s": t_train, "infer_s": t_infer}

        print(f"    Precision={prec:.3f}  Recall={rec:.3f}  "
              f"F1={f1:.3f}  PR-AUC={prauc:.3f}  "
              f"train={t_train:.2f}s  infer={t_infer:.2f}s")

    # ── Salva tabela de resultados ─────────────────────────────────────────────
    df_res = pd.DataFrame(results).T
    df_res.index.name = "Model"
    df_res.to_csv(output_dir / "baseline_comparison.csv")
    print(f"\n  Tabela salva: {output_dir / 'baseline_comparison.csv'}")
    print(df_res.round(4).to_string())

    # ── Gráfico de barras (Figuras 11-12) ─────────────────────────────────────
    _plot_baseline_bars(results, output_dir)

    return results, timing


def _plot_baseline_bars(results: dict, output_dir: Path):
    """Reproduz os gráficos de barras das Figuras 11-12."""
    models  = list(results.keys())
    metrics = ["precision", "recall", "f1", "pr_auc"]
    titles  = ["(a) Precision", "(b) Recall", "(c) F1-score", "(d) PR-AUC"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Comparação de Modelos — Dataset Imbalanceado\n(Reprodução de Kim et al. 2026)",
                 fontsize=13, fontweight="bold")

    colors = ["#2196F3" if m != "XGB" else "#F44336" for m in models]

    for ax, metric, title in zip(axes.flat, metrics, titles):
        vals = [results[m][metric] for m in models]
        bars = ax.bar(models, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_ylim(0.4, 1.02)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = output_dir / "figures11_12_baseline_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gráfico salvo: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Pipeline principal
# ══════════════════════════════════════════════════════════════════════════════

def run_training(train_csv: str, test_csv: str, output_dir: str,
                 use_ctgan: bool = True, run_baselines: bool = True):
    """
    Pipeline completo de treinamento e avaliação.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Leitura ────────────────────────────────────────────────────────────────
    print("[1/6] Lendo features...")
    df_train = pd.read_csv(train_csv)
    df_test  = pd.read_csv(test_csv)
    print(f"      Treino: {len(df_train):,}  |  Teste: {len(df_test):,}")

    # Verifica colunas disponíveis
    available_feats = [c for c in NORM_FEATURES if c in df_train.columns]
    if len(available_feats) < len(NORM_FEATURES):
        missing = set(NORM_FEATURES) - set(available_feats)
        # Fallback: usa features brutas se normalizadas não existirem
        RAW_FEATURES = [c.replace("_norm","") for c in NORM_FEATURES]
        available_feats = [c for c in RAW_FEATURES if c in df_train.columns]
        print(f"  [AVISO] Usando features brutas (normalizadas ausentes): {missing}")

    print(f"      Features usadas: {available_feats}")

    X_train = df_train[available_feats].fillna(0).values
    y_train = df_train["label"].values
    X_test  = df_test[available_feats].fillna(0).values
    y_test  = df_test["label"].values

    # ── CTGAN augmentation ─────────────────────────────────────────────────────
    print("\n[2/6] Aumentação com CTGAN (Seção 5.4.1)...")
    if use_ctgan:
        X_train_aug, y_train_aug = ctgan_augment(X_train, y_train, epochs=100)
    else:
        print("  CTGAN desabilitado (--no-ctgan). Usando dados originais.")
        X_train_aug, y_train_aug = X_train, y_train

    # ── Treinamento XGBoost ────────────────────────────────────────────────────
    print("\n[3/6] Treinando XGBoost (Tabela 2)...")
    t0    = time.time()
    model = xgb.XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train_aug, y_train_aug,
              eval_set=[(X_train, y_train)],
              verbose=False)
    t_train = time.time() - t0
    print(f"  Treinamento concluído em {t_train:.2f}s")

    # Salva modelo
    model_path = out_dir / "xgboost_someip_ids.json"
    model.save_model(str(model_path))
    print(f"  Modelo salvo: {model_path}")

    # ── Otimização do limiar no treino ─────────────────────────────────────────
    print("\n[4/6] Otimizando limiar de decisão (Seção 5.4.2)...")
    y_train_prob = model.predict_proba(X_train)[:, 1]
    opt_threshold, opt_f1 = find_optimal_threshold(y_train, y_train_prob)
    print(f"  Limiar ótimo (treino): {opt_threshold:.4f}  →  F1={opt_f1:.4f}")
    print(f"  Artigo reporta: threshold=0.36, F1=0.97")

    # ── Avaliação: cenário imbalanceado (realista) ─────────────────────────────
    print("\n[5/6] Avaliação nos dois cenários de teste...")
    t0 = time.time()
    res_imbal = evaluate_model(model, X_test, y_test, opt_threshold,
                                dataset_name="Imbalanceado (realista)")
    t_infer = time.time() - t0
    print(f"  Inferência: {t_infer:.4f}s  "
          f"({1e6*t_infer/max(len(X_test),1):.4f} µs/amostra)")

    # Cenário balanceado (downsampling da classe majoritária)
    normal_idx = np.where(y_test == 0)[0]
    attack_idx = np.where(y_test == 1)[0]
    n_bal = min(len(normal_idx), len(attack_idx))
    bal_idx = np.concatenate([
        np.random.choice(normal_idx, n_bal, replace=False),
        np.random.choice(attack_idx, n_bal, replace=False),
    ])
    X_test_bal = X_test[bal_idx]
    y_test_bal = y_test[bal_idx]
    res_bal = evaluate_model(model, X_test_bal, y_test_bal, opt_threshold,
                              dataset_name="Balanceado (downsampling)")

    # ── Gráficos de desempenho (Figura 10) ────────────────────────────────────
    print("\n  Gerando gráficos de desempenho (Figura 10)...")
    plot_performance_curves(res_imbal, res_bal, out_dir)

    # ── Comparação de baselines ────────────────────────────────────────────────
    if run_baselines:
        baseline_results, baseline_timing = run_baseline_comparison(
            X_train_aug, y_train_aug, X_test, y_test, out_dir
        )

    # ── Salva métricas finais em JSON ──────────────────────────────────────────
    metrics_summary = {
        "xgboost": {
            "imbalanced": {k: float(v) for k, v in res_imbal.items()
                           if k not in ("y_prob","y_true","y_pred","dataset_name","dataset")},
            "balanced":   {k: float(v) for k, v in res_bal.items()
                           if k not in ("y_prob","y_true","y_pred","dataset_name","dataset")},
        },
        "optimal_threshold": float(opt_threshold),
        "training_time_s":   float(t_train),
        "inference_time_s":  float(t_infer),
        "n_train":           len(X_train_aug),
        "n_test_imbal":      len(X_test),
        "n_test_bal":        len(X_test_bal),
        "feature_names":     available_feats,
    }
    metrics_path = out_dir / "metrics_summary.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"CONCLUÍDO")
    print(f"  Modelo          : {model_path}")
    print(f"  Gráficos        : {out_dir}")
    print(f"  Métricas (JSON) : {metrics_path}")
    print(f"\n  ─── Resultados finais (limiar={opt_threshold:.4f}) ───")
    print(f"  Imbalanceado → Precision={res_imbal['precision']:.4f}  "
          f"Recall={res_imbal['recall']:.4f}  F1={res_imbal['f1']:.4f}  "
          f"PR-AUC={res_imbal['pr_auc']:.4f}")
    print(f"  Balanceado    → Precision={res_bal['precision']:.4f}  "
          f"Recall={res_bal['recall']:.4f}  F1={res_bal['f1']:.4f}  "
          f"PR-AUC={res_bal['pr_auc']:.4f}")
    print(f"\n  Artigo (Kim et al.): F1=0.97, PR-AUC=0.93 (imbalanceado)")

    return model, metrics_summary


# ══════════════════════════════════════════════════════════════════════════════
# Ponto de entrada
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Treinamento XGBoost + CTGAN para SOME/IP IDS (Kim et al. 2026)"
    )
    ap.add_argument("--train-csv",      default="data/train_features.csv")
    ap.add_argument("--test-csv",       default="data/test_features.csv")
    ap.add_argument("--output-dir",     default="results/")
    ap.add_argument("--no-ctgan",       action="store_true",
                    help="Pula a aumentação CTGAN (mais rápido, menor acurácia)")
    ap.add_argument("--no-baselines",   action="store_true",
                    help="Pula a comparação com baselines (mais rápido)")
    args = ap.parse_args()

    run_training(
        train_csv     = args.train_csv,
        test_csv      = args.test_csv,
        output_dir    = args.output_dir,
        use_ctgan     = not args.no_ctgan,
        run_baselines = not args.no_baselines,
    )
