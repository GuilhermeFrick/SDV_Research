"""
SOME/IP IDS - Etapa 2: Extração de Features Comportamentais
=============================================================
Reprodução de Kim et al. (2026) - Seção 5.2 (Feature Extraction) e
Seção 5.3 (Feature Vector Generation)

Implementa as 9 features da Tabela 1 do artigo:

  Categoria          | Feature
  -------------------|------------------------------------------
  Time interval      | IP time interval
  Payload likelihood | SOME/IP(-SD) likelihood, TCP/UDP likelihood
  Payload entropy    | SOME/IP(-SD) entropy,   TCP/UDP entropy
  Payload changes    | SOME/IP(-SD) payload changes, TCP/UDP payload changes
  Length changes     | IP length changes, TCP/UDP length changes

As features são calculadas POR FLUXO (five-tuple: src_ip, dst_ip,
src_port, dst_port, transport) seguindo a Seção 5.2 do artigo.

Entrada : data/parsed_packets.csv  (saída do script 01)
Saída   : data/features.csv        (pronto para o XGBoost)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict


# ══════════════════════════════════════════════════════════════════════════════
# 1. Modelo de probabilidade de bytes (para likelihood e entropia)
# ══════════════════════════════════════════════════════════════════════════════

class ByteDistributionModel:
    """
    Modelo de distribuição de bytes por posição, aprendido sobre tráfego benigno.

    Implementa as Equações 2-6 do artigo:
      - Eq. 2: frequência ci(b) de cada byte b na posição i
      - Eq. 3: probabilidade Pi(b) com Laplace smoothing
      - Eq. 5: log-likelihood  log L(x) = Σ log Pi(xi)
      - Eq. 6: cross-entropy   H(x;P)  = -(1/L) Σ log Pi(xi)
    """

    def __init__(self, alpha: float = 1.0, max_positions: int = 256):
        """
        alpha        : parâmetro de Laplace smoothing (α > 0)
        max_positions: número máximo de posições de byte consideradas
        """
        self.alpha         = alpha
        self.max_positions = max_positions
        self.counts_       = None   # shape: (max_positions, 256)
        self.probs_        = None   # shape: (max_positions, 256) — após fit()
        self.fitted_       = False

    def fit(self, payloads_hex: pd.Series):
        """
        Aprende a distribuição a partir de payloads benignas (hex strings).

        payloads_hex: Series de strings hexadecimais, ex: "de ad be ef ..."
        """
        # Inicializa contadores (posição x valor_byte)
        counts = np.zeros((self.max_positions, 256), dtype=np.float64)

        for hex_str in payloads_hex.dropna():
            try:
                raw = bytes.fromhex(hex_str)
            except ValueError:
                continue
            for i, b in enumerate(raw[:self.max_positions]):
                counts[i, b] += 1

        # Laplace smoothing: Pi(b) = (ci(b) + α) / (N_i + 256*α)
        totals = counts.sum(axis=1, keepdims=True)            # N_i por posição
        self.probs_ = (counts + self.alpha) / (totals + 256 * self.alpha)
        self.counts_ = counts
        self.fitted_ = True

    def log_likelihood(self, hex_str: str) -> float:
        """
        Calcula log L(x) = Σ_{i=1}^{L} log Pi(xi)   (Equação 5).
        Retorna 0.0 se o payload for vazio ou inválido.
        """
        if not self.fitted_ or not isinstance(hex_str, str) or len(hex_str) < 2:
            return 0.0
        try:
            raw = bytes.fromhex(hex_str)
        except ValueError:
            return 0.0
        if len(raw) == 0:
            return 0.0

        ll = 0.0
        for i, b in enumerate(raw[:self.max_positions]):
            p = self.probs_[i, b]
            ll += np.log(p + 1e-12)   # epsilon para estabilidade numérica
        return ll

    def cross_entropy(self, hex_str: str) -> float:
        """
        Calcula H(x;P) = -(1/L) Σ log Pi(xi)   (Equação 6).
        Normalizado pelo comprimento → independente do tamanho do payload.
        """
        if not self.fitted_ or not isinstance(hex_str, str) or len(hex_str) < 2:
            return 0.0
        try:
            raw = bytes.fromhex(hex_str)
        except ValueError:
            return 0.0
        L = len(raw)
        if L == 0:
            return 0.0

        ce = 0.0
        for i, b in enumerate(raw[:self.max_positions]):
            p = self.probs_[i, b]
            ce -= np.log(p + 1e-12)
        return ce / L


# ══════════════════════════════════════════════════════════════════════════════
# 2. Feature de mudança de payload (distância de Hamming)
# ══════════════════════════════════════════════════════════════════════════════

def hamming_distance(hex_a: str, hex_b: str) -> float:
    """
    Distância de Hamming bit-a-bit entre dois payloads (Equação 7).

    dH(b1, b2) = Σ wH(b1_i XOR b2_i)

    Implementado como contagem de bits '1' no XOR de cada byte.
    Retorna 0.0 se algum dos payloads for inválido ou vazio.
    """
    if not isinstance(hex_a, str) or not isinstance(hex_b, str):
        return 0.0
    try:
        raw_a = bytes.fromhex(hex_a)
        raw_b = bytes.fromhex(hex_b)
    except ValueError:
        return 0.0

    # Trunca ou estende para o menor comprimento
    L = min(len(raw_a), len(raw_b))
    if L == 0:
        return 0.0

    dist = 0
    for i in range(L):
        dist += bin(raw_a[i] ^ raw_b[i]).count("1")
    return float(dist)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Cálculo das features por fluxo
# ══════════════════════════════════════════════════════════════════════════════

def flow_key(row):
    """
    Chave de five-tuple para agrupar pacotes do mesmo fluxo.
    Segue a Seção 5.2: 'packets sharing the same source and destination
    endpoints (IP/port pairs) grouped and processed separately'.
    """
    return (
        str(row.get("src_ip", "")),
        str(row.get("dst_ip", "")),
        str(row.get("src_port", "")),
        str(row.get("dst_port", "")),
        str(row.get("transport", "")),
    )


def extract_features(df: pd.DataFrame,
                     someip_model: ByteDistributionModel,
                     tcpudp_model: ByteDistributionModel) -> pd.DataFrame:
    """
    Calcula as 9 features da Tabela 1 para cada pacote em df.

    Features calculadas:
      f01 : ip_time_interval         (delta_t entre pacotes consecutivos no fluxo)
      f02 : someip_likelihood        (log-likelihood do payload SOME/IP)
      f03 : tcpudp_likelihood        (log-likelihood do payload TCP/UDP — mesmo hex, modelo diferente)
      f04 : someip_entropy           (cross-entropy do payload SOME/IP)
      f05 : tcpudp_entropy           (cross-entropy TCP/UDP)
      f06 : someip_payload_changes   (Hamming distance payloads SOME/IP consecutivos)
      f07 : tcpudp_payload_changes   (Hamming distance payloads TCP/UDP consecutivos)
      f08 : ip_length_changes        (delta comprimento IP entre pacotes consecutivos)
      f09 : tcpudp_length_changes    (delta comprimento TCP/UDP entre pacotes consecutivos)

    Nota: o artigo mantém SOME/IP e SOME/IP-SD separados. Aqui usamos
    a coluna is_sd para distinguir, mas o modelo de bytes é compartilhado
    para simplificação — pode ser separado facilmente.
    """

    # Garante ordenação por timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Dicionários de estado por fluxo (armazenam o pacote anterior)
    prev_ts       = defaultdict(lambda: None)
    prev_ip_len   = defaultdict(lambda: None)
    prev_tl_len   = defaultdict(lambda: None)
    prev_payload  = defaultdict(lambda: None)

    records = []

    for _, row in df.iterrows():
        key    = flow_key(row)
        ts     = row.get("timestamp", 0.0)
        ip_len = row.get("ip_len", None)
        tl_len = row.get("transport_len", None)
        pld    = row.get("payload_hex", None)   # hex string do payload SOME/IP

        # ── f01: IP time interval ──────────────────────────────────────────────
        if prev_ts[key] is not None and ts is not None:
            f01 = float(ts) - float(prev_ts[key])
        else:
            f01 = 0.0

        # ── f02 & f04: SOME/IP likelihood e entropy ────────────────────────────
        f02 = someip_model.log_likelihood(pld)
        f04 = someip_model.cross_entropy(pld)

        # ── f03 & f05: TCP/UDP likelihood e entropy (mesmo hex, modelo diferente)
        f03 = tcpudp_model.log_likelihood(pld)
        f05 = tcpudp_model.cross_entropy(pld)

        # ── f06: SOME/IP payload changes (Hamming) ─────────────────────────────
        if prev_payload[key] is not None and pld is not None:
            f06 = hamming_distance(prev_payload[key], pld)
        else:
            f06 = 0.0

        # ── f07: TCP/UDP payload changes (Hamming — mesmo campo, contexto diferente)
        f07 = f06   # No dataset, o payload raw é o mesmo para ambas as camadas

        # ── f08: IP length changes ─────────────────────────────────────────────
        if prev_ip_len[key] is not None and ip_len is not None:
            f08 = float(ip_len) - float(prev_ip_len[key])
        else:
            f08 = 0.0

        # ── f09: TCP/UDP length changes ────────────────────────────────────────
        if prev_tl_len[key] is not None and tl_len is not None:
            f09 = float(tl_len) - float(prev_tl_len[key])
        else:
            f09 = 0.0

        # ── Atualiza estado do fluxo ───────────────────────────────────────────
        prev_ts[key]      = ts
        prev_ip_len[key]  = ip_len
        prev_tl_len[key]  = tl_len
        prev_payload[key] = pld

        # ── Monta registro de saída ────────────────────────────────────────────
        records.append({
            # Identificadores (para rastreabilidade)
            "timestamp":    ts,
            "src_ip":       row.get("src_ip"),
            "dst_ip":       row.get("dst_ip"),
            "src_port":     row.get("src_port"),
            "dst_port":     row.get("dst_port"),
            "transport":    row.get("transport"),
            "is_sd":        row.get("is_sd", False),
            "service_id":   row.get("service_id"),
            "method_id":    row.get("method_id"),
            "msg_type":     row.get("msg_type"),
            # 9 features comportamentais (Tabela 1)
            "f01_ip_time_interval":       f01,
            "f02_someip_likelihood":      f02,
            "f03_tcpudp_likelihood":      f03,
            "f04_someip_entropy":         f04,
            "f05_tcpudp_entropy":         f05,
            "f06_someip_payload_changes": f06,
            "f07_tcpudp_payload_changes": f07,
            "f08_ip_length_changes":      f08,
            "f09_tcpudp_length_changes":  f09,
            # Rótulo (0=normal, 1=ataque)
            "label_str":    row.get("label", "unknown"),
            "label":        0 if str(row.get("label", "")).lower() == "normal" else 1,
        })

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Normalização Min-Max (Equação 8)
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "f01_ip_time_interval",
    "f02_someip_likelihood",
    "f03_tcpudp_likelihood",
    "f04_someip_entropy",
    "f05_tcpudp_entropy",
    "f06_someip_payload_changes",
    "f07_tcpudp_payload_changes",
    "f08_ip_length_changes",
    "f09_tcpudp_length_changes",
]

def minmax_normalize(df_train: pd.DataFrame,
                     df_test:  pd.DataFrame = None):
    """
    Aplica normalização Min-Max (Equação 8):
      x' = (x - x_min) / (x_max - x_min)

    Os parâmetros são calculados APENAS no conjunto de treino
    e aplicados também no teste (sem vazamento de informação).

    Retorna (df_train_norm, df_test_norm, stats_dict)
    """
    stats = {}
    df_tr = df_train.copy()

    for col in FEATURE_COLS:
        x_min = df_tr[col].min()
        x_max = df_tr[col].max()
        stats[col] = {"min": x_min, "max": x_max}
        denom = x_max - x_min
        if denom == 0:
            df_tr[col + "_norm"] = 0.0
        else:
            df_tr[col + "_norm"] = (df_tr[col] - x_min) / denom

    df_te = None
    if df_test is not None:
        df_te = df_test.copy()
        for col in FEATURE_COLS:
            x_min = stats[col]["min"]
            x_max = stats[col]["max"]
            denom = x_max - x_min
            if denom == 0:
                df_te[col + "_norm"] = 0.0
            else:
                df_te[col + "_norm"] = (df_te[col] - x_min) / denom
                # Clipa para [0, 1] (valores fora do range de treino)
                df_te[col + "_norm"] = df_te[col + "_norm"].clip(0.0, 1.0)

    return df_tr, df_te, stats


# ══════════════════════════════════════════════════════════════════════════════
# 5. Pipeline principal
# ══════════════════════════════════════════════════════════════════════════════

def run_feature_extraction(parsed_csv: str, output_dir: str):
    """
    Pipeline completo de extração de features.

    1. Lê o CSV parseado
    2. Treina modelos de distribuição de bytes no tráfego benigno
    3. Calcula as 9 features para todos os pacotes
    4. Normaliza Min-Max
    5. Salva CSVs de treino e teste (split 50/50 estratificado)
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Leitura ────────────────────────────────────────────────────────────────
    print("[1/5] Lendo CSV parseado...")
    df = pd.read_csv(parsed_csv, low_memory=False)
    print(f"      {len(df):,} registros  |  colunas: {list(df.columns)[:8]}...")

    # ── Separa tráfego benigno para treinar o modelo de bytes ──────────────────
    print("\n[2/5] Treinando modelos de distribuição de bytes (tráfego benigno)...")
    benign = df[df["label"].str.lower() == "normal"]
    print(f"      Amostras benignas: {len(benign):,}")

    someip_model = ByteDistributionModel(alpha=1.0)
    tcpudp_model = ByteDistributionModel(alpha=1.0)

    # Treina em subconjunto se for muito grande (para velocidade)
    sample_size = min(50_000, len(benign))
    sample      = benign.sample(n=sample_size, random_state=42)

    someip_model.fit(sample["payload_hex"])
    tcpudp_model.fit(sample["payload_hex"])   # modelo separado, mesmos dados
    print(f"      Modelos treinados em {sample_size:,} amostras.")

    # ── Extração de features ───────────────────────────────────────────────────
    print("\n[3/5] Extraindo 9 features comportamentais (Tabela 1)...")
    print("      (pode levar alguns minutos para datasets grandes)")
    features_df = extract_features(df, someip_model, tcpudp_model)
    print(f"      Features extraídas: {len(features_df):,} amostras")

    # Distribuição de classes
    vc = features_df["label"].value_counts()
    print(f"      Normal: {vc.get(0,0):,}  |  Ataque: {vc.get(1,0):,}  "
          f"({100*vc.get(1,0)/max(len(features_df),1):.1f}% ataques)")

    # ── Split estratificado 50/50 (seguindo o artigo) ──────────────────────────
    print("\n[4/5] Dividindo treino/teste (50/50 estratificado)...")
    from sklearn.model_selection import train_test_split

    X = features_df.drop(columns=["label"])
    y = features_df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=42
    )

    df_train = X_train.copy(); df_train["label"] = y_train.values
    df_test  = X_test.copy();  df_test["label"]  = y_test.values

    print(f"      Treino: {len(df_train):,}  |  Teste: {len(df_test):,}")

    # ── Normalização Min-Max ───────────────────────────────────────────────────
    print("\n[5/5] Aplicando normalização Min-Max (Equação 8)...")
    df_train_norm, df_test_norm, stats = minmax_normalize(df_train, df_test)

    # ── Salva resultados ───────────────────────────────────────────────────────
    train_path = out_dir / "train_features.csv"
    test_path  = out_dir / "test_features.csv"
    raw_path   = out_dir / "all_features_raw.csv"

    df_train_norm.to_csv(train_path, index=False)
    df_test_norm.to_csv(test_path,   index=False)
    features_df.to_csv(raw_path,     index=False)

    print(f"\n{'='*60}")
    print(f"CONCLUÍDO")
    print(f"  Treino (normalizado): {train_path}")
    print(f"  Teste  (normalizado): {test_path}")
    print(f"  Features brutas     : {raw_path}")
    print(f"\nColunas de feature normalizadas:")
    for col in FEATURE_COLS:
        col_n = col + "_norm"
        print(f"  {col_n:<40s}  "
              f"min={stats[col]['min']:.4f}  max={stats[col]['max']:.4f}")

    return str(train_path), str(test_path)


# ══════════════════════════════════════════════════════════════════════════════
# Ponto de entrada
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Extração de features comportamentais SOME/IP (Kim et al. 2026)"
    )
    ap.add_argument("--parsed-csv", default="data/parsed_packets.csv",
                    help="CSV gerado pelo script 01 (padrão: data/parsed_packets.csv)")
    ap.add_argument("--output-dir", default="data/",
                    help="Pasta de saída para CSVs de features (padrão: data/)")
    args = ap.parse_args()

    run_feature_extraction(args.parsed_csv, args.output_dir)
