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

def _delta(new_val, prev_val) -> float:
    """Retorna a diferença entre dois valores consecutivos, ou 0.0 se algum for None."""
    if prev_val is None or new_val is None:
        return 0.0
    return float(new_val) - float(prev_val)


def _payload_change(prev_pld: str | None, pld: str | None) -> float:
    """Retorna a distância de Hamming entre dois payloads, ou 0.0 se algum for None."""
    if prev_pld is None or pld is None:
        return 0.0
    return hamming_distance(prev_pld, pld)


def make_flow_state() -> dict:
    """Cria dicionário de estado por fluxo para uso entre chunks.

    Returns:
        Dicionário com quatro defaultdicts, um por campo de estado.
        Deve ser criado uma vez e passado para cada chamada de
        ``extract_features`` para preservar continuidade entre chunks.
    """
    return {
        "prev_ts":      defaultdict(lambda: None),
        "prev_ip_len":  defaultdict(lambda: None),
        "prev_tl_len":  defaultdict(lambda: None),
        "prev_payload": defaultdict(lambda: None),
    }


def extract_features(df: pd.DataFrame,
                     someip_model: ByteDistributionModel,
                     tcpudp_model: ByteDistributionModel,
                     flow_state: dict = None) -> pd.DataFrame:
    """Calcula as 9 features da Tabela 1 para cada pacote em df.

    Processa um chunk do CSV preservando o estado de fluxo entre chamadas
    sucessivas. Para processar todo o dataset sem estourar memória, chame
    esta função em loop passando sempre o mesmo ``flow_state``.

    Features calculadas:

    - f01: ip_time_interval         — delta_t entre pacotes consecutivos no fluxo
    - f02: someip_likelihood        — log-likelihood do payload SOME/IP
    - f03: tcpudp_likelihood        — log-likelihood do payload TCP/UDP
    - f04: someip_entropy           — cross-entropy do payload SOME/IP
    - f05: tcpudp_entropy           — cross-entropy TCP/UDP
    - f06: someip_payload_changes   — Hamming distance payloads SOME/IP consecutivos
    - f07: tcpudp_payload_changes   — Hamming distance payloads TCP/UDP consecutivos
    - f08: ip_length_changes        — delta comprimento IP entre pacotes consecutivos
    - f09: tcpudp_length_changes    — delta comprimento TCP/UDP entre pacotes consecutivos

    Args:
        df: Chunk do CSV parseado (saída do script 01).
        someip_model: Modelo de distribuição de bytes treinado no tráfego SOME/IP benigno.
        tcpudp_model: Modelo de distribuição de bytes treinado no tráfego TCP/UDP benigno.
        flow_state: Dicionário criado por ``make_flow_state()``. Se ``None``, cria
            um estado novo (adequado apenas para chunks isolados).

    Returns:
        DataFrame com as 9 features e colunas de identificação para este chunk.
    """
    if flow_state is None:
        flow_state = make_flow_state()

    prev_ts      = flow_state["prev_ts"]
    prev_ip_len  = flow_state["prev_ip_len"]
    prev_tl_len  = flow_state["prev_tl_len"]
    prev_payload = flow_state["prev_payload"]

    df = df.sort_values("timestamp").reset_index(drop=True)

    records = []

    for row in df.itertuples(index=False):
        key = (str(row.src_ip), str(row.dst_ip),
               str(row.src_port), str(row.dst_port), str(row.transport))
        ts     = row.timestamp
        ip_len = row.ip_len
        tl_len = row.transport_len
        pld    = row.payload_hex if isinstance(row.payload_hex, str) else None

        f01 = _delta(ts, prev_ts[key])
        f02 = someip_model.log_likelihood(pld)
        f03 = tcpudp_model.log_likelihood(pld)
        f04 = someip_model.cross_entropy(pld)
        f05 = tcpudp_model.cross_entropy(pld)
        f06 = _payload_change(prev_payload[key], pld)
        f07 = f06
        f08 = _delta(ip_len, prev_ip_len[key])
        f09 = _delta(tl_len, prev_tl_len[key])

        prev_ts[key]      = ts
        prev_ip_len[key]  = ip_len
        prev_tl_len[key]  = tl_len
        prev_payload[key] = pld

        label_raw = row.label if hasattr(row, "label") else "unknown"
        records.append({
            "timestamp":    ts,
            "src_ip":       row.src_ip,
            "dst_ip":       row.dst_ip,
            "src_port":     row.src_port,
            "dst_port":     row.dst_port,
            "transport":    row.transport,
            "is_sd":        getattr(row, "is_sd", False),
            "service_id":   getattr(row, "service_id", None),
            "method_id":    getattr(row, "method_id", None),
            "msg_type":     getattr(row, "msg_type", None),
            "f01_ip_time_interval":       f01,
            "f02_someip_likelihood":      f02,
            "f03_tcpudp_likelihood":      f03,
            "f04_someip_entropy":         f04,
            "f05_tcpudp_entropy":         f05,
            "f06_someip_payload_changes": f06,
            "f07_tcpudp_payload_changes": f07,
            "f08_ip_length_changes":      f08,
            "f09_tcpudp_length_changes":  f09,
            "label_str":    label_raw,
            "label":        0 if str(label_raw).lower() == "normal" else 1,
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

def run_feature_extraction(parsed_csv: str, output_dir: str,
                           chunk_size: int = 500_000):
    """Pipeline completo de extração de features com processamento em chunks.

    Processa o CSV em blocos de ``chunk_size`` linhas para manter o uso de
    memória limitado (~1-2 GB independente do tamanho do dataset). O estado
    de fluxo é preservado entre chunks para que features dependentes de
    sequência (time interval, payload changes) sejam calculadas corretamente.

    Etapas:

    1. Treina modelos de bytes em amostra do tráfego benigno (primeira passagem).
    2. Extrai as 9 features chunk a chunk, gravando diretamente no CSV de saída.
    3. Lê os rótulos do CSV gerado para montar o split estratificado 50/50.
    4. Relê o CSV e distribui cada linha em treino ou teste usando máscara booleana.
    5. Aplica normalização Min-Max (parâmetros calculados apenas no treino).

    Args:
        parsed_csv: Caminho para o CSV gerado pelo script 01.
        output_dir: Pasta onde os CSVs de features serão salvos.
        chunk_size: Número de linhas por chunk (padrão: 500.000 ~= 200 MB RAM).

    Returns:
        Tupla ``(train_path, test_path)`` com os caminhos dos CSVs gerados.
    """
    from sklearn.model_selection import train_test_split

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path   = out_dir / "all_features_raw.csv"
    train_path = out_dir / "train_features.csv"
    test_path  = out_dir / "test_features.csv"

    # ── [1/5] Treina modelos de bytes em amostra benigna (sem carregar tudo) ───
    print("[1/5] Treinando modelos de distribuição de bytes (tráfego benigno)...")
    benign_payloads = []
    for chunk in pd.read_csv(parsed_csv, usecols=["label", "payload_hex"],
                              chunksize=chunk_size, low_memory=False):
        sub = chunk[chunk["label"].str.lower() == "normal"]["payload_hex"].dropna()
        benign_payloads.append(sub)
        if sum(len(b) for b in benign_payloads) >= 50_000:
            break

    sample = pd.concat(benign_payloads).head(50_000)
    someip_model = ByteDistributionModel(alpha=1.0)
    tcpudp_model = ByteDistributionModel(alpha=1.0)
    someip_model.fit(sample)
    tcpudp_model.fit(sample)
    print(f"      Modelos treinados em {len(sample):,} amostras benignas.")

    # ── [2/5] Extrai features chunk a chunk, grava CSV incremental ─────────────
    print("\n[2/5] Extraindo 9 features comportamentais (Tabela 1)...")
    print(f"      Chunk size: {chunk_size:,} linhas  |  arquivo: {raw_path.name}")
    flow_state  = make_flow_state()
    first_write = True
    n_total     = 0

    for chunk in pd.read_csv(parsed_csv, chunksize=chunk_size, low_memory=False):
        feat_chunk = extract_features(chunk, someip_model, tcpudp_model, flow_state)
        feat_chunk.to_csv(raw_path, mode="a", header=first_write, index=False)
        first_write = False
        n_total += len(feat_chunk)
        print(f"      ... {n_total:,} features extraídas")

    print(f"      Total: {n_total:,} amostras")

    # ── [3/5] Lê rótulos para montar split estratificado ───────────────────────
    print("\n[3/5] Montando split estratificado 50/50...")
    labels = pd.concat(
        chunk["label"]
        for chunk in pd.read_csv(raw_path, usecols=["label"], chunksize=chunk_size)
    ).values

    idx = np.arange(len(labels))
    train_idx, _ = train_test_split(idx, test_size=0.5, stratify=labels, random_state=42)
    is_train = np.zeros(len(labels), dtype=bool)
    is_train[train_idx] = True
    print(f"      Treino: {is_train.sum():,}  |  Teste: {(~is_train).sum():,}")

    # ── [4/5] Distribui linhas entre treino e teste ────────────────────────────
    print("\n[4/5] Gravando arquivos de treino e teste...")
    row_num = 0
    first_tr = first_te = True

    for chunk in pd.read_csv(raw_path, chunksize=chunk_size):
        chunk_mask = is_train[row_num: row_num + len(chunk)]
        chunk[chunk_mask].to_csv(train_path,  mode="a", header=first_tr, index=False)
        chunk[~chunk_mask].to_csv(test_path,  mode="a", header=first_te, index=False)
        first_tr = first_te = False
        row_num += len(chunk)

    # ── [5/5] Normalização Min-Max ─────────────────────────────────────────────
    print("\n[5/5] Aplicando normalização Min-Max (Equação 8)...")

    # Calcula min/max apenas no treino
    stats = {col: {"min": float("inf"), "max": float("-inf")} for col in FEATURE_COLS}
    for chunk in pd.read_csv(train_path, usecols=FEATURE_COLS, chunksize=chunk_size):
        for col in FEATURE_COLS:
            stats[col]["min"] = min(stats[col]["min"], chunk[col].min())
            stats[col]["max"] = max(stats[col]["max"], chunk[col].max())

    # Aplica normalização gravando arquivos finais normalizados
    for src, dst, label in [(train_path, out_dir / "train_features_norm.csv", "treino"),
                             (test_path,  out_dir / "test_features_norm.csv",  "teste")]:
        first = True
        for chunk in pd.read_csv(src, chunksize=chunk_size):
            for col in FEATURE_COLS:
                denom = stats[col]["max"] - stats[col]["min"]
                if denom == 0:
                    chunk[col + "_norm"] = 0.0
                else:
                    chunk[col + "_norm"] = ((chunk[col] - stats[col]["min"]) / denom).clip(0.0, 1.0)
            chunk.to_csv(dst, mode="a", header=first, index=False)
            first = False
        # Substitui arquivo original pelo normalizado
        dst.replace(src)

    print(f"\n{'='*60}")
    print("CONCLUIDO")
    print(f"  Treino (normalizado): {train_path}")
    print(f"  Teste  (normalizado): {test_path}")
    print(f"  Features brutas     : {raw_path}")
    print("\n  Min/Max por feature (calculado no treino):")
    for col in FEATURE_COLS:
        print(f"    {col:<40s}  min={stats[col]['min']:.4f}  max={stats[col]['max']:.4f}")

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
