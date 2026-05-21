"""
SOME/IP IDS - Etapa 2: Extração de Features Comportamentais
=============================================================
Reprodução de Kim et al. (2026) - Seções 5.2-5.3

Implementa as 9 features comportamentais da Tabela 1 do artigo,
calculadas por fluxo (five-tuple) sobre o CSV gerado pela Etapa 1.
Processamento em chunks para suportar datasets de 7+ milhões de registros.

Features produzidas:

    f01  IP time interval          — delta_t entre pacotes consecutivos no fluxo
    f02  SOME/IP likelihood        — log-likelihood do payload SOME/IP vs modelo benigno
    f03  TCP/UDP likelihood        — log-likelihood do payload de transporte vs modelo benigno
    f04  SOME/IP entropy           — cross-entropy do payload SOME/IP vs modelo benigno
    f05  TCP/UDP entropy           — cross-entropy do payload de transporte vs modelo benigno
    f06  SOME/IP payload changes   — distância de Hamming vs payload SOME/IP anterior no fluxo
    f07  TCP/UDP payload changes   — distância de Hamming vs payload TCP/UDP anterior no fluxo
    f08  IP length changes         — delta ip_len entre pacotes consecutivos no fluxo
    f09  TCP/UDP length changes    — delta transport_len entre pacotes consecutivos no fluxo

Nota: para as 12 features do artigo (com SOME/IP-SD separado), use 04_pcap_to_npy.ipynb.

Entrada:
    parsed_packets.csv  — saída do script 01 (requer colunas someip_payload_hex,
                          transport_payload_hex, is_sd)

Saída:
    all_features_raw.csv        — todas as amostras sem normalização
    train_features.csv          — treino (50%, normalizado Min-Max)
    test_features.csv           — teste  (50%, normalizado Min-Max)

Referência:
    Kim et al. (2026). XGBoost-Based Anomaly Detection Framework for SOME/IP
    in In-Vehicle Networks. Systems, 14(2), 196.

Uso:
    python 02_extract_features.py --parsed-csv data/parsed_packets.csv --output-dir data/
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict


# ══════════════════════════════════════════════════════════════════════════════
# 1. Modelo de distribuição de bytes (likelihood e entropy)
# ══════════════════════════════════════════════════════════════════════════════

class ByteDistributionModel:
    """Distribuição de bytes por posição aprendida sobre tráfego benigno.

    Implementa as Equações 2-6 do artigo usando Laplace smoothing.
    Treinado uma vez sobre payloads benignos; usado para avaliar
    o desvio de novos payloads em relação ao padrão aprendido.

    Args:
        alpha:        Laplace smoothing — evita probabilidades zero (> 0).
        max_positions: Número máximo de posições de byte analisadas.
    """

    def __init__(self, alpha: float = 1.0, max_positions: int = 256):
        self.alpha         = alpha
        self.max_positions = max_positions
        self._log_probs    = None   # (max_positions, 256) — cacheado após fit

    def fit(self, payloads_hex: pd.Series) -> None:
        """Aprende a distribuição de bytes a partir de payloads benignas (Eq. 2-3).

        Args:
            payloads_hex: Series de strings hexadecimais de payloads benignos.
        """
        counts = np.zeros((self.max_positions, 256), dtype=np.float64)
        for h in payloads_hex.dropna():
            try:
                raw = bytes.fromhex(str(h))
            except ValueError:
                continue
            for i, b in enumerate(raw[:self.max_positions]):
                counts[i, b] += 1
        totals = counts.sum(axis=1, keepdims=True)
        probs  = (counts + self.alpha) / (totals + 256 * self.alpha)
        self._log_probs = np.log(probs + 1e-12)   # pré-computado para eficiência

    def log_likelihood(self, hex_str: str) -> float:
        """Log-likelihood do payload (Eq. 5): sum log P_i(x_i).

        Valor mais negativo → payload mais improvável (mais anômalo).
        Retorna 0.0 se o modelo não foi treinado ou o payload é inválido.
        """
        if self._log_probs is None or not isinstance(hex_str, str) or len(hex_str) < 2:
            return 0.0
        try:
            raw = bytes.fromhex(hex_str)
        except ValueError:
            return 0.0
        if not raw:
            return 0.0
        b = np.frombuffer(raw[:self.max_positions], dtype=np.uint8)
        return float(self._log_probs[np.arange(len(b)), b].sum())

    def cross_entropy(self, hex_str: str) -> float:
        """Cross-entropy normalizada pelo comprimento (Eq. 6): -(1/L) sum log P_i(x_i).

        Normalização por L torna a métrica comparável entre payloads de tamanhos diferentes.
        Retorna 0.0 se o modelo não foi treinado ou o payload é inválido.
        """
        if self._log_probs is None or not isinstance(hex_str, str) or len(hex_str) < 2:
            return 0.0
        try:
            raw = bytes.fromhex(hex_str)
        except ValueError:
            return 0.0
        L = len(raw)
        if L == 0:
            return 0.0
        b = np.frombuffer(raw[:self.max_positions], dtype=np.uint8)
        return float(-self._log_probs[np.arange(len(b)), b].sum() / L)

    def log_likelihood_batch(self, series: pd.Series) -> np.ndarray:
        """Versão vetorizada de log_likelihood para uma Series inteira.

        Elimina o loop Python por byte usando numpy fancy indexing,
        resultando em 5-15x menos overhead por elemento.

        Args:
            series: Series de strings hexadecimais.

        Returns:
            Array (N,) com log-likelihood por elemento.
        """
        results = np.zeros(len(series), dtype=np.float64)
        if self._log_probs is None:
            return results
        for i, h in enumerate(series):
            if not isinstance(h, str) or len(h) < 2:
                continue
            try:
                raw = bytes.fromhex(h)
            except ValueError:
                continue
            if not raw:
                continue
            b = np.frombuffer(raw[:self.max_positions], dtype=np.uint8)
            results[i] = self._log_probs[np.arange(len(b)), b].sum()
        return results

    def cross_entropy_batch(self, series: pd.Series) -> np.ndarray:
        """Versão vetorizada de cross_entropy para uma Series inteira.

        Args:
            series: Series de strings hexadecimais.

        Returns:
            Array (N,) com cross-entropy por elemento.
        """
        results = np.zeros(len(series), dtype=np.float64)
        if self._log_probs is None:
            return results
        for i, h in enumerate(series):
            if not isinstance(h, str) or len(h) < 2:
                continue
            try:
                raw = bytes.fromhex(h)
            except ValueError:
                continue
            L = len(raw)
            if L == 0:
                continue
            b = np.frombuffer(raw[:self.max_positions], dtype=np.uint8)
            results[i] = -self._log_probs[np.arange(len(b)), b].sum() / L
        return results


# ══════════════════════════════════════════════════════════════════════════════
# 2. Distância de Hamming entre payloads (payload changes)
# ══════════════════════════════════════════════════════════════════════════════

def hamming_distance(hex_a: str, hex_b: str) -> float:
    """Distância de Hamming bit-a-bit entre dois payloads (Eq. 7).

    Compara apenas os primeiros min(len_a, len_b) bytes.
    Retorna 0.0 se qualquer argumento for inválido ou ausente.

    Args:
        hex_a: Primeiro payload em hexadecimal.
        hex_b: Segundo payload em hexadecimal.

    Returns:
        Número total de bits diferentes (>= 0).
    """
    if not isinstance(hex_a, str) or not isinstance(hex_b, str):
        return 0.0
    try:
        raw_a = bytes.fromhex(hex_a)
        raw_b = bytes.fromhex(hex_b)
    except ValueError:
        return 0.0
    L = min(len(raw_a), len(raw_b))
    if L == 0:
        return 0.0
    a = np.frombuffer(raw_a[:L], dtype=np.uint8)
    b = np.frombuffer(raw_b[:L], dtype=np.uint8)
    # fração de bits diferentes [0, 1] — normalizada por comprimento
    return float(np.unpackbits(np.bitwise_xor(a, b)).sum()) / (8 * L)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Estado de fluxo e extração de features por chunk
# ══════════════════════════════════════════════════════════════════════════════

def make_flow_state() -> dict:
    """Cria o dicionário de estado por fluxo (five-tuple) entre chunks.

    Returns:
        Dict com cinco defaultdicts: timestamp, ip_len, transport_len,
        payload SOME/IP anterior e payload TCP/UDP anterior por fluxo.
    """
    return {
        "prev_ts":          defaultdict(lambda: None),
        "prev_ip_len":      defaultdict(lambda: None),
        "prev_tl_len":      defaultdict(lambda: None),
        "prev_someip_pld":  defaultdict(lambda: None),   # payload SOME/IP anterior
        "prev_tcpudp_pld":  defaultdict(lambda: None),   # payload TCP/UDP anterior
    }


def extract_features(df: pd.DataFrame,
                     someip_model: ByteDistributionModel,
                     tcpudp_model: ByteDistributionModel,
                     flow_state: dict = None) -> pd.DataFrame:
    """Calcula as 9 features da Tabela 1 para um chunk do CSV parseado.

    Combina computação em batch para likelihood/entropy (f02-f05) com
    loop sequencial para features dependentes de estado de fluxo (f01, f06-f09).

    Args:
        df:           Chunk do CSV de pacotes parseados (saída do script 01).
        someip_model: Modelo treinado sobre payloads SOME/IP benignas.
        tcpudp_model: Modelo treinado sobre payloads TCP/UDP benignas.
        flow_state:   Estado de fluxo criado por make_flow_state(). Se None,
                      cria estado novo (correto apenas para chunks isolados).

    Returns:
        DataFrame com as 9 features e metadados de identificação.
    """
    if flow_state is None:
        flow_state = make_flow_state()

    df = df.sort_values("timestamp").reset_index(drop=True)
    n  = len(df)

    # ── Batch: f02-f05 (likelihood e entropy, sem dependência de fluxo) ───────
    si_hex = df["someip_payload_hex"] if "someip_payload_hex" in df.columns \
             else df.get("payload_hex", pd.Series([""] * n))
    tu_hex = df["transport_payload_hex"] if "transport_payload_hex" in df.columns \
             else si_hex   # fallback: usa mesmo payload se coluna ausente

    f02_arr = someip_model.log_likelihood_batch(si_hex)
    f03_arr = tcpudp_model.log_likelihood_batch(tu_hex)
    f04_arr = someip_model.cross_entropy_batch(si_hex)
    f05_arr = tcpudp_model.cross_entropy_batch(tu_hex)

    # ── Prepara arrays para o loop sequencial ─────────────────────────────────
    prev_ts         = flow_state["prev_ts"]
    prev_ip_len     = flow_state["prev_ip_len"]
    prev_tl_len     = flow_state["prev_tl_len"]
    prev_someip_pld = flow_state["prev_someip_pld"]
    prev_tcpudp_pld = flow_state["prev_tcpudp_pld"]

    ts_vals    = df["timestamp"].values
    ip_vals    = df["ip_len"].values
    tl_vals    = df["transport_len"].values
    si_vals    = si_hex.values
    tu_vals    = tu_hex.values
    src_ips    = df["src_ip"].astype(str).values
    dst_ips    = df["dst_ip"].astype(str).values
    src_ports  = df["src_port"].astype(str).values
    dst_ports  = df["dst_port"].astype(str).values
    transports = df["transport"].astype(str).values

    f01 = np.zeros(n)
    f06 = np.zeros(n)   # SOME/IP payload changes
    f07 = np.zeros(n)   # TCP/UDP payload changes  (diferente de f06)
    f08 = np.zeros(n)
    f09 = np.zeros(n)

    # ── Loop sequencial: features dependentes de estado de fluxo ─────────────
    for i in range(n):
        key = (src_ips[i], dst_ips[i], src_ports[i], dst_ports[i], transports[i])

        ts    = float(ts_vals[i])
        ip_l  = float(ip_vals[i])  if not np.isnan(ip_vals[i])  else None
        tl_l  = float(tl_vals[i])  if not np.isnan(tl_vals[i])  else None
        si_h  = si_vals[i] if isinstance(si_vals[i], str) else None
        tu_h  = tu_vals[i] if isinstance(tu_vals[i], str) else None

        f01[i] = (ts - prev_ts[key])       if prev_ts[key]     is not None else 0.0
        f06[i] = hamming_distance(prev_someip_pld[key], si_h)
        f07[i] = hamming_distance(prev_tcpudp_pld[key], tu_h)
        f08[i] = (ip_l - prev_ip_len[key]) if prev_ip_len[key] is not None and ip_l is not None else 0.0
        f09[i] = (tl_l - prev_tl_len[key]) if prev_tl_len[key] is not None and tl_l is not None else 0.0

        prev_ts[key]         = ts
        prev_ip_len[key]     = ip_l
        prev_tl_len[key]     = tl_l
        prev_someip_pld[key] = si_h
        prev_tcpudp_pld[key] = tu_h

    # ── Monta DataFrame de saída ───────────────────────────────────────────────
    label_raw = df["label"] if "label" in df.columns else pd.Series(["unknown"] * n)
    out = pd.DataFrame({
        "timestamp":  ts_vals,
        "src_ip":     df["src_ip"].values,
        "dst_ip":     df["dst_ip"].values,
        "src_port":   df["src_port"].values,
        "dst_port":   df["dst_port"].values,
        "transport":  df["transport"].values,
        "is_sd":      df["is_sd"].values if "is_sd" in df.columns else False,
        "f01_ip_time_interval":       f01,
        "f02_someip_likelihood":      f02_arr,
        "f03_tcpudp_likelihood":      f03_arr,
        "f04_someip_entropy":         f04_arr,
        "f05_tcpudp_entropy":         f05_arr,
        "f06_someip_payload_changes": f06,
        "f07_tcpudp_payload_changes": f07,
        "f08_ip_length_changes":      f08,
        "f09_tcpudp_length_changes":  f09,
        "label_str": label_raw.values,
        "label":     (label_raw.str.lower() != "normal").astype(int).values,
    })
    return out


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
                     df_test: pd.DataFrame = None) -> tuple:
    """Aplica normalização Min-Max nas 9 features (Eq. 8): x' = (x - min) / (max - min).

    Parâmetros calculados exclusivamente no treino e aplicados ao teste
    para evitar data leakage. Valores do teste fora do intervalo de treino
    são clipados para [0, 1].

    Args:
        df_train: DataFrame de treino com as colunas de FEATURE_COLS.
        df_test:  DataFrame de teste opcional.

    Returns:
        (df_train_norm, df_test_norm, stats) onde stats = {col: {min, max}}.
    """
    stats  = {}
    df_tr  = df_train.copy()
    for col in FEATURE_COLS:
        lo, hi = df_tr[col].min(), df_tr[col].max()
        stats[col] = {"min": lo, "max": hi}
        denom = hi - lo
        df_tr[col + "_norm"] = 0.0 if denom == 0 else (df_tr[col] - lo) / denom

    df_te = None
    if df_test is not None:
        df_te = df_test.copy()
        for col in FEATURE_COLS:
            lo, hi = stats[col]["min"], stats[col]["max"]
            denom  = hi - lo
            if denom == 0:
                df_te[col + "_norm"] = 0.0
            else:
                df_te[col + "_norm"] = ((df_te[col] - lo) / denom).clip(0.0, 1.0)
    return df_tr, df_te, stats


# ══════════════════════════════════════════════════════════════════════════════
# 5. Pipeline principal
# ══════════════════════════════════════════════════════════════════════════════

def run_feature_extraction(parsed_csv: str, output_dir: str,
                           chunk_size: int = 500_000):
    """Pipeline completo de extração de features com processamento em chunks.

    Etapas:
        1. Treina modelos de bytes sobre amostras benignas (someip_model
           sobre someip_payload_hex; tcpudp_model sobre transport_payload_hex).
        2. Extrai as 9 features chunk a chunk, preservando estado de fluxo.
        3. Monta split estratificado 50/50 por label.
        4. Aplica normalização Min-Max (parâmetros do treino).

    Args:
        parsed_csv: CSV gerado pelo script 01 (deve conter someip_payload_hex
                    e transport_payload_hex).
        output_dir: Pasta de saída.
        chunk_size: Linhas por chunk (reduzir se a RAM for limitada).

    Returns:
        (train_path, test_path) — caminhos dos CSVs normalizados.
    """
    from sklearn.model_selection import train_test_split

    out_dir    = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path   = out_dir / "all_features_raw.csv"
    train_path = out_dir / "train_features.csv"
    test_path  = out_dir / "test_features.csv"

    # ── [1/5] Treina modelos de bytes sobre tráfego benigno ──────────────────
    print("[1/5] Treinando modelos de distribuição de bytes...")
    si_payloads, tu_payloads = [], []
    cols_needed = ["label", "someip_payload_hex", "transport_payload_hex", "is_sd"]
    for chunk in pd.read_csv(parsed_csv, usecols=cols_needed,
                             chunksize=chunk_size, low_memory=False):
        b = chunk[chunk["label"].str.lower() == "normal"]
        si_payloads.append(b["someip_payload_hex"].dropna())
        tu_payloads.append(b["transport_payload_hex"].dropna())
        if sum(len(x) for x in si_payloads) >= 50_000:
            break

    si_sample = pd.concat(si_payloads).head(50_000)
    tu_sample = pd.concat(tu_payloads).head(50_000)

    # Modelo SOME/IP: treinado sobre payloads do protocolo (após header 16B)
    someip_model = ByteDistributionModel(alpha=1.0)
    someip_model.fit(si_sample)

    # Modelo TCP/UDP: treinado sobre payloads de transporte (header + dados)
    tcpudp_model = ByteDistributionModel(alpha=1.0)
    tcpudp_model.fit(tu_sample)

    print(f"  someip_model: {len(si_sample):,} amostras benignas SOME/IP")
    print(f"  tcpudp_model: {len(tu_sample):,} amostras benignas TCP/UDP")

    # ── [2/5] Extração chunk a chunk ─────────────────────────────────────────
    print("\n[2/5] Extraindo 9 features...")
    flow_state  = make_flow_state()
    first_write = True
    n_total     = 0

    for chunk in pd.read_csv(parsed_csv, chunksize=chunk_size, low_memory=False):
        feat = extract_features(chunk, someip_model, tcpudp_model, flow_state)
        feat.to_csv(raw_path, mode="a", header=first_write, index=False)
        first_write = False
        n_total += len(feat)
        print(f"  ... {n_total:,} features extraídas")

    print(f"  Total: {n_total:,} amostras")

    # ── [3/5] Split estratificado 50/50 ──────────────────────────────────────
    print("\n[3/5] Montando split estratificado 50/50...")
    labels = pd.concat(
        c["label"] for c in pd.read_csv(raw_path, usecols=["label"], chunksize=chunk_size)
    ).values

    idx = np.arange(len(labels))
    train_idx, _ = train_test_split(idx, test_size=0.5, stratify=labels, random_state=42)
    is_train      = np.zeros(len(labels), dtype=bool)
    is_train[train_idx] = True
    print(f"  Treino: {is_train.sum():,}  |  Teste: {(~is_train).sum():,}")

    # ── [4/5] Distribui treino/teste ─────────────────────────────────────────
    print("\n[4/5] Gravando arquivos treino/teste...")
    row_num = 0
    first_tr = first_te = True
    for chunk in pd.read_csv(raw_path, chunksize=chunk_size):
        mask = is_train[row_num: row_num + len(chunk)]
        chunk[mask].to_csv(train_path, mode="a", header=first_tr, index=False)
        chunk[~mask].to_csv(test_path,  mode="a", header=first_te, index=False)
        first_tr = first_te = False
        row_num += len(chunk)

    # ── [5/5] Normalização Min-Max ────────────────────────────────────────────
    print("\n[5/5] Normalizando (Min-Max, parâmetros do treino)...")
    stats = {col: {"min": float("inf"), "max": float("-inf")} for col in FEATURE_COLS}
    for chunk in pd.read_csv(train_path, usecols=FEATURE_COLS, chunksize=chunk_size):
        for col in FEATURE_COLS:
            stats[col]["min"] = min(stats[col]["min"], chunk[col].min())
            stats[col]["max"] = max(stats[col]["max"], chunk[col].max())

    for src, dst, name in [(train_path, out_dir / "train_norm.csv", "treino"),
                            (test_path,  out_dir / "test_norm.csv",  "teste")]:
        first = True
        for chunk in pd.read_csv(src, chunksize=chunk_size):
            for col in FEATURE_COLS:
                lo, hi = stats[col]["min"], stats[col]["max"]
                denom  = hi - lo
                chunk[col + "_norm"] = 0.0 if denom == 0 else \
                    ((chunk[col] - lo) / denom).clip(0.0, 1.0)
            chunk.to_csv(dst, mode="a", header=first, index=False)
            first = False
        dst.replace(src)

    print(f"\n{'='*60}")
    print("CONCLUÍDO")
    print(f"  Treino: {train_path}")
    print(f"  Teste:  {test_path}")
    print(f"  Bruto:  {raw_path}")
    print("\n  Estatísticas por feature (calculadas no treino):")
    for col in FEATURE_COLS:
        print(f"    {col:<40}  min={stats[col]['min']:.4f}  max={stats[col]['max']:.4f}")

    return str(train_path), str(test_path)


# ══════════════════════════════════════════════════════════════════════════════
# Ponto de entrada
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Extração de features SOME/IP — Etapa 2 de Kim et al. (2026).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--parsed-csv",  default="data/parsed_packets.csv")
    ap.add_argument("--output-dir",  default="data/")
    ap.add_argument("--chunk-size",  type=int, default=500_000)
    args = ap.parse_args()

    run_feature_extraction(args.parsed_csv, args.output_dir, args.chunk_size)
