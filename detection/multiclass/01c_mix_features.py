"""
Gera dataset misto: intercala benigno + ataques selecionados por timestamp normalizado.

Diferença do 01_features.py:
  - Cada arquivo tem seus timestamps normalizados para [0, dur_benigno]
    antes da intercalação — tornando os arquivos temporalmente compatíveis.
  - O resultado simula um cenário onde ataques ocorrem DURANTE o tráfego
    legítimo, não antes nem depois.

Uso:
    python multiclass/01c_mix_features.py --attacks dos fuzzy mitm_multi mitm_single
    python multiclass/01c_mix_features.py --attacks dos --ratio_benigno 0.75
"""
import sys, time, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, deque

sys.path.insert(0, str(Path(__file__).parent.parent))
from paths import PARSED_DIR

OUT_DIR    = Path(__file__).parent / 'data'
OUT_DIR.mkdir(exist_ok=True)

RELAY_SVC  = 0x100B
CHUNK      = 500_000

ATTACK_FILES = {
    'dos':        [('dos_noti_flood.csv',          1)],
    'fuzzy':      [('fuzzy_sd_offer_rand_noti1.csv', 2),
                   ('fuzzy_sd_offer_rand_noti2.csv', 2),
                   ('fuzzy_sd_offer_rand_noti3.csv', 2)],
    'mitm_multi': [('mitm_multi_attacker.csv',     3)],
    'mitm_single':[('mitm_single_attacker.csv',    4)],
}

FEAT_COLS = [
    'f01_ip_time_interval', 'f08_someip_payload_change',
    'f11_ip_length_change',  'f12_tcpudp_length_change',
    'f13_payload_repeat_rate','f15_someip_payload_len',
    'f16_tcpudp_len',         'f17_src_packet_rate',
    'f18_src_payload_diversity','f19_is_sd',
    'f20_src_service_diversity','f21_is_relay_service',
    'f22_src_clientid_diversity',
]


# ── Feature extraction (idêntica ao 01_features.py) ───────────────────────────

def make_state():
    return {
        'prev_ts':         defaultdict(lambda: None),
        'prev_ip_len':     defaultdict(lambda: None),
        'prev_tl_len':     defaultdict(lambda: None),
        'prev_si_pld':     defaultdict(lambda: None),
        'recent_payloads': defaultdict(lambda: deque(maxlen=5)),
        'src_timestamps':  defaultdict(lambda: deque(maxlen=1000)),
        'src_payloads':    defaultdict(lambda: deque(maxlen=1000)),
        'src_services':    defaultdict(lambda: deque(maxlen=100)),
        'src_clientids':   defaultdict(lambda: deque(maxlen=100)),
    }


def _hamming(hex_a, hex_b) -> float:
    if not isinstance(hex_a, str) or not isinstance(hex_b, str):
        return 0.0
    try:
        a = bytes.fromhex(hex_a)
        b = bytes.fromhex(hex_b)
    except ValueError:
        return 0.0
    L = min(len(a), len(b))
    if L == 0:
        return 0.0
    return float(np.unpackbits(
        np.bitwise_xor(np.frombuffer(a[:L], dtype=np.uint8),
                       np.frombuffer(b[:L], dtype=np.uint8))
    ).sum()) / (8 * L)


def extract_features(df: pd.DataFrame, state: dict, mc_label: int) -> pd.DataFrame:
    df = df.sort_values('timestamp_norm').reset_index(drop=True)
    n  = len(df)

    ts_v  = df['timestamp_norm'].values
    ip_v  = pd.to_numeric(df['ip_len'],        errors='coerce').values
    tl_v  = pd.to_numeric(df['transport_len'], errors='coerce').values
    si_v  = df['someip_payload_hex'].fillna('').values
    src_v = df['src_ip'].astype(str).values
    svc_v = pd.to_numeric(df.get('service_id', pd.Series([np.nan]*n)),
                          errors='coerce').values
    mt_v  = pd.to_numeric(df.get('msg_type', pd.Series([np.nan]*n)),
                           errors='coerce').values
    cid_v = pd.to_numeric(df.get('client_id', pd.Series([np.nan]*n)),
                           errors='coerce').values
    pcap  = df.get('pcap_file', pd.Series([''] * n)).astype(str).values
    sport = df.get('src_port', pd.Series([''] * n)).astype(str).values
    dport = df.get('dst_port', pd.Series([''] * n)).astype(str).values
    trans = df.get('transport', pd.Series([''] * n)).astype(str).values
    is_sd_v = df.get('is_sd', pd.Series([False] * n)).fillna(False).astype(float).values

    f01 = np.zeros(n); f08 = np.zeros(n); f11 = np.zeros(n)
    f12 = np.zeros(n); f13 = np.zeros(n); f17 = np.zeros(n)
    f18 = np.zeros(n); f20 = np.ones(n);  f21 = np.zeros(n)
    f22 = np.ones(n)

    f15 = pd.to_numeric(df.get('someip_payload_len', pd.Series([0]*n)),
                        errors='coerce').fillna(0).values.astype(float)
    f16 = pd.to_numeric(df.get('transport_len', pd.Series([0]*n)),
                        errors='coerce').fillna(0).values.astype(float)

    for i in range(n):
        if not np.isnan(svc_v[i]) and int(svc_v[i]) == RELAY_SVC:
            f21[i] = 1.0

        key = (pcap[i], src_v[i], df['dst_ip'].iloc[i] if 'dst_ip' in df.columns else '',
               sport[i], dport[i], trans[i])
        ts   = float(ts_v[i])
        ip_l = float(ip_v[i]) if not np.isnan(ip_v[i]) else None
        tl_l = float(tl_v[i]) if not np.isnan(tl_v[i]) else None
        si_h = si_v[i] if si_v[i] else None
        src  = src_v[i]
        svc  = int(svc_v[i]) if not np.isnan(svc_v[i]) else None
        mt   = int(mt_v[i])  if not np.isnan(mt_v[i])  else None
        cid  = int(cid_v[i]) if not np.isnan(cid_v[i]) else None

        prev_ts = state['prev_ts'][key]
        f01[i]  = abs(ts - prev_ts) if prev_ts is not None else 0.0
        f08[i]  = _hamming(state['prev_si_pld'][key], si_h)

        p_ip = state['prev_ip_len'][key]
        p_tl = state['prev_tl_len'][key]
        f11[i] = abs(ip_l - p_ip) if (p_ip is not None and ip_l is not None) else 0.0
        f12[i] = abs(tl_l - p_tl) if (p_tl is not None and tl_l is not None) else 0.0

        hist = state['recent_payloads'][key]
        f13[i] = sum(1 for p in hist if p == si_h) / len(hist) if (hist and si_h) else 0.0
        hist.append(si_h)

        win_t = state['src_timestamps'][src]
        win_t.append(ts)
        if len(win_t) >= 2:
            delta = ts - win_t[0]
            f17[i] = (len(win_t) - 1) / delta if delta > 0 else float(len(win_t))

        win_p = state['src_payloads'][src]
        if si_h:
            win_p.append(si_h)
        f18[i] = len(set(win_p)) / len(win_p) if len(win_p) > 1 else 0.0

        win_s = state['src_services'][src]
        if svc is not None:
            win_s.append(svc)
        f20[i] = float(len(set(win_s))) if win_s else 1.0

        win_c = state['src_clientids'][src]
        if cid is not None and mt is not None and mt < 0x80:
            win_c.append(cid)
        f22[i] = float(len(set(win_c))) if win_c else 1.0

        state['prev_ts'][key]     = ts
        state['prev_ip_len'][key] = ip_l
        state['prev_tl_len'][key] = tl_l
        state['prev_si_pld'][key] = si_h

    out = pd.DataFrame({
        'f01_ip_time_interval':      f01, 'f08_someip_payload_change': f08,
        'f11_ip_length_change':      f11, 'f12_tcpudp_length_change':  f12,
        'f13_payload_repeat_rate':   f13, 'f15_someip_payload_len':    f15,
        'f16_tcpudp_len':            f16, 'f17_src_packet_rate':       f17,
        'f18_src_payload_diversity': f18, 'f19_is_sd':                 is_sd_v,
        'f20_src_service_diversity': f20, 'f21_is_relay_service':      f21,
        # f22 fixo em 1.0: consistente com o treino (01b_merge_fakeclientid.py
        # aplica f22=1.0 para todos os dados Kim, pois nenhum IP Kim rotaciona
        # client_ids de forma anômala — só o atacante Alkhatib faz isso).
        'f22_src_clientid_diversity': np.ones(len(f22)),
        'label': mc_label,
    })
    return out


# ── Carregamento e normalização temporal ───────────────────────────────────────

def _load_and_shift(csv_path: Path, mc_label: int,
                    only_attack: bool) -> pd.DataFrame | None:
    """Lê um CSV, filtra linhas relevantes e desloca timestamps para iniciar em 0.

    Mantém a escala real em segundos — apenas remove o offset absoluto do
    Unix timestamp. Isso preserva os intervalos reais entre pacotes, garantindo
    que f01 (intervalo) e f17 (taxa de pacotes/s) tenham os mesmos valores que
    teriam num pipeline de extração normal.
    """
    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=CHUNK, low_memory=False):
        if only_attack and 'label' in chunk.columns:
            chunk = chunk[chunk['label'] == 1]
        if len(chunk) > 0:
            chunks.append(chunk)

    if not chunks:
        return None

    df = pd.concat(chunks, ignore_index=True)
    if df.empty:
        return None

    # Deslocar para t=0 mantendo escala real em segundos
    df['timestamp_norm'] = df['timestamp'] - df['timestamp'].min()
    df['_mc_label'] = mc_label
    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def main(attacks: list[str], ratio_benigno: float, out_name: str):
    out_csv = OUT_DIR / out_name
    print(f'Dataset misto -> {out_csv}')
    print(f'Ataques: {attacks}  |  Ratio benigno: {ratio_benigno:.0%}')

    # 1. Carregar benigno e medir duração como referência
    ben_path = PARSED_DIR / 'benign_traffic.csv'
    if not ben_path.exists():
        raise FileNotFoundError(f'Não encontrado: {ben_path}')

    print('\n[1/3] Carregando benigno...')
    df_ben = _load_and_shift(ben_path, mc_label=0, only_attack=False)
    target_dur = df_ben['timestamp_norm'].max()
    print(f'      {len(df_ben):,} pacotes benigno · duracao real: {target_dur:.1f}s')

    # 2. Carregar e normalizar ataques
    attack_frames = []
    print('\n[2/3] Carregando ataques...')
    for atk in attacks:
        if atk not in ATTACK_FILES:
            print(f'  AVISO: ataque "{atk}" desconhecido — ignorado.')
            continue
        for fname, mc_label in ATTACK_FILES[atk]:
            p = PARSED_DIR / fname
            if not p.exists():
                print(f'  AVISO: {fname} não encontrado — ignorado.')
                continue
            df_atk = _load_and_shift(p, mc_label=mc_label, only_attack=True)
            if df_atk is not None:
                print(f'  {fname}: {len(df_atk):,} pacotes de ataque (label={mc_label})')
                attack_frames.append(df_atk)

    # 3. Intercalar por timestamp normalizado e ajustar proporção
    print('\n[3/3] Intercalando e extraindo features...')
    all_frames = [df_ben] + attack_frames
    df_all = pd.concat(all_frames, ignore_index=True)

    # Ajuste de proporção: sub-amostra benigno se necessário
    n_atk = (df_all['_mc_label'] != 0).sum()
    n_ben_target = int(n_atk / (1 - ratio_benigno) * ratio_benigno) if ratio_benigno < 1 else len(df_ben)
    if n_ben_target < len(df_ben):
        idx_ben = df_all[df_all['_mc_label'] == 0].sample(n_ben_target, random_state=42).index
        idx_atk = df_all[df_all['_mc_label'] != 0].index
        df_all  = df_all.loc[idx_ben.union(idx_atk)]
        print(f'  Benigno sub-amostrado: {len(idx_ben):,} -> proporção {ratio_benigno:.0%}')

    # Ordenar por timestamp normalizado (estratégia temporal)
    df_all = df_all.sort_values('timestamp_norm').reset_index(drop=True)
    n_total = len(df_all)
    print(f'  Total intercalado: {n_total:,} pacotes')

    # Extrair features em blocos mantendo estado global
    state   = make_state()
    first   = True
    t0      = time.time()
    offset  = 0
    blk     = 100_000

    while offset < n_total:
        block = df_all.iloc[offset: offset + blk].copy()
        mc_label_col = block['_mc_label'].values
        out = extract_features(block, state, mc_label=0)  # label será corrigido abaixo
        out['label'] = mc_label_col
        out.to_csv(out_csv, mode='a', header=first, index=False)
        first   = False
        offset += blk
        if offset % 500_000 == 0 or offset >= n_total:
            print(f'  {min(offset, n_total):,} / {n_total:,}  ({time.time()-t0:.0f}s)')

    print(f'\nConcluído em {time.time()-t0:.1f}s -> {out_csv}')

    # Distribuição final
    labels = pd.concat(c['label'] for c in pd.read_csv(
        out_csv, usecols=['label'], chunksize=CHUNK)).value_counts().sort_index()
    CLASS = {0:'Benigno',1:'DoS',2:'Fuzzy',3:'MITM_Multi',4:'MITM_Single',5:'FakeClientID'}
    print('\nDistribuição:')
    for lbl, cnt in labels.items():
        print(f'  {CLASS.get(lbl, str(lbl)):<15}: {cnt:>10,}  ({cnt/labels.sum()*100:.1f}%)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attacks', nargs='+',
                        choices=list(ATTACK_FILES.keys()),
                        default=list(ATTACK_FILES.keys()),
                        help='Ataques a incluir no mix')
    parser.add_argument('--ratio_benigno', type=float, default=0.6,
                        help='Proporção de pacotes benigno (0.0–1.0)')
    parser.add_argument('--out', default='features_misto.csv',
                        help='Nome do arquivo de saída em multiclass/data/')
    args = parser.parse_args()
    main(args.attacks, args.ratio_benigno, args.out)
