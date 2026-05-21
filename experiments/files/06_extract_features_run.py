# Auto-gerado do notebook 06_extract_features.ipynb

import sys, importlib.util, time, gc
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, deque
from sklearn.model_selection import train_test_split

ROOT     = Path(r'C:\Mestrado\SDV_Research')
SCRIPTS  = ROOT / 'experiments' / 'files'
CSV      = ROOT / 'data' / 'parsed_packets.csv'
NPY_DIR  = ROOT / 'data' / 'npy'
KIM_DIR  = ROOT / 'data' / 'dataset_ism_xgboost' / 'dataset_ism_xgboost' / 'tr_te_sets'
NPY_DIR.mkdir(exist_ok=True)

# Importa ByteDistributionModel e hamming_distance do script 02
spec = importlib.util.spec_from_file_location('feat', SCRIPTS / '02_extract_features.py')
feat_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feat_mod)
ByteDistributionModel = feat_mod.ByteDistributionModel
hamming_distance      = feat_mod.hamming_distance

CHUNK      = 500_000
N_SAMPLES  = 50_000   # amostras benignas para treino dos modelos
RAND_STATE = 42

print('CSV:', CSV)
print('Existe:', CSV.exists())
if CSV.exists():
    print(f'Tamanho: {CSV.stat().st_size/1e9:.2f} GB')


cols_model = ['label', 'attack_type', 'is_sd',
              'someip_payload_hex', 'transport_payload_hex']

si_payloads, sd_payloads, tu_payloads = [], [], []
n_si = n_sd = n_tu = 0

print('Coletando amostras para modelos de bytes...')
for chunk in pd.read_csv(CSV, usecols=cols_model, chunksize=CHUNK, low_memory=False):
    b_benign = chunk[(chunk['label'] == 0) & (chunk['attack_type'] == 'normal')]
    b_all    = chunk[chunk['label'] == 0]

    is_sd_b = b_benign['is_sd'].fillna(False).astype(str).str.lower().isin(['true','1'])
    is_sd_a = b_all['is_sd'].fillna(False).astype(str).str.lower().isin(['true','1'])

    # model_si + model_tu: trafego benigno puro (attack_type=normal)
    if n_si < N_SAMPLES:
        s = b_benign[~is_sd_b]['someip_payload_hex'].dropna()
        si_payloads.append(s); n_si += len(s)
    if n_tu < N_SAMPLES:
        s = b_benign['transport_payload_hex'].dropna()
        tu_payloads.append(s); n_tu += len(s)
    # model_sd: todos label=0 — ECUs legitimas em todos os PCAPs (30x mais amostras SD)
    if n_sd < N_SAMPLES:
        s = b_all[is_sd_a]['someip_payload_hex'].dropna()
        sd_payloads.append(s); n_sd += len(s)

    if n_si >= N_SAMPLES and n_sd >= N_SAMPLES and n_tu >= N_SAMPLES:
        break

si_sample = pd.concat(si_payloads).head(N_SAMPLES)
sd_sample = pd.concat(sd_payloads).head(N_SAMPLES)
tu_sample = pd.concat(tu_payloads).head(N_SAMPLES)
del si_payloads, sd_payloads, tu_payloads; gc.collect()

print(f'  SOME/IP regular  : {len(si_sample):,} amostras (benigno puro)')
print(f'  SOME/IP-SD       : {len(sd_sample):,} amostras (todos label=0)')
print(f'  TCP/UDP          : {len(tu_sample):,} amostras (benigno puro)')

t0 = time.time()
model_si = ByteDistributionModel(alpha=1.0)
model_sd = ByteDistributionModel(alpha=1.0)
model_tu = ByteDistributionModel(alpha=1.0)

model_si.fit(si_sample)
if len(sd_sample) >= 100:
    model_sd.fit(sd_sample)
    print(f'modelo SD treinado: {len(sd_sample):,} amostras')
else:
    model_sd.fit(si_sample)
    print('[AVISO] SD insuficiente — modelo SD usa SOME/IP como proxy')
model_tu.fit(tu_sample)
print(f'Modelos treinados em {time.time()-t0:.1f}s')


FEAT_COLS = [
    'f01_ip_time_interval',
    'f02_someip_likelihood',
    'f03_someipsd_likelihood',
    'f04_tcpudp_likelihood',
    'f05_someip_entropy',
    'f06_someipsd_entropy',
    'f07_tcpudp_entropy',
    'f08_someip_payload_changes',
    'f09_someipsd_payload_changes',
    'f10_tcpudp_payload_changes',
    'f11_ip_length_changes',
    'f12_tcpudp_length_changes',
    'f13_payload_repeat_rate',
    'f14_duplicate_source',
    'f15_someip_payload_length',
    'f16_tcpudp_payload_length',
]

def extract_16(df, flow_state):
    df = df.sort_values(['pcap_file', 'timestamp']).reset_index(drop=True)
    n  = len(df)

    si_hex = df['someip_payload_hex'].fillna('')
    tu_hex = df['transport_payload_hex'].fillna('')

    f02 = model_si.log_likelihood_batch(si_hex)
    f03 = model_sd.log_likelihood_batch(si_hex)
    f04 = model_tu.log_likelihood_batch(tu_hex)
    f05 = model_si.cross_entropy_batch(si_hex)
    f06 = model_sd.cross_entropy_batch(si_hex)
    f07 = model_tu.cross_entropy_batch(tu_hex)

    prev_ts          = flow_state['prev_ts']
    prev_ip_len      = flow_state['prev_ip_len']
    prev_tl_len      = flow_state['prev_tl_len']
    prev_si_pld      = flow_state['prev_si_pld']
    prev_sd_pld      = flow_state['prev_sd_pld']
    prev_tu_pld      = flow_state['prev_tu_pld']
    recent_payloads  = flow_state['recent_payloads']   # per-flow deque(maxlen=5)
    last_src_payload = flow_state['last_src_payload']  # (pcap, hash) -> src_ip

    ts_v   = df['timestamp'].values
    ip_v   = pd.to_numeric(df['ip_len'], errors='coerce').values
    tl_v   = pd.to_numeric(df['transport_len'], errors='coerce').values
    si_v   = si_hex.values
    tu_v   = tu_hex.values
    is_sd  = df['is_sd'].fillna(False).astype(str).str.lower().isin(['true','1']).values
    src_ip = df['src_ip'].astype(str).values
    dst_ip = df['dst_ip'].astype(str).values
    sport  = df['src_port'].astype(str).values
    dport  = df['dst_port'].astype(str).values
    trans  = df['transport'].astype(str).values
    pcap   = df['pcap_file'].astype(str).values

    f01 = np.zeros(n); f08 = np.zeros(n); f09 = np.zeros(n)
    f10 = np.zeros(n); f11 = np.zeros(n); f12 = np.zeros(n)
    f13 = np.zeros(n); f14 = np.zeros(n)
    # f15: someip_payload_len do cabeçalho SOME/IP (não truncado pelo parser)
    # fuzzy SD: 1332 bytes vs benigno SD: 44–60 bytes — discriminação direta
    f15 = pd.to_numeric(df['someip_payload_len'], errors='coerce').fillna(0).values.astype(float)
    # f16: transport_len do cabeçalho UDP/TCP (campo real do pacote)
    f16 = pd.to_numeric(df['transport_len'],      errors='coerce').fillna(0).values.astype(float)

    for i in range(n):
        key = (pcap[i], src_ip[i], dst_ip[i], sport[i], dport[i], trans[i])
        ts   = float(ts_v[i])
        ip_l = float(ip_v[i]) if not np.isnan(ip_v[i]) else None
        tl_l = float(tl_v[i]) if not np.isnan(tl_v[i]) else None
        si_h = si_v[i] if si_v[i] else None
        tu_h = tu_v[i] if tu_v[i] else None
        sd   = bool(is_sd[i])

        f01[i] = abs(ts - prev_ts[key]) if prev_ts[key] is not None else 0.0
        f08[i] = hamming_distance(prev_si_pld[key], si_h) if not sd else 0.0
        f09[i] = hamming_distance(prev_sd_pld[key], si_h) if sd     else 0.0
        f10[i] = hamming_distance(prev_tu_pld[key], tu_h)
        f11[i] = abs(ip_l - prev_ip_len[key]) if prev_ip_len[key] is not None and ip_l is not None else 0.0
        f12[i] = abs(tl_l - prev_tl_len[key]) if prev_tl_len[key] is not None and tl_l is not None else 0.0

        # f13: fração dos últimos 5 payloads do fluxo idênticos ao atual
        # alto em DoS (flood sempre igual) e MITM relay (mesmo pacote interceptado)
        # zero em fuzzy (payload aleatório a cada pacote)
        hist = recent_payloads[key]
        if hist and si_h:
            f13[i] = sum(1 for p in hist if p == si_h) / len(hist)
        hist.append(si_h)

        # f14: 1 se mesmo payload foi visto de src_ip diferente em qualquer PCAP
        # chave sem pcap_file — necessário para detectar relay cross-PCAP:
        # vítima envia P em normal.pcap; atacante relay P em mitm.pcap
        if si_h:
            pld_key = hash(si_h)
            prev_src = last_src_payload.get(pld_key)
            if prev_src is not None and prev_src != src_ip[i]:
                f14[i] = 1.0
            last_src_payload[pld_key] = src_ip[i]

        prev_ts[key]     = ts
        prev_ip_len[key] = ip_l
        prev_tl_len[key] = tl_l
        prev_tu_pld[key] = tu_h
        if not sd: prev_si_pld[key] = si_h
        else:      prev_sd_pld[key] = si_h

    labels      = df['label'].values if 'label' in df.columns else np.zeros(n, dtype=int)
    attack_type = df['attack_type'].values if 'attack_type' in df.columns else np.full(n, 'unknown')
    return pd.DataFrame({
        'f01_ip_time_interval':        f01,
        'f02_someip_likelihood':        f02,
        'f03_someipsd_likelihood':      f03,
        'f04_tcpudp_likelihood':        f04,
        'f05_someip_entropy':           f05,
        'f06_someipsd_entropy':         f06,
        'f07_tcpudp_entropy':           f07,
        'f08_someip_payload_changes':   f08,
        'f09_someipsd_payload_changes': f09,
        'f10_tcpudp_payload_changes':   f10,
        'f11_ip_length_changes':        f11,
        'f12_tcpudp_length_changes':    f12,
        'f13_payload_repeat_rate':      f13,
        'f14_duplicate_source':         f14,
        'f15_someip_payload_length':    f15,
        'f16_tcpudp_payload_length':    f16,
        'label':       labels,
        'attack_type': attack_type,
    })

RAW_CSV = NPY_DIR / 'all_features_raw.csv'

if RAW_CSV.exists():
    print(f'[OK] {RAW_CSV.name} ja existe — pulando extração.')
else:
    flow_state = {
        'prev_ts':          defaultdict(lambda: None),
        'prev_ip_len':      defaultdict(lambda: None),
        'prev_tl_len':      defaultdict(lambda: None),
        'prev_si_pld':      defaultdict(lambda: None),
        'prev_sd_pld':      defaultdict(lambda: None),
        'prev_tu_pld':      defaultdict(lambda: None),
        'recent_payloads':  defaultdict(lambda: deque(maxlen=5)),
        'last_src_payload': {},
    }
    first = True; n_total = 0; t0 = time.time()
    for chunk in pd.read_csv(CSV, chunksize=CHUNK, low_memory=False):
        out = extract_16(chunk, flow_state)
        out.to_csv(RAW_CSV, mode='a', header=first, index=False)
        first = False; n_total += len(out)
        print(f'  {n_total:,} features extraidas  ({time.time()-t0:.0f}s)')
    print(f'Concluido: {n_total:,} amostras')


TRAIN_CSV = NPY_DIR / 'train_features.csv'
TEST_CSV  = NPY_DIR / 'test_features.csv'

if TRAIN_CSV.exists() and TEST_CSV.exists():
    print('[OK] train/test_features.csv ja existem — pulando split.')
else:
    print('Lendo labels para split estratificado...')
    labels = pd.concat(
        c['label'] for c in pd.read_csv(RAW_CSV, usecols=['label'], chunksize=CHUNK)
    ).values

    idx = np.arange(len(labels))
    train_idx, _ = train_test_split(idx, test_size=0.5, stratify=labels, random_state=RAND_STATE)
    is_train = np.zeros(len(labels), dtype=bool)
    is_train[train_idx] = True

    n0_tr = (labels[is_train] == 0).sum()
    n1_tr = (labels[is_train] == 1).sum()
    print(f'Treino: {is_train.sum():,}  (normal={n0_tr:,}  ataque={n1_tr:,})')
    print(f'Teste : {(~is_train).sum():,}')

    row = 0; ftr = fte = True
    for chunk in pd.read_csv(RAW_CSV, chunksize=CHUNK):
        mask = is_train[row: row + len(chunk)]
        chunk[mask].to_csv(TRAIN_CSV,  mode='a', header=ftr, index=False)
        chunk[~mask].to_csv(TEST_CSV,  mode='a', header=fte, index=False)
        ftr = fte = False; row += len(chunk)
    print('Split concluido.')


# Calcula min/max somente no treino
stats = {c: {'min': float('inf'), 'max': float('-inf')} for c in FEAT_COLS}
for chunk in pd.read_csv(TRAIN_CSV, usecols=FEAT_COLS, chunksize=CHUNK):
    for c in FEAT_COLS:
        stats[c]['min'] = min(stats[c]['min'], chunk[c].min())
        stats[c]['max'] = max(stats[c]['max'], chunk[c].max())

print('Min/Max por feature (treino):')
for c in FEAT_COLS:
    print(f'  {c:<35}  min={stats[c]["min"]:10.4f}  max={stats[c]["max"]:10.4f}')

def export_npy(csv_path, split_name):
    Xc, yc = [], []
    for chunk in pd.read_csv(csv_path, chunksize=CHUNK):
        X = chunk[FEAT_COLS].values.astype(np.float32)
        for j, c in enumerate(FEAT_COLS):
            lo, hi = stats[c]['min'], stats[c]['max']
            d = hi - lo
            X[:, j] = np.clip((X[:, j] - lo) / d, 0.0, 1.0) if d > 0 else 0.0
        Xc.append(X)
        yc.append(chunk['label'].values.astype(np.int8))
    X_all = np.vstack(Xc); y_all = np.concatenate(yc)
    np.save(NPY_DIR / f'X_{split_name}.npy', X_all)
    np.save(NPY_DIR / f'y_{split_name}.npy', y_all)
    n0, n1 = (y_all==0).sum(), (y_all==1).sum()
    print(f'{split_name}: X={X_all.shape}  normal={n0:,} ({100*n0/len(y_all):.1f}%)  '
          f'ataque={n1:,} ({100*n1/len(y_all):.1f}%)')
    return X_all, y_all

print('\nExportando .npy...')
X_train, y_train = export_npy(TRAIN_CSV, 'train')
X_test,  y_test  = export_npy(TEST_CSV,  'test')

print('\nArquivos salvos:')
for f in sorted(NPY_DIR.glob('*.npy')):
    print(f'  {f.name}  ({f.stat().st_size/1e6:.1f} MB)')


KIM_COLS = [
    'SOMEIP_likelihood', 'SOMEIP_entropy',
    'SOMEIPSD_likelihood', 'TCPUDP_likelihood',
    'SOMEIPSD_entropy', 'TCPUDP_entropy',
    'SOMEIP_pld_changes', 'SOMEIPSD_pld_changes', 'TCPUDP_pld_changes',
    'IP_time_interval', 'IP_length_changes', 'TCPUDP_length_changes',
]
# Mapeamento correto: índice da nossa feature -> coluna Kim no .npy
# Kim: col0=SOMEIP_lik, col1=SOMEIP_ent, col2=SD_lik, col3=TU_lik,
#      col4=SD_ent, col5=TU_ent, col6=SOMEIP_chg, col7=SD_chg, col8=TU_chg,
#      col9=IP_interval, col10=IP_len_chg, col11=TU_len_chg
# Nossas features: f01→col9, f02→col0, f03→col2, f04→col3, f05→col1,
#                  f06→col4, f07→col5, f08→col6, f09→col7, f10→col8, f11→col10, f12→col11
OUR_TO_KIM = [9, 0, 2, 3, 1, 4, 5, 6, 7, 8, 10, 11]

kim_x = KIM_DIR / 'X_train.npy'
kim_y = KIM_DIR / 'y_train.npy'

if not kim_x.exists():
    print(f'Dataset Kim nao encontrado em {KIM_DIR}')
else:
    X_kim = np.load(kim_x, mmap_mode='r')
    y_kim = np.load(kim_y)
    print(f'Kim   : X={X_kim.shape}  normal={(y_kim==0).sum():,}  ataque={(y_kim==1).sum():,}')
    print(f'Nosso : X={X_train.shape}  normal={(y_train==0).sum():,}  ataque={(y_train==1).sum():,}')

    print(f'\n{"Feature":<35} {"nossa_mean_n":>14} {"kim_col":>7} {"kim_mean_n":>12} {"nossa_mean_a":>14} {"kim_mean_a":>12}')
    print('-' * 100)
    for j, (col, k_idx) in enumerate(zip(FEAT_COLS[:12], OUR_TO_KIM)):
        our_n = X_train[y_train==0, j].mean()
        our_a = X_train[y_train==1, j].mean()
        kim_n = X_kim[y_kim==0, k_idx].mean()
        kim_a = X_kim[y_kim==1, k_idx].mean()
        print(f'{col:<35}  {our_n:>14.4f}  col{k_idx:<3}  {kim_n:>12.4f}  {our_a:>14.4f}  {kim_a:>12.4f}')

    print('\nNota: diferenças esperadas em f08-f12 — Kim pode usar abs(delta) e normalizacao por payload length.')
    print('As features f02-f07 (likelihood/entropy) devem ser as mais comparaveis.')
    print('f13/f14 sao novas — sem equivalente no dataset Kim.')
