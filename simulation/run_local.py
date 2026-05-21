"""
Simulador local do IDS SOME/IP — sem rede, sem Docker, sem root.

Lê CSVs parseados, cria SomeIPPackets diretamente das linhas e passa pelo
pipeline completo: feature extraction stateful → XGBoost multi-classe → alertas.

Suporta cenários mistos (benigno + ataque interleaved por timestamp).

Uso:
    python simulation/run_local.py --scenario dos
    python simulation/run_local.py --scenario fuzzy --speed 0 --limit 50000
    python simulation/run_local.py --scenario mixed_dos
    python simulation/run_local.py --csv path/to/file.csv --label 1
"""
import sys, json, time, argparse, collections
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'simulation' / 'ids'))

from parser_inline import SomeIPPacket, SOMEIP_SD_SVC
from feature_extractor import FeatureExtractor, FEAT_COLS

PARSED_DIR = ROOT / 'detection' / 'data' / 'parsed'
MODEL_DIR  = ROOT / 'detection' / 'multiclass' / 'model'
LOG_DIR    = ROOT / 'simulation' / 'results'
LOG_DIR.mkdir(exist_ok=True)

CLASS_NAMES = ['Benigno', 'DoS', 'Fuzzy', 'MITM_Multi', 'MITM_Single']
CHUNK       = 200_000

# csv_file, label_filter, ground_truth_label
SCENARIOS = {
    'benign':      [('benign_traffic.csv',            None, 'Benigno')],
    'dos':         [('dos_noti_flood.csv',             1,   'DoS')],
    'fuzzy':       [('fuzzy_sd_offer_rand_noti1.csv',  1,   'Fuzzy')],
    'mitm_multi':  [('mitm_multi_attacker.csv',        1,   'MITM_Multi')],
    'mitm_single': [('mitm_single_attacker.csv',       1,   'MITM_Single')],
    # Mistos: benigno + ataque interleaved por timestamp
    'mixed_dos':         [('benign_traffic.csv', None, 'Benigno'),
                          ('dos_noti_flood.csv',  1,   'DoS')],
    'mixed_fuzzy':       [('benign_traffic.csv', None, 'Benigno'),
                          ('fuzzy_sd_offer_rand_noti1.csv', 1, 'Fuzzy')],
    'mixed_mitm_multi':  [('benign_traffic.csv', None, 'Benigno'),
                          ('mitm_multi_attacker.csv', 1, 'MITM_Multi')],
    'mixed_mitm_single': [('benign_traffic.csv', None, 'Benigno'),
                          ('mitm_single_attacker.csv', 1, 'MITM_Single')],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model():
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_DIR / 'multiclass_classifier.json'))
    with open(MODEL_DIR / 'norm_params.json') as f:
        norm = json.load(f)
    return model, norm


def normalize(x: np.ndarray, norm: dict) -> np.ndarray:
    x = x.copy()
    for j, col in enumerate(FEAT_COLS):
        lo = norm[col]['min']; hi = norm[col]['max']; d = hi - lo
        x[j] = float(np.clip((x[j] - lo) / d, 0, 1)) if d > 0 else 0.0
    return x


def row_to_packet(row, sim_ts: float) -> SomeIPPacket:
    """Converte linha do CSV em SomeIPPacket sem passar pela rede."""
    def _int(v, default=0):
        try:
            return int(v) if pd.notna(v) else default
        except (ValueError, TypeError):
            return default

    def _str(v, default=''):
        return str(v) if pd.notna(v) else default

    svc_id  = _int(getattr(row, 'service_id', None), 0)
    pld_hex = _str(getattr(row, 'someip_payload_hex', None))
    pld_len = _int(getattr(row, 'someip_payload_len', None))

    return SomeIPPacket(
        timestamp          = sim_ts,
        src_ip             = _str(getattr(row, 'src_ip', '0.0.0.0')),
        dst_ip             = _str(getattr(row, 'dst_ip', '0.0.0.0')),
        src_port           = _int(getattr(row, 'src_port', 0)),
        dst_port           = _int(getattr(row, 'dst_port', 30490)),
        transport          = _str(getattr(row, 'transport', 'UDP')).upper(),
        ip_len             = _int(getattr(row, 'ip_len', 0)),
        transport_len      = _int(getattr(row, 'transport_len', 0)),
        service_id         = svc_id,
        method_id          = _int(getattr(row, 'method_id', 0)),
        someip_payload_hex = pld_hex,
        someip_payload_len = pld_len if pld_len else len(pld_hex) // 2,
        is_sd              = (svc_id == SOMEIP_SD_SVC),
    )


def load_sources(sources: list, limit: int | None) -> pd.DataFrame:
    """Carrega e combina CSVs, ordena por timestamp."""
    frames = []
    for csv_name, label_filter, gt_label in sources:
        path = PARSED_DIR / csv_name
        if not path.exists():
            print(f'  [AVISO] {path} nao encontrado, pulando.')
            continue
        print(f'  Lendo {csv_name}  (label={label_filter})...')
        parts = []
        for chunk in pd.read_csv(path, chunksize=CHUNK, low_memory=False):
            if label_filter is not None and 'label' in chunk.columns:
                chunk = chunk[chunk['label'] == label_filter]
            chunk['_gt'] = gt_label
            parts.append(chunk)
        if parts:
            frames.append(pd.concat(parts, ignore_index=True))

    if not frames:
        print('Nenhum dado carregado.')
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True).sort_values('timestamp')
    if limit:
        df = df.iloc[:limit]
    return df.reset_index(drop=True)


# ── IDS local ─────────────────────────────────────────────────────────────────

class LocalIDS:
    def __init__(self, model, norm, threshold=0.5, speed=1.0):
        self.model     = model
        self.norm      = norm
        self.threshold = threshold
        self.speed     = speed
        self.extractor = FeatureExtractor()

        self.ip_probs      = collections.defaultdict(lambda: np.zeros(len(CLASS_NAMES)))
        self.ip_counts     = collections.defaultdict(int)
        self.ip_gt         = {}
        self.ip_first_seen = {}
        self.ip_alerts     = {}

        self.total   = 0
        self.t_wall0 = time.perf_counter()
        self.t_sim0  = None
        self._t_rep  = self.t_wall0

    def _wait(self, sim_ts: float):
        """Throttle para simular tempo real (speed=1.0) ou burst (speed=0)."""
        if self.speed == 0:
            return
        if self.t_sim0 is None:
            self.t_sim0 = sim_ts
            return
        sim_elapsed  = (sim_ts - self.t_sim0) / self.speed
        wall_elapsed = time.perf_counter() - self.t_wall0
        wait = sim_elapsed - wall_elapsed
        if wait > 0.001:
            time.sleep(wait)

    def process(self, row, sim_ts: float):
        self._wait(sim_ts)
        pkt = row_to_packet(row, sim_ts)

        src = pkt.src_ip
        gt  = getattr(row, '_gt', 'Benigno')

        if src not in self.ip_first_seen:
            self.ip_first_seen[src] = sim_ts
            self.ip_gt[src] = gt

        feats = self.extractor.extract(pkt)
        feats = normalize(feats, self.norm)
        probs = self.model.predict_proba(feats.reshape(1, -1))[0]

        n = self.ip_counts[src]
        self.ip_probs[src]  = (self.ip_probs[src] * n + probs) / (n + 1)
        self.ip_counts[src] += 1
        self.total += 1

        pred  = int(np.argmax(self.ip_probs[src]))
        conf  = float(self.ip_probs[src][pred])

        if pred > 0 and conf >= self.threshold and src not in self.ip_alerts:
            wall_elapsed = time.perf_counter() - self.t_wall0
            self.ip_alerts[src] = {
                'src_ip':         src,
                'ground_truth':   self.ip_gt[src],
                'predicted':      CLASS_NAMES[pred],
                'confidence':     round(conf, 4),
                'pkts_to_detect': self.ip_counts[src],
                'sim_ts_alert':   round(sim_ts, 3),
                'wall_elapsed_s': round(wall_elapsed, 3),
                'correct':        (CLASS_NAMES[pred] == self.ip_gt[src]),
            }
            status = 'OK' if self.ip_alerts[src]['correct'] else 'ERRO'
            print(f'  [{wall_elapsed:7.2f}s] [{status}] *** ATAQUE ***  '
                  f'{src:<16}  pred={CLASS_NAMES[pred]:<12}  '
                  f'gt={self.ip_gt[src]:<12}  '
                  f'conf={conf:.4f}  '
                  f'(após {self.ip_counts[src]:,} pkts)')

        now = time.perf_counter()
        if now - self._t_rep >= 5.0:
            el  = now - self.t_wall0
            pps = self.total / max(el, 1e-9)
            print(f'  [{el:7.1f}s]  {self.total:>10,} pkts  {pps:>8,.0f} pkt/s  '
                  f'{len(self.ip_counts)} IPs')
            self._t_rep = now

    def report(self, log_path=None):
        elapsed = time.perf_counter() - self.t_wall0
        n_ips   = len(self.ip_counts)

        print('\n' + '=' * 75)
        print('  RELATÓRIO — Simulação Local IDS SOME/IP')
        print('=' * 75)
        print(f'  Total pacotes : {self.total:,}')
        print(f'  Duração wall  : {elapsed:.1f}s  ({self.total/max(elapsed,1e-9):,.0f} pkt/s)')
        print(f'  IPs únicos    : {n_ips}')

        # Tabela por IP
        top = sorted(self.ip_counts.items(), key=lambda x: -x[1])
        print(f'\n  {"src_ip":<18}  {"pkts":>8}  {"GT":<12}  '
              + ''.join(f'  {n[:10]:<12}' for n in CLASS_NAMES)
              + '  Detecção')
        print('  ' + '-' * 100)
        for src, cnt in top:
            probs = self.ip_probs[src]
            pred  = int(np.argmax(probs))
            conf  = float(probs[pred])
            gt    = self.ip_gt.get(src, '?')
            det   = (f'{CLASS_NAMES[pred]} ***' if (pred > 0 and conf >= self.threshold)
                     else 'Benigno')
            p_str = ''.join(f'  {p:.4f}      ' for p in probs)
            print(f'  {src:<18}  {cnt:>8,}  {gt:<12}{p_str}  {det}')

        # Métricas de detecção
        print(f'\n  Alertas emitidos: {len(self.ip_alerts)}')
        if self.ip_alerts:
            correct = sum(1 for a in self.ip_alerts.values() if a['correct'])
            print(f'  Corretos        : {correct} / {len(self.ip_alerts)}')
            print(f'\n  {"src_ip":<16}  {"GT":<12}  {"Pred":<12}  '
                  f'{"pkts":>8}  {"conf":>6}  {"wall(s)":>8}')
            print('  ' + '-' * 70)
            for src, a in self.ip_alerts.items():
                ok = 'OK ' if a['correct'] else 'ERR'
                print(f'  [{ok}] {src:<16}  {a["ground_truth"]:<12}  '
                      f'{a["predicted"]:<12}  '
                      f'{a["pkts_to_detect"]:>8,}  '
                      f'{a["confidence"]:>6.4f}  '
                      f'{a["wall_elapsed_s"]:>8.2f}s')

        if log_path:
            with open(log_path, 'w') as f:
                json.dump({
                    'total_pkts':  self.total,
                    'elapsed_s':   round(elapsed, 2),
                    'throughput':  round(self.total / max(elapsed, 1e-9)),
                    'alerts':      list(self.ip_alerts.values()),
                }, f, indent=2)
            print(f'\n  Log salvo em {log_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Simulação local do IDS SOME/IP')

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--scenario', choices=list(SCENARIOS.keys()),
                     help='Cenário predefinido')
    src.add_argument('--csv',      help='CSV parseado direto')

    ap.add_argument('--label',     type=int, default=None,
                    help='Filtrar por label (com --csv)')
    ap.add_argument('--speed',     type=float, default=0,
                    help='Velocidade: 0=burst (max), 1.0=tempo real, 0.1=10x mais rápido')
    ap.add_argument('--limit',     type=int, default=None,
                    help='Limitar número de pacotes')
    ap.add_argument('--threshold', type=float, default=0.5,
                    help='Limiar de confiança para alertar')
    ap.add_argument('--log',       default=None,
                    help='Salvar resultado em JSON (ex: results/run.json)')
    args = ap.parse_args()

    if args.scenario:
        sources = SCENARIOS[args.scenario]
    else:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f'CSV nao encontrado: {csv_path}')
            sys.exit(1)
        sources = [(csv_path.name, args.label, 'Desconhecido')]
        # override PARSED_DIR para usar o path direto
        global PARSED_DIR
        PARSED_DIR = csv_path.parent

    print(f'Carregando modelo de {MODEL_DIR}...')
    model, norm = load_model()
    print('  OK\n')

    print('Carregando dados...')
    df = load_sources(sources, args.limit)
    print(f'  {len(df):,} pacotes carregados\n')

    ids = LocalIDS(model, norm, threshold=args.threshold, speed=args.speed)

    speed_str = 'burst (máximo)' if args.speed == 0 else f'{args.speed}x'
    print(f'Iniciando simulação  (speed={speed_str}  threshold={args.threshold})\n')

    ts_col = df['timestamp'].values
    for i, row in enumerate(df.itertuples(index=False)):
        ids.process(row, float(ts_col[i]))

    log_path = args.log or str(LOG_DIR / f'{args.scenario or "custom"}_result.json')
    ids.report(log_path=log_path)


if __name__ == '__main__':
    main()
