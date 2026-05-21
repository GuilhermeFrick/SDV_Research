"""
IDS SOME/IP em tempo real.

Captura pacotes numa interface de rede, extrai features stateful e classifica
com o modelo multi-classe XGBoost.

Uso:
    python simulation/ids/ids_live.py --iface lo
    python simulation/ids/ids_live.py --iface eth0 --threshold 0.6 --timeout 60
    python simulation/ids/ids_live.py --iface lo --log results/run1.jsonl
"""
import sys, json, time, argparse, collections
from pathlib import Path

import numpy as np
import xgboost as xgb

try:
    from scapy.all import sniff, conf as scapy_conf
except ImportError:
    print('scapy nao instalado: pip install scapy')
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
from parser_inline import parse_packet
from feature_extractor import FeatureExtractor, FEAT_COLS

ROOT      = Path(__file__).parent.parent.parent
MODEL_DIR = ROOT / 'detection' / 'multiclass' / 'model'
LOG_DIR   = Path(__file__).parent.parent / 'results'
LOG_DIR.mkdir(exist_ok=True)

CLASS_NAMES = ['Benigno', 'DoS', 'Fuzzy', 'MITM_Multi', 'MITM_Single']
ATTACK_CLR  = '\033[91m'
RESET_CLR   = '\033[0m'


def load_model(model_dir: Path):
    model = xgb.XGBClassifier()
    model.load_model(str(model_dir / 'multiclass_classifier.json'))
    with open(model_dir / 'norm_params.json') as f:
        norm = json.load(f)
    return model, norm


def normalize(x: np.ndarray, norm: dict) -> np.ndarray:
    x = x.copy()
    for j, col in enumerate(FEAT_COLS):
        lo = norm[col]['min']
        hi = norm[col]['max']
        d  = hi - lo
        x[j] = float(np.clip((x[j] - lo) / d, 0, 1)) if d > 0 else 0.0
    return x


class LiveIDS:
    def __init__(self, model, norm, threshold=0.5, top_n=20, log_path=None):
        self.model      = model
        self.norm       = norm
        self.threshold  = threshold
        self.top_n      = top_n
        self.extractor  = FeatureExtractor()
        self.log_path   = log_path

        self.ip_probs      = collections.defaultdict(lambda: np.zeros(len(CLASS_NAMES)))
        self.ip_counts     = collections.defaultdict(int)
        self.ip_first_seen = {}
        self.ip_alerts     = {}

        self.total_pkts = 0
        self.t_start    = time.perf_counter()
        self._t_report  = self.t_start

    def process(self, raw_pkt):
        pkt = parse_packet(raw_pkt)
        if pkt is None:
            return

        self.total_pkts += 1
        src = pkt.src_ip

        if src not in self.ip_first_seen:
            self.ip_first_seen[src] = pkt.timestamp

        feats = self.extractor.extract(pkt)
        feats = normalize(feats, self.norm)

        probs = self.model.predict_proba(feats.reshape(1, -1))[0]
        n     = self.ip_counts[src]
        self.ip_probs[src]  = (self.ip_probs[src] * n + probs) / (n + 1)
        self.ip_counts[src] += 1

        pred_class = int(np.argmax(self.ip_probs[src]))
        confidence = float(self.ip_probs[src][pred_class])

        if pred_class > 0 and confidence >= self.threshold and src not in self.ip_alerts:
            elapsed = time.perf_counter() - self.t_start
            alert = {
                'elapsed_s':    round(elapsed, 3),
                'src_ip':       src,
                'attack_class': CLASS_NAMES[pred_class],
                'confidence':   round(confidence, 4),
                'pkts_to_detect': self.ip_counts[src],
                'ts_first_pkt': self.ip_first_seen[src],
                'ts_alert':     pkt.timestamp,
            }
            self.ip_alerts[src] = alert
            print(f'{ATTACK_CLR}'
                  f'  [{elapsed:7.2f}s] *** ATAQUE ***  '
                  f'{src:<16}  {CLASS_NAMES[pred_class]:<12}  '
                  f'conf={confidence:.4f}  (detectado em {self.ip_counts[src]} pkts)'
                  f'{RESET_CLR}')
            if self.log_path:
                with open(self.log_path, 'a') as f:
                    f.write(json.dumps(alert) + '\n')

        now = time.perf_counter()
        if now - self._t_report >= 5.0:
            elapsed = now - self.t_start
            pps     = self.total_pkts / elapsed
            print(f'  [{elapsed:7.1f}s]  {self.total_pkts:>10,} pkts  '
                  f'{pps:>8,.0f} pkt/s  {len(self.ip_counts)} IPs')
            self._t_report = now

    def report(self):
        elapsed = time.perf_counter() - self.t_start
        print('\n' + '=' * 75)
        print('  RELATÓRIO FINAL — IDS SOME/IP Live')
        print('=' * 75)
        print(f'  Total pacotes : {self.total_pkts:,}')
        print(f'  Duração       : {elapsed:.1f}s  '
              f'({self.total_pkts / max(elapsed, 0.001):,.0f} pkt/s)')
        print(f'  IPs únicos    : {len(self.ip_counts)}')

        header = (f'\n  {"src_ip":<18}  {"pkts":>8}  '
                  + ''.join(f'  {n[:10]:<12}' for n in CLASS_NAMES)
                  + '  Detecção')
        print(header)
        print('  ' + '-' * 95)

        top = sorted(self.ip_counts.items(), key=lambda x: -x[1])
        for src, cnt in top[:self.top_n]:
            probs = self.ip_probs[src]
            pred  = int(np.argmax(probs))
            conf  = float(probs[pred])
            det   = (f'{ATTACK_CLR}{CLASS_NAMES[pred]} ***{RESET_CLR}'
                     if (pred > 0 and conf >= self.threshold) else 'Benigno')
            prob_s = ''.join(f'  {p:.4f}      ' for p in probs)
            print(f'  {src:<18}  {cnt:>8,}{prob_s}  {det}')

        if self.ip_alerts:
            print(f'\n  Alertas emitidos ({len(self.ip_alerts)}):')
            for src, a in self.ip_alerts.items():
                print(f'    {src:<16}  {a["attack_class"]:<12}  '
                      f'conf={a["confidence"]:.4f}  '
                      f'após {a["pkts_to_detect"]} pkts  '
                      f'em {a["elapsed_s"]:.2f}s')
        else:
            print('\n  Nenhum ataque detectado.')


def main():
    ap = argparse.ArgumentParser(description='IDS SOME/IP em tempo real')
    ap.add_argument('--iface',     default='lo',
                    help='Interface de rede (ex: lo, eth0, \\Device\\NPF_Loopback)')
    ap.add_argument('--model',     default=str(MODEL_DIR),
                    help='Diretório do modelo XGBoost')
    ap.add_argument('--threshold', type=float, default=0.5,
                    help='Limiar de confiança para alertar (padrão: 0.5)')
    ap.add_argument('--top-n',     type=int, default=20,
                    help='Top N IPs no relatório final')
    ap.add_argument('--log',       default=str(LOG_DIR / 'detections.jsonl'),
                    help='Arquivo JSONL para log de alertas')
    ap.add_argument('--timeout',   type=int, default=None,
                    help='Parar após N segundos (None = infinito)')
    args = ap.parse_args()

    model_dir = Path(args.model)
    if not (model_dir / 'multiclass_classifier.json').exists():
        print(f'Modelo nao encontrado em {model_dir}')
        print('Execute detection/multiclass/02_train.py primeiro.')
        sys.exit(1)

    print(f'Carregando modelo de {model_dir}...')
    model, norm = load_model(model_dir)
    print('  OK\n')

    ids = LiveIDS(model, norm,
                  threshold=args.threshold,
                  top_n=args.top_n,
                  log_path=args.log)

    print(f'Escutando em "{args.iface}"  (threshold={args.threshold})')
    print('Ctrl+C para parar.\n')

    try:
        sniff(iface=args.iface,
              prn=ids.process,
              store=False,
              timeout=args.timeout)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'\nErro ao capturar: {e}')
        print('No Windows, instale Npcap: https://npcap.com')
        print('No Linux, execute com sudo ou conceda CAP_NET_RAW.')

    ids.report()


if __name__ == '__main__':
    main()
