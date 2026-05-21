"""
Replay de cenários SOME/IP a partir de CSV parseado.

Lê o CSV gerado pelo parser (com campos timestamp, src_ip, dst_ip, src_port,
dst_port, transport, service_id, someip_payload_hex, label) e forja pacotes
SOME/IP via scapy, enviando numa interface de rede.

Uso:
    python simulation/replay/replay_sim.py --csv detection/data/parsed/dos_noti_flood.csv
    python simulation/replay/replay_sim.py --csv ... --speed 10.0 --label 1
    python simulation/replay/replay_sim.py --scenario fuzzy --speed 5.0
"""
import argparse, struct, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scapy.all import Ether, IP, TCP, UDP, Raw, sendp
except ImportError:
    print('scapy nao instalado: pip install scapy')
    sys.exit(1)

ROOT       = Path(__file__).parent.parent.parent
PARSED_DIR = ROOT / 'detection' / 'data' / 'parsed'

SOMEIP_HDR_LEN = 16
CHUNK          = 100_000

SCENARIOS = {
    'benign':      ('benign_traffic.csv',          None),
    'dos':         ('dos_noti_flood.csv',           1),
    'fuzzy':       ('fuzzy_sd_offer_rand_noti1.csv', 1),
    'mitm_multi':  ('mitm_multi_attacker.csv',      1),
    'mitm_single': ('mitm_single_attacker.csv',     1),
}


def build_someip(service_id, method_id, payload_hex: str) -> bytes:
    try:
        payload = bytes.fromhex(payload_hex) if payload_hex else b''
    except ValueError:
        payload = b''
    length     = 8 + len(payload)
    header     = struct.pack('>HHIHHBBBB',
        service_id & 0xFFFF,
        method_id  & 0xFFFF,
        length,
        0x0001, 0x0001,     # client_id, session_id
        0x01, 0x01,         # protocol_ver, interface_ver
        0x02, 0x00)         # msg_type=NOTIFICATION, return_code=OK
    return header + payload


def forge_packet(row, src_mac='02:00:00:00:01:00', dst_mac='ff:ff:ff:ff:ff:ff'):
    svc_id  = int(row.service_id) if pd.notna(row.service_id) else 0xFFFF
    mth_id  = int(row.method_id)  if pd.notna(row.method_id)  else 0x0000
    pld_hex = str(row.someip_payload_hex) if pd.notna(row.someip_payload_hex) else ''
    sport   = int(row.src_port) if pd.notna(row.src_port) else 30501
    dport   = int(row.dst_port) if pd.notna(row.dst_port) else 30490
    trans   = str(row.transport).upper() if pd.notna(row.transport) else 'UDP'

    someip  = build_someip(svc_id, mth_id, pld_hex)
    ip_pkt  = IP(src=str(row.src_ip), dst=str(row.dst_ip))
    l4      = (TCP(sport=sport, dport=dport, flags='PA')
               if trans == 'TCP' else UDP(sport=sport, dport=dport))

    return Ether(src=src_mac, dst=dst_mac) / ip_pkt / l4 / Raw(load=someip)


def main():
    ap = argparse.ArgumentParser(description='Replay SOME/IP a partir de CSV')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--csv',      help='CSV parseado direto')
    src.add_argument('--scenario', choices=list(SCENARIOS.keys()),
                     help='Cenário predefinido: ' + ', '.join(SCENARIOS))

    ap.add_argument('--iface',  default='lo',
                    help='Interface de rede (ex: lo, eth0, \\Device\\NPF_Loopback)')
    ap.add_argument('--speed',  type=float, default=1.0,
                    help='Multiplicador de velocidade (1.0=tempo real, 10.0=10× mais rápido)')
    ap.add_argument('--label',  type=int, default=None,
                    help='Filtrar por label (0=benigno, 1=ataque)')
    ap.add_argument('--limit',  type=int, default=None,
                    help='Máximo de pacotes a enviar')
    ap.add_argument('--dry-run', action='store_true',
                    help='Apenas conta pacotes, não envia')
    args = ap.parse_args()

    if args.scenario:
        csv_name, default_label = SCENARIOS[args.scenario]
        csv_path = PARSED_DIR / csv_name
        label_filter = args.label if args.label is not None else default_label
    else:
        csv_path     = Path(args.csv)
        label_filter = args.label

    if not csv_path.exists():
        print(f'CSV nao encontrado: {csv_path}')
        sys.exit(1)

    print(f'Carregando {csv_path.name}  (label={label_filter}  speed={args.speed}x)...')

    frames = []
    for chunk in pd.read_csv(csv_path, chunksize=CHUNK, low_memory=False):
        if label_filter is not None and 'label' in chunk.columns:
            chunk = chunk[chunk['label'] == label_filter]
        frames.append(chunk)
    df = pd.concat(frames, ignore_index=True).sort_values('timestamp')

    if args.limit:
        df = df.iloc[:args.limit]

    n = len(df)
    print(f'  {n:,} pacotes a enviar\n')

    if args.dry_run:
        print('[dry-run] Nenhum pacote enviado.')
        return

    ts_vals     = df['timestamp'].values
    t_start_sim = float(ts_vals[0])
    t_start_rl  = time.perf_counter()
    sent        = 0
    t_report    = t_start_rl

    for row in df.itertuples(index=False):
        ts_sim  = float(row.timestamp) - t_start_sim
        target  = ts_sim / args.speed
        elapsed = time.perf_counter() - t_start_rl
        wait    = target - elapsed
        if wait > 0.001:
            time.sleep(wait)

        pkt = forge_packet(row)
        sendp(pkt, iface=args.iface, verbose=False)
        sent += 1

        now = time.perf_counter()
        if now - t_report >= 5.0:
            el  = now - t_start_rl
            pps = sent / el
            sim_ts = float(row.timestamp) - t_start_sim
            print(f'  [{el:6.1f}s]  {sent:>8,} pkts  {pps:>8,.0f} pkt/s  '
                  f'sim_time={sim_ts:.1f}s')
            t_report = now

    elapsed = time.perf_counter() - t_start_rl
    print(f'\nConcluido: {sent:,} pacotes em {elapsed:.1f}s  '
          f'({sent / max(elapsed, 0.001):,.0f} pkt/s)')


if __name__ == '__main__':
    main()
