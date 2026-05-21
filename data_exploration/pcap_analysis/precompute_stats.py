#!/usr/bin/env python3
"""
Extrai estatísticas de todos os PCAPs via tshark e salva em pcap_stats.json.
Roda uma vez fora do notebook para evitar timeout.
"""
import subprocess, io, json, os
import pandas as pd

TSHARK = r'C:\Program Files\Wireshark\tshark.exe'
_B     = r'C:\Mestrado\SDV_Research\experiments\notebooks\data\pcap'

PCAPS = {
    'benign':      _B + r'\benign_traffic.pcap',
    'dos_noti':    _B + r'\dos_noti_flood.pcap',
    'fuzzy(1)':    _B + r'\fuzzy_sd_offer_rand_noti(1).pcap',
    'fuzzy(2)':    _B + r'\fuzzy_sd_offer_rand_noti(2).pcap',
    'fuzzy(3)':    _B + r'\fuzzy_sd_offer_rand_noti(3).pcap',
    'mitm_multi':  _B + r'\mitm_multi_attacker.pcap',
    'mitm_single': _B + r'\mitm_single_attacker.pcap',
}

FIELDS = [
    'frame.number', 'frame.time_relative', 'frame.len', '_ws.col.Protocol',
    'ip.src', 'ip.dst', 'ip.proto', 'ip.ttl',
    'tcp.srcport', 'tcp.dstport', 'udp.srcport', 'udp.dstport',
    'someip.serviceid', 'someip.methodid', 'someip.messagetype',
    'someip.returncode', 'someip.length', 'someipsd.entry.type',
]
COL = [
    'num','time','frame_len','proto','src_ip','dst_ip','ip_proto','ttl',
    'tcp_sport','tcp_dport','udp_sport','udp_dport',
    'svc_id','meth_id','msg_type','ret_code','someip_len','sd_entry_type',
]
MSG = {
    '0x00':'REQUEST', '0x01':'REQUEST_NO_RETURN', '0x02':'NOTIFICATION',
    '0x40':'REQUEST_ACK', '0x41':'RQST_NRET_ACK', '0x42':'NOTIF_ACK',
    '0x80':'RESPONSE', '0x81':'ERROR', '0xc0':'RESPONSE_ACK', '0xc1':'ERROR_ACK',
}

results = {}

for label, pcap in PCAPS.items():
    print(f'\nProcessando {label} ...')
    cmd = [
        TSHARK,
        '--enable-protocol', 'someip',
        '--enable-heuristic', 'someip_tcp_heur',
        '--enable-heuristic', 'someip_udp_heur',
        '-r', pcap, '-T', 'fields',
        '-E', 'header=y', '-E', 'separator=,', '-E', 'occurrence=f',
    ]
    for f in FIELDS:
        cmd += ['-e', f]

    res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    if res.returncode != 0:
        print(f'  ERRO: {res.stderr[:300]}')
        continue

    df = pd.read_csv(io.StringIO(res.stdout), low_memory=False)
    df.columns = COL

    for c in ['time','frame_len','ttl','tcp_dport','tcp_sport','udp_dport','udp_sport']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in ['svc_id','meth_id','msg_type','ret_code']:
        df[c] = df[c].astype(str).str.strip().str.lower()
        df[c] = df[c].where(df[c] != 'nan', other='')

    df_ip = df[df['src_ip'].notna() & (df['src_ip'] != '')]
    df_si = df[df['svc_id'] != ''].copy()

    ip_num   = pd.to_numeric(df['ip_proto'], errors='coerce')
    si_mask  = (df['proto'].str.contains('SOME/IP', na=False, case=False) &
                ~df['proto'].str.contains('SD',     na=False, case=False))

    n_si_tcp = int((si_mask & (ip_num == 6)).sum())
    n_sd     = int(df['proto'].str.contains('SOME/IP-SD', na=False, case=False).sum())
    n_si_udp = int((si_mask & (ip_num == 17)).sum())
    n_tcp    = int(df['tcp_dport'].notna().sum())
    n_tcp_c  = n_tcp - n_si_tcp
    n_igmp   = int((df['proto'] == 'IGMPv3').sum())
    n_arp    = int((df['proto'] == 'ARP').sum())

    df_si['mt'] = df_si['msg_type'].map(MSG).fillna(df_si['msg_type'])
    df_si['tb'] = df_si['time'].round(0).astype('Int64')
    ts = df_si.groupby('tb').size()
    dur = float(df['time'].max())
    peak = int(ts.max()) if len(ts) else 0

    # time series como lista de [t, count] para serialização JSON
    ts_list = [[int(t), int(c)] for t, c in ts.items()]

    results[label] = dict(
        n_frame   = len(df),
        dur       = dur,
        avg_pps   = round(len(df) / dur, 1) if dur > 0 else 0,
        peak_pps  = peak,
        n_si      = len(df_si),
        n_si_tcp  = n_si_tcp,
        n_si_udp  = n_si_udp,
        n_sd      = n_sd,
        n_tcp_c   = n_tcp_c,
        n_igmp    = n_igmp,
        n_arp     = n_arp,
        n_ips     = int(df_ip['src_ip'].nunique()),
        ttl       = sorted(df_ip['ttl'].dropna().unique().astype(int).tolist()),
        mt_counts = df_si['mt'].value_counts().to_dict(),
        svc_counts= df_si['svc_id'].value_counts().head(10).to_dict(),
        ts        = ts_list,
    )
    print(f'  {len(df):,} frames | {len(df_si):,} SOME/IP | dur={dur:.0f}s | peak={peak:,} pkt/s')

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pcap_stats.json')
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f'\nSalvo em: {OUT}')
print(f'PCAPs processados: {list(results.keys())}')
