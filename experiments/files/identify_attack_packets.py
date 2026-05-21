#!/usr/bin/env python3
"""
Identifica quais pacotes dentro dos PCAPs de ataque são tráfego malicioso
vs tráfego legítimo — reproduzindo o esquema de rotulagem de Kim et al. (2026).

Estratégia:
  1. IPs do tráfego benigno = IPs legítimos dos ECUs
  2. IPs presentes em PCAPs de ataque mas AUSENTES no benigno = attacker IP
  3. Padrões por PCAP: rate, msg_type, service_id, SD ratio, payload anomaly
"""
import pandas as pd
import numpy as np

CSV = r'C:\Mestrado\SDV_Research\data\parsed_packets.csv'
CHUNK = 500_000

# ── 1. Coleta básica ───────────────────────────────────────────────────────────
print("Carregando metadados (src_ip, dst_ip, service_id, msg_type, is_sd, pcap_file)...")
cols = ['pcap_file', 'label', 'src_ip', 'dst_ip', 'service_id',
        'msg_type', 'is_sd', 'ip_ttl', 'transport', 'ip_len']
chunks = []
for c in pd.read_csv(CSV, usecols=cols, chunksize=CHUNK, low_memory=False):
    chunks.append(c)
df = pd.concat(chunks, ignore_index=True)
print(f"Total: {len(df):,} rows\n")

# ── 2. IPs legítimos (benigno) ────────────────────────────────────────────────
benign = df[df['pcap_file'] == 'benign_traffic.pcap']
legit_src = set(benign['src_ip'].dropna().unique())
legit_dst = set(benign['dst_ip'].dropna().unique())
legit_ips = legit_src | legit_dst
legit_svcs = set(benign['service_id'].dropna().unique())
legit_ttls = set(benign['ip_ttl'].dropna().unique())

print("=== IPs legítimos (benigno) ===")
print(f"  src IPs únicos: {sorted(legit_src)}")
print(f"  dst IPs únicos: {sorted(legit_dst)}")
print(f"  Service IDs   : {sorted([f'0x{int(s):04x}' for s in legit_svcs if not pd.isna(s)])}")
print(f"  TTLs          : {sorted(legit_ttls)}")
print(f"  msg_types     : {sorted(benign['msg_type'].dropna().unique())}")

# ── 3. Por PCAP de ataque: identifica IPs suspeitos e padrões ────────────────
ATTACK_PCAPS = [
    'dos_noti_flood.pcap',
    'fuzzy_sd_offer_rand_noti(1).pcap',
    'fuzzy_sd_offer_rand_noti(2).pcap',
    'fuzzy_sd_offer_rand_noti(3).pcap',
    'mitm_multi_attacker.pcap',
    'mitm_single_attacker.pcap',
]

results = {}

for pcap in ATTACK_PCAPS:
    sub = df[df['pcap_file'] == pcap]
    total = len(sub)
    print(f"\n{'='*60}")
    print(f"PCAP: {pcap}  ({total:,} packets)")
    print(f"{'='*60}")

    # IPs no ataque
    src_ips = sub['src_ip'].dropna().unique()
    dst_ips = sub['dst_ip'].dropna().unique()
    all_pcap_ips = set(src_ips) | set(dst_ips)
    attacker_ips = all_pcap_ips - legit_ips
    new_dst = set(dst_ips) - legit_dst

    print(f"\n  src IPs: {sorted(set(src_ips))}")
    print(f"  dst IPs: {sorted(set(dst_ips))}")
    print(f"  IPs NOVOS (não em benigno): {sorted(attacker_ips)}")
    print(f"  dst NOVOS: {sorted(new_dst)}")

    # Service IDs no ataque
    svc_counts = sub['service_id'].dropna().value_counts()
    new_svcs = set(sub['service_id'].dropna().unique()) - legit_svcs
    print(f"\n  Service IDs (top5):")
    for svc, cnt in svc_counts.head(5).items():
        flag = " ** NOVO" if svc in new_svcs else ""
        print(f"    0x{int(svc):04x}: {cnt:,}{flag}")

    # TTLs
    ttls = sub['ip_ttl'].dropna().unique()
    new_ttls = set(ttls) - legit_ttls
    print(f"\n  TTLs: {sorted(ttls)}  (novos: {sorted(new_ttls)})")

    # msg_type
    msg_dist = sub['msg_type'].value_counts()
    print(f"\n  msg_type dist:")
    for mt, cnt in msg_dist.items():
        print(f"    {mt}: {cnt:,}  ({100*cnt/total:.1f}%)")

    # SD ratio
    sd_count = sub['is_sd'].fillna(False).astype(bool).sum()
    print(f"\n  SD ratio: {sd_count:,}/{total:,} = {100*sd_count/total:.1f}%")

    # Pacotes por src_ip
    print(f"\n  Pacotes por src_ip:")
    for ip, cnt in sub['src_ip'].value_counts().items():
        flag = " ** ATTACKER?" if ip not in legit_src else ""
        print(f"    {ip}: {cnt:,}{flag}")

    # Candidatos a 'attack label'
    # Critério 1: src_ip não está nos IPs legítimos
    mask_new_ip = sub['src_ip'].isin(attacker_ips)
    # Critério 2: service_id não visto no benigno
    mask_new_svc = sub['service_id'].isin(new_svcs)
    # Critério 3: TTL diferente dos legítimos
    mask_new_ttl = ~sub['ip_ttl'].isin(legit_ttls)

    n_new_ip  = mask_new_ip.sum()
    n_new_svc = mask_new_svc.sum()
    n_new_ttl = mask_new_ttl.sum()
    n_any     = (mask_new_ip | mask_new_svc | mask_new_ttl).sum()

    print(f"\n  Candidatos a 'attack' por critério:")
    print(f"    src_ip novo   : {n_new_ip:,}  ({100*n_new_ip/total:.1f}%)")
    print(f"    service_id novo: {n_new_svc:,}  ({100*n_new_svc/total:.1f}%)")
    print(f"    TTL novo      : {n_new_ttl:,}  ({100*n_new_ttl/total:.1f}%)")
    print(f"    qualquer      : {n_any:,}  ({100*n_any/total:.1f}%)")

    results[pcap] = {
        'total': total,
        'attacker_ips': sorted(attacker_ips),
        'new_svcs': sorted([f'0x{int(s):04x}' for s in new_svcs]),
        'new_ttls': sorted(new_ttls),
        'n_attack_ip': int(n_new_ip),
        'n_attack_svc': int(n_new_svc),
        'n_attack_ttl': int(n_new_ttl),
        'n_attack_any': int(n_any),
    }

# ── 4. Resumo ─────────────────────────────────────────────────────────────────
print(f"\n\n{'='*80}")
print("RESUMO — candidatos a rotulagem 'attack'")
print(f"{'='*80}")
total_normal = len(benign)
total_attack_candidate = 0
for pcap, r in results.items():
    best_n = r['n_attack_any']
    total_attack_candidate += best_n
    pct = 100 * best_n / r['total']
    print(f"  {pcap:<42}  total={r['total']:>9,}  attack≈{best_n:>7,} ({pct:.1f}%)")

total_normal_in_attacks = sum(r['total'] - r['n_attack_any'] for r in results.values())
print(f"\n  Benigno (tudo normal)       :  {total_normal:>9,}")
print(f"  Normal dentro de ataques    :  {total_normal_in_attacks:>9,}")
print(f"  TOTAL NORMAL estimado       :  {total_normal + total_normal_in_attacks:>9,}")
print(f"  TOTAL ATTACK estimado       :  {total_attack_candidate:>9,}")
print(f"\n  Kim et al. (referência):")
print(f"    normal : 12,571,029")
print(f"    attack :  1,662,318")
