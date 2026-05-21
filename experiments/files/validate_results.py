#!/usr/bin/env python3
"""Validate parsed_packets.csv against tshark reference (pcap_stats.json)."""
import pandas as pd, json

REF_PATH = r"C:\Mestrado\SDV_Research\data_exploration\pcap_analysis\pcap_stats.json"
BENIGN   = r"C:\Mestrado\SDV_Research\data\parsed_benign_test.csv"
ALL_CSV  = r"C:\Mestrado\SDV_Research\data\parsed_packets.csv"

with open(REF_PATH, encoding="utf-8") as f:
    REF = json.load(f)

LABEL_MAP = {
    "benign_traffic.pcap":              "benign",
    "dos_noti_flood.pcap":              "dos_noti",
    "fuzzy_sd_offer_rand_noti(1).pcap": "fuzzy(1)",
    "fuzzy_sd_offer_rand_noti(2).pcap": "fuzzy(2)",
    "fuzzy_sd_offer_rand_noti(3).pcap": "fuzzy(3)",
    "mitm_multi_attacker.pcap":         "mitm_multi",
    "mitm_single_attacker.pcap":        "mitm_single",
}

# ---- Benign quick check ----
print("=== 1. VALIDAÇÃO BENIGNO ===")
df = pd.read_csv(BENIGN, low_memory=False)
ref = REF["benign"]
p_si = int(df["someip_valid"].sum())
p_sd = int(df["is_sd"].fillna(False).sum())
r_si, r_sd = ref["n_si"], ref["n_sd"]
d_si, d_sd = p_si - r_si, p_sd - r_sd

def pct(d, r): return d / r * 100 if r else 0
def mark(d, r, tol): return "OK" if abs(d)/max(r,1) < tol else "FALHOU"

print(f"Linhas totais   : {len(df):,}")
print(f"SOME/IP frames  : ref={r_si:,}  parser={p_si:,}  diff={d_si:+,}  ({pct(d_si,r_si):+.3f}%)  {mark(d_si,r_si,0.005)}")
print(f"SOME/IP-SD      : ref={r_sd:,}  parser={p_sd:,}  diff={d_sd:+,}  ({pct(d_sd,r_sd):+.3f}%)  {mark(d_sd,r_sd,0.01)}")

si = df[df["someip_valid"] == True]
print(f"\nPayloads (em frames SOME/IP validos={len(si):,}):")
print(f"  transport_payload_hex not-null : {si['transport_payload_hex'].notna().sum():,}")
print(f"  someip_payload_hex    not-null : {si['someip_payload_hex'].notna().sum():,}")

parsed_svc = (si.assign(s=lambda x: x["service_id"].apply(
    lambda v: f"0x{int(v):04x}" if pd.notna(v) else None))["s"].value_counts().to_dict())
all_svcs = sorted(set(ref["svc_counts"]) | set(parsed_svc))
print("\nService IDs — tshark vs parser:")
for svc in all_svcs:
    rv = ref["svc_counts"].get(svc, 0)
    pv = parsed_svc.get(svc, 0)
    d  = pv - rv
    m  = "OK" if abs(d)/max(rv,1) < 0.01 else "DIFF"
    print(f"  {svc:<8}  tshark={rv:>10,}  parser={pv:>10,}  {d:>+8,}  {m}")

# ---- Full validation ----
print("\n\n=== 2. VALIDAÇÃO COMPLETA (todos os PCAPs) ===")
df_all = pd.read_csv(ALL_CSV, low_memory=False)
print(f"Linhas totais: {len(df_all):,}")
print(f"Colunas: {list(df_all.columns)}\n")

hdr = f"{'PCAP':<14} {'tshark_si':>11} {'parser_si':>11} {'diff_si':>8} {'pct_si':>7}  {'tshark_sd':>10} {'parser_sd':>10} {'diff_sd':>8} {'pct_sd':>7}  OK?"
print(hdr)
print("-" * len(hdr))

all_ok = True
for pcap_file, ref_label in LABEL_MAP.items():
    sub = df_all[df_all["pcap_file"] == pcap_file]
    p_si = int(sub["someip_valid"].sum())
    p_sd = int(sub["is_sd"].fillna(False).sum())
    ref  = REF.get(ref_label, {})
    r_si = ref.get("n_si", 0); r_sd = ref.get("n_sd", 0)
    d_si = p_si - r_si;        d_sd = p_sd - r_sd
    ok_si = abs(d_si)/max(r_si,1) < 0.005
    ok_sd = abs(d_sd)/max(r_sd,1) < 0.01
    ok    = ok_si and ok_sd
    all_ok = all_ok and ok
    print(f"{ref_label:<14} {r_si:>11,} {p_si:>11,} {d_si:>+8,} {pct(d_si,r_si):>+6.2f}%  "
          f"{r_sd:>10,} {p_sd:>10,} {d_sd:>+8,} {pct(d_sd,r_sd):>+6.2f}%  {'OK' if ok else 'DIFF'}")

print(f"\nRESULTADO GLOBAL: {'APROVADO' if all_ok else 'VERIFICAR DIFFS'}")

# ---- Inspect payloads ----
print("\n\n=== 3. INSPEÇÃO DE PAYLOADS ===")
si_all = df_all[df_all["someip_valid"] == True]
print(f"Total SOME/IP validos: {len(si_all):,}")
print(f"  transport_payload_hex not-null: {si_all['transport_payload_hex'].notna().sum():,}")
print(f"  someip_payload_hex    not-null: {si_all['someip_payload_hex'].notna().sum():,}")
print("\nmsg_type distribuicao:")
print(si_all["msg_type"].value_counts().head(10).to_string())
print("\nservice_id top10:")
print(si_all["service_id"].apply(lambda v: f"0x{int(v):04x}" if pd.notna(v) else None)
      .value_counts().head(10).to_string())
