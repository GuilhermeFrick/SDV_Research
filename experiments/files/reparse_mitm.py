#!/usr/bin/env python3
"""
Re-run parser for mitm_multi + mitm_single, patch parsed_packets.csv.

Steps:
1. Parse only the two MITM PCAPs -> temp CSV
2. Remove stale mitm rows from parsed_packets.csv
3. Append new rows
"""
import importlib.util, sys, pandas as pd
from pathlib import Path

ROOT     = Path(r'C:\Mestrado\SDV_Research')
SCRIPTS  = ROOT / 'experiments' / 'files'
PCAP_DIR = ROOT / 'experiments' / 'notebooks' / 'data' / 'pcap'
MAIN_CSV = ROOT / 'data' / 'parsed_packets.csv'
TEMP_CSV = ROOT / 'data' / 'parsed_mitm_patch.csv'

spec = importlib.util.spec_from_file_location('parse_pcap', SCRIPTS / '01_parse_pcap.py')
parse_pcap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parse_pcap)

MITM_FILES = ['mitm_multi_attacker.pcap', 'mitm_single_attacker.pcap']

print("Step 1 — Parsing MITM PCAPs...")
parse_pcap.process_all_pcaps(
    pcap_dir=str(PCAP_DIR),
    output_csv=str(TEMP_CSV),
    pcap_filter=MITM_FILES,
)

print("\nStep 2 — Loading existing CSV (mitm rows removed)...")
df_main = pd.read_csv(MAIN_CSV, low_memory=False)
print(f"  Rows before patch: {len(df_main):,}")
df_clean = df_main[~df_main['pcap_file'].isin(MITM_FILES)].copy()
print(f"  Rows after removing stale mitm: {len(df_clean):,}")

print("\nStep 3 — Loading new MITM rows...")
df_mitm = pd.read_csv(TEMP_CSV, low_memory=False)
print(f"  New MITM rows: {len(df_mitm):,}")

print("\nStep 4 — Concatenating and saving...")
df_all = pd.concat([df_clean, df_mitm], ignore_index=True)
df_all.to_csv(MAIN_CSV, index=False)
print(f"  Total rows written: {len(df_all):,}")
print(f"  Saved to: {MAIN_CSV}")

# Quick validation
print("\n=== Quick validation ===")
import json
with open(ROOT / 'data_exploration' / 'pcap_analysis' / 'pcap_stats.json', encoding='utf-8') as f:
    REF = json.load(f)

LABEL_MAP = {
    'mitm_multi_attacker.pcap':  'mitm_multi',
    'mitm_single_attacker.pcap': 'mitm_single',
}
for pcap_file, ref_label in LABEL_MAP.items():
    sub = df_all[df_all['pcap_file'] == pcap_file]
    p_si = int(sub['someip_valid'].sum())
    p_sd = int(sub['is_sd'].fillna(False).sum())
    ref  = REF[ref_label]
    r_si = ref['n_si']; r_sd = ref['n_sd']
    d_si = p_si - r_si; d_sd = p_sd - r_sd
    pct_si = d_si/r_si*100; pct_sd = d_sd/r_sd*100
    ok = abs(d_si)/r_si < 0.005 and abs(d_sd)/r_sd < 0.01
    print(f"{ref_label:<14}  n_si: ref={r_si:,} parser={p_si:,} ({pct_si:+.3f}%)  "
          f"n_sd: ref={r_sd:,} parser={p_sd:,} ({pct_sd:+.3f}%)  {'OK' if ok else 'DIFF'}")
