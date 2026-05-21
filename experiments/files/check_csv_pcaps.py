#!/usr/bin/env python3
"""Quick row-count per pcap_file in parsed_packets.csv."""
import pandas as pd

df = pd.read_csv(r'C:\Mestrado\SDV_Research\data\parsed_packets.csv',
                 usecols=['pcap_file', 'someip_valid', 'is_sd'], low_memory=False)
print(f"Total rows: {len(df):,}\n")
g = df.groupby('pcap_file').agg(
    rows=('someip_valid', 'count'),
    si=('someip_valid', 'sum'),
    sd=('is_sd', lambda x: x.fillna(False).sum())
).reset_index()
for _, row in g.iterrows():
    print(f"{row['pcap_file']:<40}  rows={int(row['rows']):>9,}  si={int(row['si']):>9,}  sd={int(row['sd']):>8,}")
