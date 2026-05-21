#!/usr/bin/env python3
"""Diagnose why mitm_single_attacker.pcap yields 0 SOME/IP rows."""
import struct
from scapy.all import PcapReader, IP, TCP, UDP, Raw, Ether

PCAP_DIR = r'C:\Mestrado\SDV_Research\experiments\notebooks\data\pcap'
SOMEIP_VALID_MSG_TYPES = {0x00,0x01,0x02,0x40,0x41,0x42,0x80,0x81,0xC0,0xC1}

def is_valid_someip(b):
    if len(b) < 16: return False
    try:
        proto_ver = b[12]; msg_type = b[14]
        length = struct.unpack_from(">I", b, 4)[0]
    except: return False
    return (proto_ver == 0x01 and msg_type in SOMEIP_VALID_MSG_TYPES and length >= 8)

for pcap in ['mitm_single_attacker.pcap', 'mitm_multi_attacker.pcap']:
    print(f"\n{'='*60}")
    print(f"PCAP: {pcap}")
    counts = {'total':0,'no_ip':0,'no_tcp_udp':0,'no_raw':0,'is_valid':0,'invalid_si':0}
    proto_ver_hist = {}
    msg_type_hist  = {}
    layer_hist     = {}

    with PcapReader(f"{PCAP_DIR}\\{pcap}") as r:
        for i, pkt in enumerate(r):
            if i >= 5000: break
            counts['total'] += 1

            # Layer summary
            summary = pkt.summary()[:60]
            layers = []
            cur = pkt
            while cur:
                layers.append(type(cur).__name__)
                cur = cur.payload if hasattr(cur, 'payload') and cur.payload else None
                if cur and len(bytes(cur)) == 0: break
            layer_key = " / ".join(layers[:5])
            layer_hist[layer_key] = layer_hist.get(layer_key, 0) + 1

            if not pkt.haslayer(IP):
                counts['no_ip'] += 1
                continue
            if not (pkt.haslayer(TCP) or pkt.haslayer(UDP)):
                counts['no_tcp_udp'] += 1
                continue
            if not pkt.haslayer(Raw):
                counts['no_raw'] += 1
                continue

            raw = bytes(pkt[Raw].load)
            if len(raw) >= 16:
                pv = raw[12]; mt = raw[14]
                proto_ver_hist[pv] = proto_ver_hist.get(pv, 0) + 1
                msg_type_hist[mt]  = msg_type_hist.get(mt, 0) + 1

            if is_valid_someip(raw):
                counts['is_valid'] += 1
            else:
                counts['invalid_si'] += 1

    print(f"  Packets sampled : {counts['total']}")
    print(f"  No IP layer     : {counts['no_ip']}")
    print(f"  No TCP/UDP      : {counts['no_tcp_udp']}")
    print(f"  No Raw payload  : {counts['no_raw']}")
    print(f"  is_valid_someip : {counts['is_valid']}")
    print(f"  invalid someip  : {counts['invalid_si']}")
    print(f"\n  Top layer stacks:")
    for k,v in sorted(layer_hist.items(), key=lambda x:-x[1])[:8]:
        print(f"    {v:>5}  {k}")
    print(f"\n  proto_ver distribution (raw byte 12):")
    for k,v in sorted(proto_ver_hist.items(), key=lambda x:-x[1])[:8]:
        print(f"    0x{k:02x}  {v}")
    print(f"\n  msg_type distribution (raw byte 14):")
    for k,v in sorted(msg_type_hist.items(), key=lambda x:-x[1])[:8]:
        print(f"    0x{k:02x}  {v}")
