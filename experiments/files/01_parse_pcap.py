"""
SOME/IP IDS - Etapa 1: Parsing de PCAPs em Camadas
====================================================
Reprodução de Kim et al. (2026) - Seção 5.1 (Layered Packet Extraction)

Lê os arquivos PCAP do dataset público (Figshare) e extrai registros
por camada: IP, TCP/UDP e SOME/IP (incluindo SOME/IP-SD).

Saída: CSV com todos os pacotes parseados e rotulados.

Dataset: https://figshare.com/articles/dataset/30970450
Arquivos esperados na pasta data/pcap/:
  - benign_traffic.pcap
  - dos_noti_flood.pcap
  - fuzzy_sd_offer_rand_noti(1).pcap
  - fuzzy_sd_offer_rand_noti(2).pcap
  - fuzzy_sd_offer_rand_noti(3).pcap
  - mitm_multi_attacker.pcap
  - mitm_single_attacker.pcap
"""

import os
import struct
import socket
import csv
from pathlib import Path

# ── Tenta importar scapy (parser de rede) ─────────────────────────────────────
try:
    from scapy.all import PcapReader, IP, TCP, UDP, Raw, Ether
    SCAPY_OK = True
except ImportError:
    SCAPY_OK = False
    print("[AVISO] scapy nao encontrado. Instale com:  pip install scapy")

# ── Constantes do protocolo SOME/IP ───────────────────────────────────────────
SOMEIP_PORT       = 30490          # Porta padrão SOME/IP-SD
SOMEIP_MIN_LEN    = 16             # Tamanho mínimo do cabeçalho SOME/IP (bytes)
SOMEIP_SD_SERVICE = 0xFFFF         # Service ID reservado para SOME/IP-SD
MSG_TYPE_NAMES = {
    0x00: "REQUEST",
    0x01: "REQUEST_NO_RETURN",
    0x02: "NOTIFICATION",
    0x80: "RESPONSE",
    0x81: "ERROR",
}

# ── Mapeamento de arquivos → rótulo ───────────────────────────────────────────
PCAP_LABEL_MAP = {
    "benign_traffic.pcap":               "normal",
    "dos_noti_flood.pcap":               "dos",
    "fuzzy_sd_offer_rand_noti(1).pcap":  "fuzzy",
    "fuzzy_sd_offer_rand_noti(2).pcap":  "fuzzy",
    "fuzzy_sd_offer_rand_noti(3).pcap":  "fuzzy",
    "mitm_multi_attacker.pcap":          "mitm",
    "mitm_single_attacker.pcap":         "mitm",
}


# ══════════════════════════════════════════════════════════════════════════════
# Funções de parsing SOME/IP
# ══════════════════════════════════════════════════════════════════════════════

def parse_someip_header(payload_bytes):
    """
    Faz o parsing do cabeçalho SOME/IP (16 bytes fixos).

    Estrutura do cabeçalho SOME/IP (big-endian):
      Bytes 0-1  : Service ID     (uint16)
      Bytes 2-3  : Method/Event ID (uint16)
      Bytes 4-7  : Length         (uint32) — comprimento do restante da mensagem
      Bytes 8-9  : Client ID      (uint16)
      Bytes 10-11: Session ID     (uint16)
      Byte  12   : Protocol Ver.  (uint8)
      Byte  13   : Interface Ver. (uint8)
      Byte  14   : Message Type   (uint8)
      Byte  15   : Return Code    (uint8)

    Retorna dict com os campos ou None se o payload for curto demais.
    """
    if len(payload_bytes) < SOMEIP_MIN_LEN:
        return None

    try:
        service_id, method_id, length = struct.unpack_from(">HHI", payload_bytes, 0)
        client_id, session_id         = struct.unpack_from(">HH",  payload_bytes, 8)
        proto_ver, iface_ver          = struct.unpack_from(">BB",  payload_bytes, 12)
        msg_type, return_code         = struct.unpack_from(">BB",  payload_bytes, 14)
        someip_payload = payload_bytes[SOMEIP_MIN_LEN:]

        return {
            "service_id":    service_id,
            "method_id":     method_id,
            "length":        length,
            "client_id":     client_id,
            "session_id":    session_id,
            "proto_ver":     proto_ver,
            "iface_ver":     iface_ver,
            "msg_type":      msg_type,
            "msg_type_name": MSG_TYPE_NAMES.get(msg_type, f"0x{msg_type:02X}"),
            "return_code":   return_code,
            "is_sd":         service_id == SOMEIP_SD_SERVICE,
            "payload_bytes": someip_payload,
            "payload_hex":   someip_payload[:32].hex(),  # primeiros 32 bytes para debug
        }
    except struct.error:
        return None


def is_someip_port(sport, dport):
    """Verifica se a porta corresponde a tráfego SOME/IP típico.

    30490       = SOME/IP-SD multicast (padrão AUTOSAR)
    30491-30510 = portas de serviço usadas pelo vSomeIP neste dataset
    """
    return (30490 <= sport <= 30510) or (30490 <= dport <= 30510)


# ══════════════════════════════════════════════════════════════════════════════
# Parsing principal por pacote
# ══════════════════════════════════════════════════════════════════════════════

def parse_packet(pkt, label, pcap_file):
    """
    Extrai campos de todas as camadas de um pacote Scapy.

    Retorna um dict com colunas que serão salvas no CSV,
    ou None se o pacote não for SOME/IP válido.
    """
    if not pkt.haslayer(IP):
        return None

    ip   = pkt[IP]
    ts   = float(pkt.time)

    # ── Camada IP ─────────────────────────────────────────────────────────────
    record = {
        "timestamp":   ts,
        "src_ip":      ip.src,
        "dst_ip":      ip.dst,
        "ip_proto":    ip.proto,
        "ip_ttl":      ip.ttl,
        "ip_len":      ip.len,
        "ip_id":       ip.id,
        "ip_flags":    int(ip.flags),
        "transport":   None,
        "src_port":    None,
        "dst_port":    None,
        "transport_len": None,
        "tcp_seq":     None,
        "tcp_ack":     None,
        "tcp_flags":   None,
        # SOME/IP
        "someip_valid":  False,
        "service_id":    None,
        "method_id":     None,
        "someip_len":    None,
        "client_id":     None,
        "session_id":    None,
        "proto_ver":     None,
        "iface_ver":     None,
        "msg_type":      None,
        "msg_type_name": None,
        "return_code":   None,
        "is_sd":         None,
        "payload_hex":   None,
        "someip_payload_len": None,
        # Metadados
        "label":       label,
        "pcap_file":   pcap_file,
    }

    raw_payload = None

    # ── Camada TCP ────────────────────────────────────────────────────────────
    if pkt.haslayer(TCP):
        tcp = pkt[TCP]
        record.update({
            "transport":    "TCP",
            "src_port":     tcp.sport,
            "dst_port":     tcp.dport,
            "transport_len": len(tcp),
            "tcp_seq":      tcp.seq,
            "tcp_ack":      tcp.ack,
            "tcp_flags":    int(tcp.flags),
        })
        if pkt.haslayer(Raw):
            raw_payload = bytes(pkt[Raw].load)

    # ── Camada UDP ────────────────────────────────────────────────────────────
    elif pkt.haslayer(UDP):
        udp = pkt[UDP]
        record.update({
            "transport":    "UDP",
            "src_port":     udp.sport,
            "dst_port":     udp.dport,
            "transport_len": udp.len,
        })
        if pkt.haslayer(Raw):
            raw_payload = bytes(pkt[Raw].load)
    else:
        return None  # sem camada de transporte relevante

    # ── Filtra apenas pacotes SOME/IP ─────────────────────────────────────────
    if not is_someip_port(record["src_port"], record["dst_port"]):
        return None

    # ── Parsing SOME/IP ───────────────────────────────────────────────────────
    if raw_payload:
        sh = parse_someip_header(raw_payload)
        if sh:
            record.update({
                "someip_valid":       True,
                "service_id":         sh["service_id"],
                "method_id":          sh["method_id"],
                "someip_len":         sh["length"],
                "client_id":          sh["client_id"],
                "session_id":         sh["session_id"],
                "proto_ver":          sh["proto_ver"],
                "iface_ver":          sh["iface_ver"],
                "msg_type":           sh["msg_type"],
                "msg_type_name":      sh["msg_type_name"],
                "return_code":        sh["return_code"],
                "is_sd":              sh["is_sd"],
                "payload_hex":        sh["payload_hex"],
                "someip_payload_len": len(sh["payload_bytes"]),
            })
        # Aceita pacotes com porto SOME/IP mesmo sem header válido (tráfego SD, etc.)

    return record


# ══════════════════════════════════════════════════════════════════════════════
# Processamento de todos os PCAPs
# ══════════════════════════════════════════════════════════════════════════════

def process_all_pcaps(pcap_dir: str, output_csv: str):
    """
    Itera sobre todos os PCAPs mapeados, faz o parsing e salva um CSV único.

    Parâmetros
    ----------
    pcap_dir   : pasta contendo os arquivos .pcap
    output_csv : caminho do arquivo de saída
    """
    if not SCAPY_OK:
        raise RuntimeError("scapy é necessário. Execute: pip install scapy")

    pcap_dir  = Path(pcap_dir)
    out_path  = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_pkts   = 0
    total_parsed = 0
    rows_written = 0

    # Colunas do CSV (ordem fixa para reproducibilidade)
    COLUMNS = [
        "timestamp", "src_ip", "dst_ip", "ip_proto", "ip_ttl", "ip_len",
        "ip_id", "ip_flags", "transport", "src_port", "dst_port",
        "transport_len", "tcp_seq", "tcp_ack", "tcp_flags",
        "someip_valid", "service_id", "method_id", "someip_len",
        "client_id", "session_id", "proto_ver", "iface_ver",
        "msg_type", "msg_type_name", "return_code", "is_sd",
        "payload_hex", "someip_payload_len",
        "label", "pcap_file",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=COLUMNS)
        writer.writeheader()

        for pcap_name, label in PCAP_LABEL_MAP.items():
            pcap_path = pcap_dir / pcap_name
            if not pcap_path.exists():
                print(f"  [PULANDO] Arquivo não encontrado: {pcap_path}")
                continue

            print(f"\n[>>] Processando: {pcap_name}  (rotulo={label})")
            n_pkts = 0
            n_parsed = 0
            try:
                with PcapReader(str(pcap_path)) as reader:
                    for pkt in reader:
                        n_pkts += 1
                        rec = parse_packet(pkt, label=label, pcap_file=pcap_name)
                        if rec:
                            row = {col: rec.get(col, None) for col in COLUMNS}
                            writer.writerow(row)
                            n_parsed += 1
                            rows_written += 1
                        if n_pkts % 100_000 == 0:
                            print(f"  ... {n_pkts:,} pkts lidos, {n_parsed:,} SOME/IP")
            except Exception as e:
                print(f"  [ERRO] Falha ao ler PCAP: {e}")
                continue

            total_pkts   += n_pkts
            total_parsed += n_parsed
            print(f"  Pacotes totais: {n_pkts:>8,} | SOME/IP extraidos: {n_parsed:>8,} "
                  f"({100*n_parsed/max(n_pkts,1):.1f}%)")

    print(f"\n{'='*60}")
    print(f"CONCLUÍDO")
    print(f"  Pacotes lidos    : {total_pkts:>10,}")
    print(f"  Registros SOME/IP: {total_parsed:>10,}")
    print(f"  Linhas no CSV    : {rows_written:>10,}")
    print(f"  Saída            : {out_path}")
    return str(out_path)


# ══════════════════════════════════════════════════════════════════════════════
# Ponto de entrada
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="SOME/IP PCAP Parser (Kim et al. 2026)")
    ap.add_argument("--pcap-dir",  default=r"C:\Mestrado\SDV_Research\data\dataset_ism_xgboost",
                    help="Pasta com os arquivos .pcap")
    ap.add_argument("--output",    default=r"C:\Mestrado\SDV_Research\data\parsed_packets.csv",
                    help="CSV de saída")
    args = ap.parse_args()

    process_all_pcaps(args.pcap_dir, args.output)
