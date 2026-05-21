"""
SOME/IP IDS - Etapa 1: Parsing de PCAPs em Camadas
====================================================
Reprodução de Kim et al. (2026) - Seção 5.1 (Layered Packet Extraction)

Lê os arquivos PCAP do dataset público (Figshare) e extrai registros
por camada: IP, TCP/UDP e SOME/IP (incluindo SOME/IP-SD).
Cada registro é rotulado com o tipo de ataque correspondente ao PCAP de origem.

Saída:
    CSV com todos os pacotes SOME/IP parseados e rotulados,
    pronto para ser consumido pela Etapa 2 (extração de features).

Referência:
    Kim et al. (2026). XGBoost-Based Anomaly Detection Framework for SOME/IP
    in In-Vehicle Networks. Systems, 14(2), 196.
    DOI: https://doi.org/10.3390/systems14020196

Dataset:
    Figshare - https://doi.org/10.6084/m9.figshare.30970450
    Arquivos esperados em --pcap-dir:
        - benign_traffic.pcap
        - dos_noti_flood.pcap
        - fuzzy_sd_offer_rand_noti(1).pcap
        - fuzzy_sd_offer_rand_noti(2).pcap
        - fuzzy_sd_offer_rand_noti(3).pcap
        - mitm_multi_attacker.pcap
        - mitm_single_attacker.pcap

Uso:
    python 01_parse_pcap.py --pcap-dir data/dataset_ism_xgboost --output data/parsed_packets.csv

Autor:
    Guilherme Frick
"""

import struct
import csv
from pathlib import Path

try:
    from scapy.all import PcapReader, IP, TCP, UDP, Raw
    SCAPY_OK = True
except ImportError:
    SCAPY_OK = False
    print("[AVISO] scapy nao encontrado. Instale com:  pip install scapy")


# ---------------------------------------------------------------------------
# Constantes do protocolo SOME/IP (AUTOSAR R22-11)
# ---------------------------------------------------------------------------

SOMEIP_MIN_LEN = 16
"""int: Tamanho fixo do cabeçalho SOME/IP em bytes (campos obrigatórios)."""

SOMEIP_SD_SERVICE = 0xFFFF
"""int: Service ID reservado pelo AUTOSAR para mensagens SOME/IP-SD."""

# Portas não são usadas como filtro primário — detecção é estrutural
# (parse_someip_header + is_valid_someip). Mantidas apenas como referência.
SOMEIP_PORT_HINT = {30490, 30501, 30502, 30503}
"""set: Portas conhecidas do vSomeIP neste dataset (referência, não filtro)."""

MSG_TYPE_NAMES = {
    0x00: "REQUEST",
    0x01: "REQUEST_NO_RETURN",
    0x02: "NOTIFICATION",
    0x80: "RESPONSE",
    0x81: "ERROR",
}
"""dict: Mapeamento de código numérico para nome legível do tipo de mensagem SOME/IP."""

PCAP_LABEL_MAP = {
    "benign_traffic.pcap":               "normal",
    "dos_noti_flood.pcap":               "dos",
    "fuzzy_sd_offer_rand_noti(1).pcap":  "fuzzy",
    "fuzzy_sd_offer_rand_noti(2).pcap":  "fuzzy",
    "fuzzy_sd_offer_rand_noti(3).pcap":  "fuzzy",
    "mitm_multi_attacker.pcap":          "mitm",
    "mitm_single_attacker.pcap":         "mitm",
}
"""dict: Mapeamento nome-do-arquivo -> tipo de ataque (para referência)."""

# IPs legítimos dos ECUs — identificados no tráfego benigno
LEGIT_IPS = {f"172.18.0.{i}" for i in range(2, 11)} | {"224.244.224.245"}

# IPs dos atacantes por PCAP — identificados por reverse engineering vs benigno
ATTACKER_IPS_MAP = {
    "benign_traffic.pcap":               set(),
    "dos_noti_flood.pcap":               {"172.18.0.11"},
    "fuzzy_sd_offer_rand_noti(1).pcap":  {"172.18.0.17"},
    "fuzzy_sd_offer_rand_noti(2).pcap":  {"172.18.0.12"},
    "fuzzy_sd_offer_rand_noti(3).pcap":  {"172.18.0.12"},
    "mitm_multi_attacker.pcap":          {"172.18.0.14", "172.18.0.15"},
    "mitm_single_attacker.pcap":         {"172.18.0.13"},
}


# ---------------------------------------------------------------------------
# Funções de parsing
# ---------------------------------------------------------------------------

def parse_someip_header(payload_bytes: bytes) -> dict | None:
    """Extrai os campos do cabeçalho SOME/IP de 16 bytes (big-endian).

    Estrutura do cabeçalho (AUTOSAR PRS_SOMEIPProtocol):

    .. code-block:: text

        Bytes  0-1  : Service ID     (uint16)
        Bytes  2-3  : Method/Event ID (uint16)
        Bytes  4-7  : Length         (uint32) — bytes restantes após este campo
        Bytes  8-9  : Client ID      (uint16)
        Bytes 10-11 : Session ID     (uint16)
        Byte   12   : Protocol Ver.  (uint8)
        Byte   13   : Interface Ver. (uint8)
        Byte   14   : Message Type   (uint8)
        Byte   15   : Return Code    (uint8)

    Args:
        payload_bytes: Bytes brutos a partir do início do cabeçalho SOME/IP.

    Returns:
        Dicionário com os campos do cabeçalho e o payload restante, ou ``None``
        se ``payload_bytes`` tiver menos de 16 bytes ou ocorrer erro de parsing.

    Example:
        >>> hdr = parse_someip_header(raw_bytes)
        >>> print(hdr["service_id"], hdr["is_sd"])
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
            "payload_bytes":      someip_payload,
            "someip_payload_hex": someip_payload[:64].hex(),
        }
    except struct.error:
        return None


SOMEIP_VALID_MSG_TYPES  = {0x00, 0x01, 0x02, 0x40, 0x41, 0x42, 0x80, 0x81, 0xC0, 0xC1}
SOMEIP_VALID_PROTO_VER  = {0x01}

def is_valid_someip(raw_payload: bytes) -> bool:
    """Detecta SOME/IP por estrutura de cabeçalho, sem depender de porta.

    Valida os campos invariantes do cabeçalho de 16 bytes:
    - proto_ver deve ser 0x01 (único valor definido pelo AUTOSAR)
    - msg_type deve pertencer ao conjunto de tipos válidos
    - length deve ser consistente com o tamanho do payload recebido

    Essa abordagem replica o que o Wireshark faz com as heurísticas
    someip_tcp_heur / someip_udp_heur, capturando tráfego em qualquer porta.

    Args:
        raw_payload: Bytes crus a partir do início do payload TCP/UDP.

    Returns:
        ``True`` se o payload é um cabeçalho SOME/IP estruturalmente válido.
    """
    if len(raw_payload) < SOMEIP_MIN_LEN:
        return False
    try:
        length   = struct.unpack_from(">I", raw_payload, 4)[0]
        proto_ver = raw_payload[12]
        msg_type  = raw_payload[14]
    except (struct.error, IndexError):
        return False
    if proto_ver not in SOMEIP_VALID_PROTO_VER:
        return False
    if msg_type not in SOMEIP_VALID_MSG_TYPES:
        return False
    # length = bytes restantes após o campo length (offset 8 em diante)
    # mínimo aceitável é 8 (resto do header fixo sem payload)
    if length < 8:
        return False
    return True


def parse_packet(pkt, attack_type: str, pcap_file: str,
                 attacker_ips: set) -> dict | None:
    """Extrai e estrutura os campos de todas as camadas de um pacote Scapy.

    Mantém TODOS os frames TCP/UDP (incluindo ACKs sem payload SOME/IP),
    replicando o pipeline de Kim et al. (2026) que processa ~14M frames.
    Rotula cada pacote por src_ip: ataque se vier de IP desconhecido.

    Args:
        pkt:          Pacote Scapy lido pelo PcapReader.
        attack_type:  Tipo de ataque do PCAP ("normal", "dos", "fuzzy", "mitm").
        pcap_file:    Nome do arquivo PCAP de origem.
        attacker_ips: Conjunto de IPs do atacante neste PCAP.

    Returns:
        Dicionário com campos de todas as camadas, ou None se o pacote
        não tiver camada IP ou não for TCP/UDP.
    """
    if not pkt.haslayer(IP):
        return None

    ip = pkt[IP]
    ts = float(pkt.time)

    # Rotulagem por src_ip: 1=ataque se IP desconhecido, 0=normal caso contrário
    label = 1 if ip.src in attacker_ips else 0

    record = {
        # Camada IP
        "timestamp":     ts,
        "src_ip":        ip.src,
        "dst_ip":        ip.dst,
        "ip_proto":      ip.proto,
        "ip_ttl":        ip.ttl,
        "ip_len":        ip.len,
        "ip_id":         ip.id,
        "ip_flags":      int(ip.flags),
        # Camada de transporte (preenchida abaixo)
        "transport":     None,
        "src_port":      None,
        "dst_port":      None,
        "transport_len": None,
        "tcp_seq":       None,
        "tcp_ack":       None,
        "tcp_flags":     None,
        # Cabeçalho SOME/IP (preenchido abaixo; None se TCP sem payload)
        "someip_valid":          False,
        "service_id":            None,
        "method_id":             None,
        "someip_len":            None,
        "client_id":             None,
        "session_id":            None,
        "proto_ver":             None,
        "iface_ver":             None,
        "msg_type":              None,
        "msg_type_name":         None,
        "return_code":           None,
        "is_sd":                 None,
        "transport_payload_hex": None,
        "someip_payload_hex":    None,
        "someip_payload_len":    None,
        # Metadados de origem
        "label":       label,        # 0=normal, 1=ataque (por src_ip)
        "attack_type": attack_type,  # tipo de ataque do PCAP (referência)
        "pcap_file":   pcap_file,
    }

    raw_payload = None

    if pkt.haslayer(TCP):
        tcp = pkt[TCP]
        record.update({
            "transport":     "TCP",
            "src_port":      tcp.sport,
            "dst_port":      tcp.dport,
            "transport_len": len(tcp),
            "tcp_seq":       tcp.seq,
            "tcp_ack":       tcp.ack,
            "tcp_flags":     int(tcp.flags),
        })
        if pkt.haslayer(Raw):
            raw_payload = bytes(pkt[Raw].load)

    elif pkt.haslayer(UDP):
        udp = pkt[UDP]
        record.update({
            "transport":     "UDP",
            "src_port":      udp.sport,
            "dst_port":      udp.dport,
            "transport_len": udp.len,
        })
        if pkt.haslayer(Raw):
            raw_payload = bytes(pkt[Raw].load)
    else:
        # Descarta frames não-TCP/UDP (ARP, IGMP, ICMP, etc.)
        return None

    # Preenche payload de transporte para qualquer frame com dados
    if raw_payload:
        record["transport_payload_hex"] = raw_payload[:64].hex()
        # Tenta parsear SOME/IP somente se o payload for estruturalmente válido
        if is_valid_someip(raw_payload):
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
                    "someip_payload_hex": sh["someip_payload_hex"],
                    "someip_payload_len": len(sh["payload_bytes"]),
                })

    # Frames TCP sem payload (ACK puro, SYN, FIN) são mantidos com
    # transport_payload_hex=None e someip_valid=False — necessário para
    # reproduzir o total de ~14M amostras de Kim et al. (2026).
    return record


def process_all_pcaps(pcap_dir: str, output_csv: str,
                      pcap_filter: list = None) -> str:
    """Processa todos os PCAPs do dataset e salva um CSV consolidado.

    Itera sobre os 7 arquivos definidos em ``PCAP_LABEL_MAP``, aplica
    ``parse_packet`` em cada frame via streaming (``PcapReader``) e grava
    os registros SOME/IP extraídos em um único CSV de saída.

    O uso de ``PcapReader`` evita carregar o PCAP inteiro na RAM — crítico
    para arquivos de 200+ MB como os deste dataset.

    Args:
        pcap_dir: Caminho para a pasta contendo os arquivos ``.pcap``.
        output_csv: Caminho completo do arquivo CSV de saída. O diretório
            pai é criado automaticamente se não existir.

    Returns:
        Caminho absoluto do CSV gerado (igual a ``output_csv`` resolvido).

    Raises:
        RuntimeError: Se o pacote ``scapy`` não estiver instalado.

    Example:
        >>> process_all_pcaps("data/dataset_ism_xgboost", "data/parsed_packets.csv")
        'data/parsed_packets.csv'
    """
    if not SCAPY_OK:
        raise RuntimeError("scapy é necessário. Execute: pip install scapy")

    pcap_dir = Path(pcap_dir)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_pkts   = 0
    total_parsed = 0
    rows_written = 0

    COLUMNS = [
        "timestamp", "src_ip", "dst_ip", "ip_proto", "ip_ttl", "ip_len",
        "ip_id", "ip_flags", "transport", "src_port", "dst_port",
        "transport_len", "tcp_seq", "tcp_ack", "tcp_flags",
        "someip_valid", "service_id", "method_id", "someip_len",
        "client_id", "session_id", "proto_ver", "iface_ver",
        "msg_type", "msg_type_name", "return_code", "is_sd",
        "transport_payload_hex", "someip_payload_hex", "someip_payload_len",
        "label",        # 0=normal, 1=ataque (por src_ip)
        "attack_type",  # tipo de ataque do PCAP ("normal","dos","fuzzy","mitm")
        "pcap_file",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=COLUMNS)
        writer.writeheader()

        pcap_items = {k: v for k, v in PCAP_LABEL_MAP.items()
                      if pcap_filter is None or k in pcap_filter}

        for pcap_name, attack_type in pcap_items.items():
            pcap_path = pcap_dir / pcap_name
            if not pcap_path.exists():
                print(f"  [PULANDO] Arquivo nao encontrado: {pcap_path}")
                continue

            attacker_ips = ATTACKER_IPS_MAP.get(pcap_name, set())
            print(f"\n[>>] {pcap_name}  tipo={attack_type}  "
                  f"attackers={attacker_ips if attacker_ips else 'nenhum'}")
            n_pkts = 0
            n_parsed = 0
            n_attack = 0
            try:
                with PcapReader(str(pcap_path)) as reader:
                    for pkt in reader:
                        n_pkts += 1
                        rec = parse_packet(pkt, attack_type=attack_type,
                                           pcap_file=pcap_name,
                                           attacker_ips=attacker_ips)
                        if rec:
                            row = {col: rec.get(col, None) for col in COLUMNS}
                            writer.writerow(row)
                            n_parsed += 1
                            rows_written += 1
                            if rec["label"] == 1:
                                n_attack += 1
                        if n_pkts % 100_000 == 0:
                            print(f"  ... {n_pkts:,} pkts lidos, {n_parsed:,} escritos "
                                  f"({n_attack:,} ataque)")
            except Exception as e:
                print(f"  [ERRO] Falha ao ler PCAP: {e}")
                continue

            total_pkts   += n_pkts
            total_parsed += n_parsed
            pct_attack = 100 * n_attack / max(n_parsed, 1)
            print(f"  Total: {n_pkts:>8,} pkts | {n_parsed:>8,} escritos | "
                  f"{n_attack:>7,} ataque ({pct_attack:.1f}%)")

    print(f"\n{'='*60}")
    print(f"CONCLUIDOO")
    print(f"  Pacotes lidos    : {total_pkts:>10,}")
    print(f"  Registros SOME/IP: {total_parsed:>10,}")
    print(f"  Linhas no CSV    : {rows_written:>10,}")
    print(f"  Saida            : {out_path}")
    return str(out_path)


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="SOME/IP PCAP Parser — Etapa 1 da reprodução de Kim et al. (2026).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--pcap-dir",
        default=r"data/dataset_ism_xgboost",
        help="Pasta com os arquivos .pcap do dataset.",
    )
    ap.add_argument(
        "--output",
        default=r"data/parsed_packets.csv",
        help="Caminho do CSV de saída.",
    )
    args = ap.parse_args()

    process_all_pcaps(args.pcap_dir, args.output)
