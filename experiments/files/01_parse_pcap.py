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

SOMEIP_PORT_MIN = 30490
SOMEIP_PORT_MAX = 30510
"""int: Intervalo de portas SOME/IP usado pelo vSomeIP neste dataset.
    30490 = SOME/IP-SD multicast (padrão AUTOSAR).
    30501-30503 = serviços GPS, IMU e VDE da simulação.
"""

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
"""dict: Mapeamento nome-do-arquivo -> rótulo de classe para os 7 PCAPs do dataset."""


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
            "payload_bytes": someip_payload,
            "payload_hex":   someip_payload[:32].hex(),
        }
    except struct.error:
        return None


def is_someip_port(sport: int, dport: int) -> bool:
    """Verifica se o par de portas indica tráfego SOME/IP.

    O intervalo 30490-30510 cobre a porta SOME/IP-SD padrão (30490) e as
    portas de serviço usadas pelo vSomeIP neste dataset (30501-30503).
    O intervalo estendido até 30510 garante tolerância a variações de
    configuração sem capturar tráfego não-SOME/IP.

    Args:
        sport: Porta de origem do pacote TCP/UDP.
        dport: Porta de destino do pacote TCP/UDP.

    Returns:
        ``True`` se ao menos uma das portas estiver no intervalo SOME/IP.
    """
    return (SOMEIP_PORT_MIN <= sport <= SOMEIP_PORT_MAX) or \
           (SOMEIP_PORT_MIN <= dport <= SOMEIP_PORT_MAX)


def parse_packet(pkt, label: str, pcap_file: str) -> dict | None:
    """Extrai e estrutura os campos de todas as camadas de um pacote Scapy.

    Percorre as camadas IP > TCP/UDP > SOME/IP e monta um registro tabular
    com todos os campos relevantes para a Etapa 2 (extração de features).
    Pacotes sem camada IP, sem camada de transporte ou fora das portas
    SOME/IP são descartados (retornam ``None``).

    Args:
        pkt: Pacote Scapy lido pelo ``PcapReader``.
        label: Rótulo de classe do PCAP de origem (ex: ``"normal"``, ``"dos"``).
        pcap_file: Nome do arquivo PCAP de origem, usado para rastreabilidade.

    Returns:
        Dicionário com os campos de todas as camadas prontos para gravação
        no CSV, ou ``None`` se o pacote não for SOME/IP válido.
    """
    if not pkt.haslayer(IP):
        return None

    ip = pkt[IP]
    ts = float(pkt.time)

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
        # Cabeçalho SOME/IP (preenchido abaixo)
        "someip_valid":      False,
        "service_id":        None,
        "method_id":         None,
        "someip_len":        None,
        "client_id":         None,
        "session_id":        None,
        "proto_ver":         None,
        "iface_ver":         None,
        "msg_type":          None,
        "msg_type_name":     None,
        "return_code":       None,
        "is_sd":             None,
        "payload_hex":       None,
        "someip_payload_len": None,
        # Metadados de origem
        "label":     label,
        "pcap_file": pcap_file,
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
        return None

    if not is_someip_port(record["src_port"], record["dst_port"]):
        return None

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
        # Pacotes na porta SOME/IP sem header válido ainda são mantidos
        # (ex: pacotes SD fragmentados ou com payload vazio)

    return record


def process_all_pcaps(pcap_dir: str, output_csv: str) -> str:
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
