"""
Parser SOME/IP inline — extrai campos de um pacote scapy sem passar por CSV.
"""
import struct
from dataclasses import dataclass
from typing import Optional

SOMEIP_PORTS    = {30490, 30491, 30492, 30501, 30502, 30503}
SOMEIP_SD_SVC   = 0xFFFF
SOMEIP_HDR_LEN  = 16


@dataclass
class SomeIPPacket:
    timestamp:          float
    src_ip:             str
    dst_ip:             str
    src_port:           int
    dst_port:           int
    transport:          str
    ip_len:             int
    transport_len:      int
    service_id:         Optional[int]
    method_id:          Optional[int]
    someip_payload_hex: str
    someip_payload_len: int
    is_sd:              bool
    client_id:          Optional[int]


def parse_packet(pkt) -> Optional[SomeIPPacket]:
    """Recebe pacote scapy, retorna SomeIPPacket ou None se não for SOME/IP."""
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.packet import Raw

    if not pkt.haslayer(IP):
        return None

    ip     = pkt[IP]
    ip_len = ip.len

    if pkt.haslayer(UDP):
        l4            = pkt[UDP]
        transport     = 'UDP'
        transport_len = max(0, l4.len - 8)
    elif pkt.haslayer(TCP):
        l4            = pkt[TCP]
        transport     = 'TCP'
        transport_len = len(bytes(l4.payload))
    else:
        return None

    sport = l4.sport
    dport = l4.dport

    if sport not in SOMEIP_PORTS and dport not in SOMEIP_PORTS:
        return None

    raw = bytes(l4.payload) if pkt.haslayer(Raw) else b''
    if len(raw) < SOMEIP_HDR_LEN:
        return None

    try:
        service_id, method_id, _ = struct.unpack_from('>HHI', raw, 0)
        client_id, _session_id   = struct.unpack_from('>HH', raw, 8)
        payload     = raw[SOMEIP_HDR_LEN:]
        payload_hex = payload.hex() if payload else ''
        payload_len = len(payload)
    except struct.error:
        return None

    ts = float(pkt.time) if hasattr(pkt, 'time') else 0.0

    return SomeIPPacket(
        timestamp          = ts,
        src_ip             = ip.src,
        dst_ip             = ip.dst,
        src_port           = sport,
        dst_port           = dport,
        transport          = transport,
        ip_len             = ip_len,
        transport_len      = transport_len,
        service_id         = service_id,
        method_id          = method_id,
        someip_payload_hex = payload_hex,
        someip_payload_len = payload_len,
        is_sd              = (service_id == SOMEIP_SD_SVC),
        client_id          = client_id,
    )
