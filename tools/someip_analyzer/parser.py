"""
Pure-Python PCAP parser for SOME/IP traffic.
No scapy — uses struct for speed on 200 MB+ files.

Detection strategy: no port filter.
Every TCP/UDP payload is tested against the SOME/IP header structure:
  - proto_ver  must be 0x01
  - msg_type   must be a known value
  - length     must be consistent with actual payload size (>= 8, <= available bytes)
  - service_id 0x0000 and method_id 0x0000 rejected (null header = not SOME/IP)
"""

import struct
from dataclasses import dataclass
from typing import List, Optional

MAX_PACKETS = 20_000

VALID_MSG_TYPES = {
    0x00, 0x01, 0x02,          # REQUEST, REQUEST_NO_RETURN, NOTIFICATION
    0x40, 0x41, 0x42,          # *_ACK variants
    0x80, 0x81,                # RESPONSE, ERROR
    0xC0, 0xC1,                # RESPONSE_ACK, ERROR_ACK
}

MSG_TYPE_NAMES = {
    0x00: "REQUEST",
    0x01: "REQUEST_NO_RETURN",
    0x02: "NOTIFICATION",
    0x40: "REQUEST_ACK",
    0x41: "REQUEST_NO_RETURN_ACK",
    0x42: "NOTIFICATION_ACK",
    0x80: "RESPONSE",
    0x81: "ERROR",
    0xC0: "RESPONSE_ACK",
    0xC1: "ERROR_ACK",
}

RETURN_CODE_NAMES = {
    0x00: "E_OK",
    0x01: "E_NOT_OK",
    0x02: "E_UNKNOWN_SERVICE",
    0x03: "E_UNKNOWN_METHOD",
    0x04: "E_NOT_READY",
    0x05: "E_NOT_REACHABLE",
    0x06: "E_TIMEOUT",
    0x07: "E_WRONG_PROTOCOL_VERSION",
    0x08: "E_WRONG_INTERFACE_VERSION",
    0x09: "E_MALFORMED_MESSAGE",
    0x0A: "E_WRONG_MESSAGE_TYPE",
}


@dataclass
class SomeIpPacket:
    index: int
    rel_time: float
    src_ip: str
    dst_ip: str
    sport: int
    dport: int
    proto: str
    service_id: int
    method_id: int
    length: int
    client_id: int
    session_id: int
    proto_ver: int
    iface_ver: int
    msg_type: int
    return_code: int
    payload_hex: str
    payload_len: int
    msg_type_name: str
    return_code_name: str


def _u16(buf, off): return struct.unpack_from(">H", buf, off)[0]
def _u32(buf, off): return struct.unpack_from(">I", buf, off)[0]


def _parse_ip(buf, off):
    return ".".join(str(b) for b in buf[off:off+4])


def _try_someip(payload: bytes) -> Optional[dict]:
    """
    Attempt to interpret payload as a SOME/IP message.
    Returns parsed fields if the header is structurally valid, None otherwise.

    Validation rules (from SOME/IP spec, confirmed on observed packets):
      1. Minimum 16 bytes (fixed header size)
      2. proto_ver == 0x01
      3. msg_type in known set
      4. length field >= 8 (covers at least the 8 post-length bytes)
      5. length + 8 <= total payload available
      6. service_id and method_id both zero is rejected (null/padding)
    """
    if len(payload) < 16:
        return None

    service_id  = _u16(payload, 0)
    method_id   = _u16(payload, 2)
    length      = _u32(payload, 4)
    client_id   = _u16(payload, 8)
    session_id  = _u16(payload, 10)
    proto_ver   = payload[12]
    iface_ver   = payload[13]
    msg_type    = payload[14]
    return_code = payload[15]

    if proto_ver != 0x01:
        return None
    if msg_type not in VALID_MSG_TYPES:
        return None
    if length < 8:
        return None
    if length + 8 > len(payload) + 8:
        return None
    if service_id == 0x0000 and method_id == 0x0000:
        return None

    app_payload = payload[16:16 + (length - 8)]
    return dict(
        service_id=service_id,
        method_id=method_id,
        length=length,
        client_id=client_id,
        session_id=session_id,
        proto_ver=proto_ver,
        iface_ver=iface_ver,
        msg_type=msg_type,
        return_code=return_code,
        payload_hex=app_payload[:64].hex().upper(),
        payload_len=len(app_payload),
        msg_type_name=MSG_TYPE_NAMES.get(msg_type, f"0x{msg_type:02X}"),
        return_code_name=RETURN_CODE_NAMES.get(return_code, f"0x{return_code:02X}"),
    )


def parse_pcap(path: str, max_packets: int = MAX_PACKETS) -> List[SomeIpPacket]:
    packets: List[SomeIpPacket] = []
    pkt_index = 0

    with open(path, "rb") as f:
        hdr = f.read(24)
        if len(hdr) < 24:
            return packets

        magic, _vmaj, _vmin, _tz, _sf, _snaplen, link_type = struct.unpack("<IHHiIII", hdr)

        if magic == 0xA1B2C3D4:
            ts_factor = 1e-6
        elif magic == 0xA1B23C4D:
            ts_factor = 1e-9
        else:
            raise ValueError(f"Not a PCAP file (magic=0x{magic:08X})")

        first_ts = None

        while len(packets) < max_packets:
            rec_hdr = f.read(16)
            if len(rec_hdr) < 16:
                break
            ts_sec, ts_usec, incl_len, _orig = struct.unpack("<IIII", rec_hdr)
            raw = f.read(incl_len)
            if len(raw) < incl_len:
                break

            abs_ts = ts_sec + ts_usec * ts_factor
            if first_ts is None:
                first_ts = abs_ts
            rel_ts = abs_ts - first_ts

            if link_type != 1:  # Ethernet only
                continue
            if len(raw) < 14:
                continue

            eth_type = _u16(raw, 12)
            ip_start = 14
            if eth_type == 0x8100:  # 802.1Q VLAN
                if len(raw) < 18:
                    continue
                eth_type = _u16(raw, 16)
                ip_start = 18

            if eth_type != 0x0800:  # IPv4 only
                continue
            if len(raw) < ip_start + 20:
                continue

            ihl       = (raw[ip_start] & 0x0F) * 4
            ip_proto  = raw[ip_start + 9]
            src_ip    = _parse_ip(raw, ip_start + 12)
            dst_ip    = _parse_ip(raw, ip_start + 16)
            t_start   = ip_start + ihl

            if ip_proto == 6:  # TCP
                if len(raw) < t_start + 20:
                    continue
                sport      = _u16(raw, t_start)
                dport      = _u16(raw, t_start + 2)
                data_off   = ((raw[t_start + 12] >> 4) & 0xF) * 4
                app_start  = t_start + data_off
                proto_label = "TCP"
            elif ip_proto == 17:  # UDP
                if len(raw) < t_start + 8:
                    continue
                sport      = _u16(raw, t_start)
                dport      = _u16(raw, t_start + 2)
                app_start  = t_start + 8
                proto_label = "UDP"
            else:
                continue

            app_payload = raw[app_start:]
            si = _try_someip(app_payload)
            if si is None:
                continue

            pkt_index += 1
            packets.append(SomeIpPacket(
                index=pkt_index,
                rel_time=round(rel_ts, 6),
                src_ip=src_ip,
                dst_ip=dst_ip,
                sport=sport,
                dport=dport,
                proto=proto_label,
                **si,
            ))

    return packets


def packets_to_rows(packets: List[SomeIpPacket]) -> List[dict]:
    rows = []
    for p in packets:
        rows.append({
            "#":           p.index,
            "Time":        f"{p.rel_time:.6f}",
            "Source":      f"{p.src_ip}:{p.sport}",
            "Destination": f"{p.dst_ip}:{p.dport}",
            "Proto":       p.proto,
            "Service ID":  f"0x{p.service_id:04X}",
            "Method ID":   f"0x{p.method_id:04X}",
            "Msg Type":    p.msg_type_name,
            "Length":      p.length,
            "Pay Len":     p.payload_len,
            "_service_id":  p.service_id,
            "_method_id":   p.method_id,
            "_client_id":   p.client_id,
            "_session_id":  p.session_id,
            "_proto_ver":   p.proto_ver,
            "_iface_ver":   p.iface_ver,
            "_msg_type":    p.msg_type,
            "_return_code": p.return_code,
            "_rc_name":     p.return_code_name,
            "_payload_hex": p.payload_hex,
        })
    return rows
