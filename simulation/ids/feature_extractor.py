"""
Extração stateful de features SOME/IP — mantém estado entre pacotes.

Port de multiclass/01_features.py para modo incremental (um pacote por vez).
"""
import numpy as np
from collections import defaultdict, deque
from parser_inline import SomeIPPacket

RELAY_SVC = 0x100B

FEAT_COLS = [
    'f01_ip_time_interval',
    'f08_someip_payload_change',
    'f11_ip_length_change',
    'f12_tcpudp_length_change',
    'f13_payload_repeat_rate',
    'f15_someip_payload_len',
    'f16_tcpudp_len',
    'f17_src_packet_rate',
    'f18_src_payload_diversity',
    'f19_is_sd',
    'f20_src_service_diversity',
    'f21_is_relay_service',
    'f22_src_clientid_diversity',
]


def _hamming(hex_a: str, hex_b: str) -> float:
    if not hex_a or not hex_b:
        return 0.0
    try:
        a = bytes.fromhex(hex_a)
        b = bytes.fromhex(hex_b)
    except ValueError:
        return 0.0
    L = min(len(a), len(b))
    if L == 0:
        return 0.0
    return float(np.unpackbits(
        np.bitwise_xor(np.frombuffer(a[:L], np.uint8),
                       np.frombuffer(b[:L], np.uint8))
    ).sum()) / (8 * L)


class FeatureExtractor:
    """Extrai 13 features SOME/IP com estado persistente entre pacotes."""

    def __init__(self):
        self._prev_ts         = defaultdict(lambda: None)
        self._prev_ip_len     = defaultdict(lambda: None)
        self._prev_tl_len     = defaultdict(lambda: None)
        self._prev_si_pld     = defaultdict(lambda: None)
        self._recent_payloads = defaultdict(lambda: deque(maxlen=5))
        self._src_timestamps  = defaultdict(lambda: deque(maxlen=1000))
        self._src_payloads    = defaultdict(lambda: deque(maxlen=1000))
        self._src_services    = defaultdict(lambda: deque(maxlen=100))
        self._src_clientids   = defaultdict(lambda: deque(maxlen=100))

    def _src_rate(self, src: str, ts: float) -> float:
        win = self._src_timestamps[src]
        win.append(ts)
        if len(win) < 2:
            return 0.0
        delta = ts - win[0]
        return (len(win) - 1) / delta if delta > 0 else float(len(win))

    def _deque_uniq(self, win: deque, value) -> float:
        if value is not None:
            win.append(value)
        return float(len(set(win))) if win else 1.0

    def extract(self, pkt: SomeIPPacket) -> np.ndarray:
        """Retorna vetor float32 com 13 features para um único pacote."""
        key  = (pkt.src_ip, pkt.dst_ip, pkt.src_port, pkt.dst_port, pkt.transport)
        ts   = pkt.timestamp
        src  = pkt.src_ip
        si_h = pkt.someip_payload_hex or None
        svc  = pkt.service_id

        prev_ts = self._prev_ts[key]
        f01 = abs(ts - prev_ts) if prev_ts is not None else 0.0

        f08 = _hamming(self._prev_si_pld[key], si_h)

        p_ip = self._prev_ip_len[key]
        p_tl = self._prev_tl_len[key]
        f11  = abs(pkt.ip_len  - p_ip) if p_ip is not None else 0.0
        f12  = abs(pkt.transport_len - p_tl) if p_tl is not None else 0.0

        hist = self._recent_payloads[key]
        f13  = sum(1 for p in hist if p == si_h) / len(hist) if (hist and si_h) else 0.0
        hist.append(si_h)

        f15 = float(pkt.someip_payload_len)
        f16 = float(pkt.transport_len)
        f17 = self._src_rate(src, ts)

        win_p = self._src_payloads[src]
        if si_h:
            win_p.append(si_h)
        f18 = len(set(win_p)) / len(win_p) if len(win_p) > 1 else 0.0

        f19 = 1.0 if pkt.is_sd else 0.0
        f20 = self._deque_uniq(self._src_services[src], svc)
        f21 = 1.0 if (svc == RELAY_SVC) else 0.0
        f22 = self._deque_uniq(self._src_clientids[src], pkt.client_id)

        self._prev_ts[key]     = ts
        self._prev_ip_len[key] = pkt.ip_len
        self._prev_tl_len[key] = pkt.transport_len
        self._prev_si_pld[key] = si_h

        return np.array([f01, f08, f11, f12, f13, f15, f16, f17, f18, f19, f20, f21, f22],
                        dtype=np.float32)
