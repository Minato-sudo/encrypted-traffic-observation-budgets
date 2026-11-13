"""Synthetic captures with known ground truth, for validating the pipeline.

The budget sweep is only meaningful if the machinery underneath it is
correct, and the corpora are too large and too opaque to debug against. So
the generator writes real pcap files whose answers we already know:

* each class carries a *weak early cue* in the first packet and a *strong
  late cue* in a chosen later packet, so a correct budget sweep must show
  accuracy rising in two steps at predictable budgets;
* flow lengths are drawn from a heavy short-flow distribution, so padding
  fraction varies the way it does on the real corpora (P8 reports ~87% of
  ISCX-VPN sessions below 1024 bytes);
* MAC and IP addresses are made perfectly predictive of the label, so the
  Step 11 leakage control has something to find. If anonymisation works the
  probe must sit near chance; if the probe scores well, the pipeline is
  broken and the gate should halt the study.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

_PCAP_GLOBAL = struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)  # linktype 1 = Ethernet


def _pcap_record(ts: float, frame: bytes) -> bytes:
    sec = int(ts)
    usec = int(round((ts - sec) * 1_000_000)) % 1_000_000
    return struct.pack("<IIII", sec, usec, len(frame), len(frame)) + frame


def _ipv4_checksum(header: bytes) -> int:
    if len(header) % 2:
        header += b"\x00"
    total = sum(struct.unpack(f">{len(header) // 2}H", header))
    while total >> 16:
        total = (total & 0xFFFF) + (total >> 16)
    return (~total) & 0xFFFF


def _frame(src_mac: bytes, dst_mac: bytes, src_ip: bytes, dst_ip: bytes,
           sport: int, dport: int, payload: bytes, seq: int, proto: int = 6) -> bytes:
    if proto == 6:
        l4 = struct.pack(">HHIIBBHHH", sport, dport, seq, 0, 0x50, 0x18, 8192, 0, 0)
    else:
        l4 = struct.pack(">HHHH", sport, dport, 8 + len(payload), 0)
    total_len = 20 + len(l4) + len(payload)
    ip_hdr = struct.pack(">BBHHHBBH", 0x45, 0, total_len, seq & 0xFFFF, 0, 64, proto, 0) + src_ip + dst_ip
    ip_hdr = ip_hdr[:10] + struct.pack(">H", _ipv4_checksum(ip_hdr)) + ip_hdr[12:]
    return dst_mac + src_mac + b"\x08\x00" + ip_hdr + l4 + payload


def _client_hello(cipher: int) -> bytes:
    body = (struct.pack(">H", 0x0303) + bytes(32) + b"\x00"
            + struct.pack(">H", 2) + struct.pack(">H", cipher)
            + b"\x01\x00" + struct.pack(">H", 0))
    hs = b"\x01" + len(body).to_bytes(3, "big") + body
    return b"\x16\x03\x01" + struct.pack(">H", len(hs)) + hs


def _server_hello(cipher: int, tls13: bool = False) -> bytes:
    exts = b""
    if tls13:
        exts = struct.pack(">HH", 43, 2) + struct.pack(">H", 0x0304)
    body = (struct.pack(">H", 0x0303) + bytes(32) + b"\x00"
            + struct.pack(">H", cipher) + b"\x00"
            + struct.pack(">H", len(exts)) + exts)
    hs = b"\x02" + len(body).to_bytes(3, "big") + body
    return b"\x16\x03\x03" + struct.pack(">H", len(hs)) + hs


def _tls_app_data(rng: np.random.Generator, n: int) -> bytes:
    return b"\x17\x03\x03" + struct.pack(">H", n) + rng.integers(0, 256, n, dtype=np.uint8).tobytes()


class SyntheticCorpus:
    """Writes one pcap per class, with recorded ground truth."""

    def __init__(self, n_classes: int = 8, sessions_per_class: int = 400,
                 weak_cue_offset: int = 24, strong_cue_packet: int = 4,
                 length_mu: float = 1.0, length_sigma: float = 0.9,
                 min_packets: int = 1, seed: int = 0):
        self.n_classes = n_classes
        self.sessions_per_class = sessions_per_class
        self.weak_cue_offset = weak_cue_offset
        self.strong_cue_packet = strong_cue_packet
        # Flow-length distribution. The defaults put most mass on 1-4 packet
        # flows, matching the short-flow dominance P8's statistics imply;
        # raise min_packets to study a population that all carries the late cue.
        self.length_mu = length_mu
        self.length_sigma = length_sigma
        self.min_packets = min_packets
        self.seed = seed
        # Half the classes are TLS-encrypted, half plaintext HTTP, so the
        # stratified reporting of Step 3b has both strata to separate.
        self.ciphers = [0xC02F, 0x002F, 0x1301, 0xC030]

    def class_names(self) -> list[str]:
        return [f"class{i:02d}" for i in range(self.n_classes)]

    def write(self, out_dir: str | Path) -> list[tuple[Path, str]]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        captures = []
        for cls in range(self.n_classes):
            path = out_dir / f"{self.class_names()[cls]}.pcap"
            self._write_class(path, cls)
            captures.append((path, path.stem))
        return captures

    def _write_class(self, path: Path, cls: int) -> None:
        rng = np.random.default_rng(self.seed * 1000 + cls)
        encrypted = cls % 2 == 0
        # Identifiers that give the label away, for the leakage probe.
        src_mac = bytes([0x02, 0x00, 0x00, 0x00, cls, 0x01])
        dst_mac = bytes([0x02, 0x00, 0x00, 0x00, cls, 0x02])
        src_ip = bytes([10, cls + 1, 0, 1])

        # Group cue: coarse family, visible in the first packet.
        group = cls % 3
        weak_sig = bytes([0xA0 + group]) * 4
        strong_sig = bytes([0xE0, cls, 0xE1, cls, 0xE2, cls, 0xE3, cls])

        with open(path, "wb") as fh:
            fh.write(_PCAP_GLOBAL)
            ts = 1_600_000_000.0
            for s in range(self.sessions_per_class):
                dst_ip = bytes([192, 168, (s >> 8) & 0xFF, s & 0xFF])
                sport = 40000 + (s % 20000)
                dport = 443 if encrypted else 80
                # Heavy short-flow distribution: a large mass of 1-4 packet
                # flows alongside a long tail.
                n_pkts = int(np.clip(rng.lognormal(self.length_mu, self.length_sigma),
                                     self.min_packets, 40))
                # Vary the suite across class pairs so every entry in the
                # table appears, including the deprecated CBC one.
                cipher = self.ciphers[(cls // 2) % len(self.ciphers)]

                for p in range(n_pkts):
                    body = bytearray(rng.integers(0, 256, int(np.clip(rng.normal(220, 90), 8, 1200)), dtype=np.uint8).tobytes())
                    if p == 0:
                        if encrypted:
                            body = bytearray(_client_hello(cipher))
                        else:
                            body = bytearray(b"GET /index.html HTTP/1.1\r\nHost: example.invalid\r\n\r\n")
                        off = min(self.weak_cue_offset, max(0, len(body) - len(weak_sig)))
                        body[off:off + len(weak_sig)] = weak_sig
                    elif p == 1 and encrypted:
                        body = bytearray(_server_hello(cipher, tls13=(cipher == 0x1301)))
                    elif encrypted:
                        body = bytearray(_tls_app_data(rng, int(np.clip(rng.normal(200, 80), 16, 1100))))
                    if p == self.strong_cue_packet and len(body) > len(strong_sig) + 8:
                        body[8:8 + len(strong_sig)] = strong_sig

                    forward = (p % 2 == 0)
                    frame = (_frame(src_mac, dst_mac, src_ip, dst_ip, sport, dport, bytes(body), p + 1)
                             if forward else
                             _frame(dst_mac, src_mac, dst_ip, src_ip, dport, sport, bytes(body), p + 1))
                    fh.write(_pcap_record(ts, frame))
                    ts += 0.001 + float(rng.random()) * 0.01
                ts += 0.05
