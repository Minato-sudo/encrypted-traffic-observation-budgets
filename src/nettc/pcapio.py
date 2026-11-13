"""Streaming readers for classic pcap and pcapng capture files.

Written against the file formats directly rather than via scapy/dpkt: the
sweep in Phase 2 needs every byte of every layer, including the link layer,
and it needs to stream multi-gigabyte captures twice without holding them in
memory. Both readers yield ``(timestamp, linktype, frame_bytes)``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple

# Classic pcap magics, keyed on the bytes as they appear on disk, which is
# what actually identifies the writer's byte order. The 0xa1b23c4d variant
# carries nanosecond resolution in the sub-second field instead of microsecond.
#   on-disk bytes -> (struct endian prefix, sub-second unit in seconds)
_PCAP_MAGICS: dict[bytes, tuple[str, float]] = {
    b"\xd4\xc3\xb2\xa1": ("<", 1e-6),
    b"\xa1\xb2\xc3\xd4": (">", 1e-6),
    b"\x4d\x3c\xb2\xa1": ("<", 1e-9),
    b"\xa1\xb2\x3c\x4d": (">", 1e-9),
}

_PCAPNG_MAGIC = b"\x0a\x0d\x0d\x0a"

Packet = Tuple[float, int, bytes]


class CaptureFormatError(ValueError):
    """Raised when a file is not a capture we can read."""


@dataclass(frozen=True)
class CaptureInfo:
    path: Path
    format: str          # "pcap" | "pcapng"
    linktypes: tuple[int, ...]


def _read_exactly(fh, n: int) -> bytes:
    """Read n bytes, tolerating short reads from the OS."""
    chunks = []
    remaining = n
    while remaining:
        chunk = fh.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _iter_pcap(fh, endian: str, frac: float) -> Iterator[Packet]:
    """Stream a classic pcap whose 4-byte magic has already been consumed."""
    rest = _read_exactly(fh, 20)
    if len(rest) != 20:
        raise CaptureFormatError("truncated pcap global header")
    _vmaj, _vmin, _tz, _sig, _snaplen, linktype = struct.unpack(endian + "HHiIII", rest)

    rec_hdr = struct.Struct(endian + "IIII")
    while True:
        hdr = _read_exactly(fh, 16)
        if len(hdr) < 16:
            return
        ts_sec, ts_sub, caplen, _origlen = rec_hdr.unpack(hdr)
        if caplen == 0:
            continue
        data = _read_exactly(fh, caplen)
        if len(data) < caplen:
            return  # truncated final record
        yield (ts_sec + ts_sub * frac, linktype, data)


def _iter_pcapng(fh) -> Iterator[Packet]:
    # Section Header Block determines byte order via its magic field.
    first = _read_exactly(fh, 12)
    if len(first) < 12:
        raise CaptureFormatError("truncated pcapng section header")
    byte_order_magic = first[8:12]
    if byte_order_magic == b"\x1a\x2b\x3c\x4d":
        endian = "<"
    elif byte_order_magic == b"\x4d\x3c\x2b\x1a":
        endian = ">"
    else:
        raise CaptureFormatError("unrecognised pcapng byte-order magic")

    total_len = struct.unpack(endian + "I", first[4:8])[0]
    _read_exactly(fh, total_len - 12)  # skip rest of SHB

    # Interface IDs index into this list; each carries its own linktype and
    # timestamp resolution.
    interfaces: list[tuple[int, int]] = []  # (linktype, ts_divisor)
    blk_hdr = struct.Struct(endian + "II")

    while True:
        hdr = _read_exactly(fh, 8)
        if len(hdr) < 8:
            return
        btype, blen = blk_hdr.unpack(hdr)
        if blen < 12:
            raise CaptureFormatError(f"implausible pcapng block length {blen}")
        body = _read_exactly(fh, blen - 12)
        if len(body) < blen - 12:
            return
        _read_exactly(fh, 4)  # trailing total length

        if btype == 0x00000001:  # Interface Description Block
            linktype = struct.unpack(endian + "H", body[0:2])[0]
            interfaces.append((linktype, _ng_ts_divisor(body[8:], endian)))
        elif btype == 0x00000006:  # Enhanced Packet Block
            iface_id, ts_hi, ts_lo, caplen, _origlen = struct.unpack(endian + "IIIII", body[:20])
            data = body[20:20 + caplen]
            linktype, divisor = interfaces[iface_id] if iface_id < len(interfaces) else (1, 1_000_000)
            ts = ((ts_hi << 32) | ts_lo) / divisor
            yield (ts, linktype, data)
        elif btype == 0x00000003:  # Simple Packet Block
            _origlen = struct.unpack(endian + "I", body[:4])[0]
            data = body[4:]
            linktype, _ = interfaces[0] if interfaces else (1, 1_000_000)
            yield (0.0, linktype, data)
        elif btype == 0x0A0D0D0A:  # a further Section Header Block
            interfaces.clear()


def _ng_ts_divisor(options: bytes, endian: str) -> int:
    """Read if_tsresol (option code 9) from an IDB option list."""
    pos = 0
    while pos + 4 <= len(options):
        code, length = struct.unpack(endian + "HH", options[pos:pos + 4])
        pos += 4
        if code == 0:  # opt_endofopt
            break
        value = options[pos:pos + length]
        pos += length + ((4 - length % 4) % 4)
        if code == 9 and value:
            raw = value[0]
            # High bit set means the resolution is a power of two.
            return (1 << (raw & 0x7F)) if (raw & 0x80) else (10 ** (raw & 0x7F))
    return 1_000_000


def iter_packets(path: str | Path) -> Iterator[Packet]:
    """Stream ``(timestamp, linktype, frame_bytes)`` from a capture file."""
    path = Path(path)
    with open(path, "rb") as fh:
        head = fh.read(4)
        if len(head) < 4:
            raise CaptureFormatError(f"{path}: file shorter than a magic number")
        if head == _PCAPNG_MAGIC:
            fh.seek(0)
            yield from _iter_pcapng(fh)
        elif head in _PCAP_MAGICS:
            endian, frac = _PCAP_MAGICS[head]
            yield from _iter_pcap(fh, endian, frac)
        else:
            raise CaptureFormatError(f"{path}: unknown capture magic {head.hex()}")


def probe(path: str | Path) -> CaptureInfo:
    """Identify a capture's format and the link types it uses."""
    path = Path(path)
    with open(path, "rb") as fh:
        head = fh.read(4)
    if head == _PCAPNG_MAGIC:
        fmt = "pcapng"
    elif head in _PCAP_MAGICS:
        fmt = "pcap"
    else:
        raise CaptureFormatError(f"{path}: unknown capture magic {head.hex()}")

    seen: set[int] = set()
    for i, (_ts, linktype, _data) in enumerate(iter_packets(path)):
        seen.add(linktype)
        if i >= 64:
            break
    return CaptureInfo(path=path, format=fmt, linktypes=tuple(sorted(seen)))
