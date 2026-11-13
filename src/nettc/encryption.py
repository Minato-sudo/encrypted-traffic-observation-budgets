"""Assign each session an encryption stratum and record its TLS cipher suite.

Phase 2 Step 3b, following P5 (SoK, IEEE S&P 2025), which measured
ISCXVPN2016 at 98.9% unencrypted and USTC-TFC2016 at 94.7% unencrypted. If
that holds on our own preprocessing then every downstream number has to be
split into encrypted and plaintext strata rather than reported as one figure,
so the tagging runs before training and is stored alongside the bytes.

The classification is deliberately conservative: a session is only called
encrypted_tls when a real handshake is parsed out of it, and anything we
cannot positively identify stays ``unknown`` rather than being guessed into
one of the two strata we report on.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

from .dissect import (
    STRATUM_BINARY,
    STRATUM_ENCRYPTED_OTHER,
    STRATUM_ENCRYPTED_TLS,
    STRATUM_OPAQUE,
    STRATUM_PLAINTEXT,
    STRATUM_UNKNOWN,
)

# Byte entropy at or above this, over a payload long enough to measure it, is
# indistinguishable from random. That is necessary for encryption but not
# sufficient -- compressed payload looks the same -- so it yields
# STRATUM_OPAQUE, never STRATUM_ENCRYPTED_*.
ENTROPY_OPAQUE_BITS = 7.3
ENTROPY_MIN_BYTES = 256

# How much L4 payload per direction we buffer for handshake inspection. A
# ClientHello with a long extension list can exceed one segment, so this is
# larger than a single MSS.
SCAN_BYTES = 4096

TLS_VERSIONS = {
    0x0300: "SSL3.0", 0x0301: "TLS1.0", 0x0302: "TLS1.1",
    0x0303: "TLS1.2", 0x0304: "TLS1.3",
}

# Cipher suites appearing in the benchmark corpora, split the way P5 splits
# them: AEAD constructions against the deprecated CBC/3DES/RC4 families.
_DEPRECATED_MARKERS = ("_CBC_", "_RC4_", "_3DES_", "_DES_", "_NULL_", "_EXPORT", "_MD5")

CIPHER_NAMES = {
    0x0004: "TLS_RSA_WITH_RC4_128_MD5",
    0x0005: "TLS_RSA_WITH_RC4_128_SHA",
    0x000A: "TLS_RSA_WITH_3DES_EDE_CBC_SHA",
    0x002F: "TLS_RSA_WITH_AES_128_CBC_SHA",
    0x0035: "TLS_RSA_WITH_AES_256_CBC_SHA",
    0x003C: "TLS_RSA_WITH_AES_128_CBC_SHA256",
    0x003D: "TLS_RSA_WITH_AES_256_CBC_SHA256",
    0x009C: "TLS_RSA_WITH_AES_128_GCM_SHA256",
    0x009D: "TLS_RSA_WITH_AES_256_GCM_SHA384",
    0x1301: "TLS_AES_128_GCM_SHA256",
    0x1302: "TLS_AES_256_GCM_SHA384",
    0x1303: "TLS_CHACHA20_POLY1305_SHA256",
    0xC009: "TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA",
    0xC00A: "TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA",
    0xC013: "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA",
    0xC014: "TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA",
    0xC027: "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256",
    0xC028: "TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA384",
    0xC02B: "TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256",
    0xC02C: "TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384",
    0xC02F: "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256",
    0xC030: "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384",
    0xCCA8: "TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256",
    0xCCA9: "TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256",
}

_PLAINTEXT_PREFIXES = (
    b"GET ", b"POST ", b"HEAD ", b"PUT ", b"DELETE ", b"OPTIONS ", b"PATCH ",
    b"CONNECT ", b"TRACE ", b"HTTP/1.", b"220 ", b"USER ", b"EHLO", b"HELO",
    b"+OK", b"* OK", b"RTSP/", b"SIP/2.0", b"REGISTER sip:", b"INVITE sip:",
)


def cipher_is_deprecated(suite: int) -> bool:
    name = CIPHER_NAMES.get(suite)
    if name is None:
        return False
    return any(marker in name for marker in _DEPRECATED_MARKERS)


@dataclass(slots=True)
class EncryptionProbe:
    """Accumulates a bounded prefix of each direction and classifies at close."""
    fwd: bytearray = field(default_factory=bytearray)
    rev: bytearray = field(default_factory=bytearray)
    proto: int = 6
    sport: int = 0
    dport: int = 0

    def feed(self, payload: bytes, is_forward: bool) -> None:
        if not payload:
            return
        buf = self.fwd if is_forward else self.rev
        if len(buf) < SCAN_BYTES:
            buf.extend(payload[: SCAN_BYTES - len(buf)])

    def payload_entropy(self) -> float:
        """Shannon entropy in bits per byte over both directions' payload."""
        data = bytes(self.fwd) + bytes(self.rev)
        return shannon_entropy(data)

    def classify(self) -> tuple[int, int, int]:
        """Return (stratum, cipher_suite, tls_version); 0 where not applicable."""
        for buf in (self.rev, self.fwd):  # ServerHello is usually the reverse side
            suite, version = _parse_tls_handshake(bytes(buf))
            if suite is not None:
                return STRATUM_ENCRYPTED_TLS, suite, version
        # A handshake we could not parse, but TLS record framing is present.
        for buf in (self.fwd, self.rev):
            version = _tls_record_version(bytes(buf))
            if version:
                return STRATUM_ENCRYPTED_TLS, 0, version

        if self.proto == 17 and _looks_like_quic(bytes(self.fwd)):
            return STRATUM_ENCRYPTED_OTHER, 0, 0
        for buf in (self.fwd, self.rev):
            if bytes(buf[:4]) == b"SSH-":
                return STRATUM_ENCRYPTED_OTHER, 0, 0

        combined = bytes(self.fwd[:512]) + bytes(self.rev[:512])
        if not combined:
            return STRATUM_UNKNOWN, 0, 0
        for buf in (self.fwd, self.rev):
            if any(bytes(buf).startswith(p) for p in _PLAINTEXT_PREFIXES):
                return STRATUM_PLAINTEXT, 0, 0
        if _printable_ratio(combined) >= 0.85:
            return STRATUM_PLAINTEXT, 0, 0

        # Everything left is binary. Separating it by entropy is what lets the
        # Gap 3 figure be stated at all: without this, protocols like SMB,
        # MySQL and game traffic fall into an "unknown" bucket that is most of
        # the corpus and says nothing about whether it is encrypted.
        full = bytes(self.fwd) + bytes(self.rev)
        if len(full) >= ENTROPY_MIN_BYTES:
            return ((STRATUM_OPAQUE if shannon_entropy(full) >= ENTROPY_OPAQUE_BITS
                     else STRATUM_BINARY), 0, 0)
        return STRATUM_UNKNOWN, 0, 0


def shannon_entropy(data: bytes) -> float:
    """Bits per byte. 8.0 is uniform random; English text sits near 4."""
    if not data:
        return 0.0
    counts = [0] * 256
    for b in data:
        counts[b] += 1
    n = len(data)
    total = 0.0
    for c in counts:
        if c:
            p = c / n
            total -= p * math.log2(p)
    return total


def _printable_ratio(data: bytes) -> float:
    if not data:
        return 0.0
    printable = sum(1 for b in data if 32 <= b < 127 or b in (9, 10, 13))
    return printable / len(data)


def _tls_record_version(data: bytes) -> int:
    """Version from a plausible TLS record header at the start of a stream."""
    if len(data) < 5:
        return 0
    content_type, version, length = data[0], struct.unpack_from(">H", data, 1)[0], struct.unpack_from(">H", data, 3)[0]
    if content_type in (20, 21, 22, 23) and version in TLS_VERSIONS and 0 < length <= 0x4000:
        return version
    return 0


def _parse_tls_handshake(data: bytes) -> tuple[int | None, int]:
    """Find a ServerHello and return (cipher_suite, negotiated_version).

    Walks the record layer, concatenating handshake-type records so a
    handshake message split across records is still readable.
    """
    pos, handshake = 0, bytearray()
    while pos + 5 <= len(data):
        content_type = data[pos]
        version = struct.unpack_from(">H", data, pos + 1)[0]
        length = struct.unpack_from(">H", data, pos + 3)[0]
        if content_type not in (20, 21, 22, 23) or version not in TLS_VERSIONS or length == 0 or length > 0x4000:
            break
        if content_type == 22:
            handshake.extend(data[pos + 5: pos + 5 + length])
        pos += 5 + length

    hpos = 0
    while hpos + 4 <= len(handshake):
        msg_type = handshake[hpos]
        msg_len = int.from_bytes(handshake[hpos + 1:hpos + 4], "big")
        body = bytes(handshake[hpos + 4: hpos + 4 + msg_len])
        hpos += 4 + msg_len
        if msg_type != 2:  # only ServerHello carries the selected suite
            continue
        if len(body) < 38:
            continue
        legacy_version = struct.unpack_from(">H", body, 0)[0]
        session_id_len = body[34]
        off = 35 + session_id_len
        if len(body) < off + 2:
            continue
        suite = struct.unpack_from(">H", body, off)[0]
        version = _server_hello_supported_version(body, off + 3) or legacy_version
        return suite, version
    return None, 0


def _server_hello_supported_version(body: bytes, ext_start: int) -> int:
    """TLS 1.3 reports the real version in the supported_versions extension."""
    if len(body) < ext_start + 2:
        return 0
    ext_total = struct.unpack_from(">H", body, ext_start)[0]
    pos, end = ext_start + 2, min(len(body), ext_start + 2 + ext_total)
    while pos + 4 <= end:
        ext_type, ext_len = struct.unpack_from(">HH", body, pos)
        pos += 4
        if ext_type == 43 and ext_len >= 2:  # supported_versions
            return struct.unpack_from(">H", body, pos)[0]
        pos += ext_len
    return 0


def _looks_like_quic(data: bytes) -> bool:
    """QUIC long header: fixed bit set, followed by a known version number."""
    if len(data) < 5:
        return False
    if (data[0] & 0xC0) != 0xC0:
        return False
    version = struct.unpack_from(">I", data, 1)[0]
    return version == 1 or version == 0 or (version & 0x0F0F0F0F) == 0x0A0A0A0A or 0xFF000000 <= version <= 0xFF00002F
