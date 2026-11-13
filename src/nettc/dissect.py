"""Per-packet layer dissection, anonymisation, and encryption tagging.

Three jobs, all required by Phase 2 Step 3:

* derive the direction-agnostic five-tuple that defines a session (Step 3a);
* zero the MAC and IP fields while retaining ports, and hand back the bytes
  that were removed so the leakage control of Step 11 can be trained on
  exactly the fields anonymisation was supposed to destroy (Step 3c);
* expose the L4 payload so TLS cipher suites can be recorded and each
  session assigned to an encrypted or plaintext stratum (Step 3b, after P5).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional

# Link types we handle. USTC-TFC2016 and ISCX VPN-nonVPN are Ethernet, but
# raw-IP and Linux cooked captures turn up in redistributed copies.
LINKTYPE_NULL = 0
LINKTYPE_ETHERNET = 1
LINKTYPE_RAW = 101
LINKTYPE_LINUX_SLL = 113
LINKTYPE_IPV4 = 228
LINKTYPE_IPV6 = 229
LINKTYPE_LINUX_SLL2 = 276

IDENT_LEN = 48  # 12 bytes of MAC pair + up to 32 bytes of IP pair, padded

# Encryption strata, stored as a small integer per session.
# Encryption strata. The split that matters for Gap 3 is encrypted against
# not, but "high entropy" is not the same claim as "encrypted" -- compressed
# payload is also near-random -- so the uncertain case gets its own name
# rather than being folded into either side.
STRATUM_UNKNOWN = 0            # no payload to judge
STRATUM_PLAINTEXT = 1          # printable text protocol
STRATUM_ENCRYPTED_TLS = 2      # TLS handshake parsed; cipher suite recorded
STRATUM_ENCRYPTED_OTHER = 3    # SSH or QUIC identified by protocol signature
STRATUM_BINARY = 4             # structured binary, low entropy: not encrypted
STRATUM_OPAQUE = 5             # near-random bytes: encrypted or compressed
STRATUM_NAMES = ("unknown", "plaintext", "encrypted_tls", "encrypted_other",
                 "unencrypted_binary", "opaque_high_entropy")

# Strata that are definitely not encrypted, for the Gap 3 headline figure.
STRATA_UNENCRYPTED = (STRATUM_PLAINTEXT, STRATUM_BINARY)
STRATA_ENCRYPTED = (STRATUM_ENCRYPTED_TLS, STRATUM_ENCRYPTED_OTHER)


@dataclass(slots=True)
class Dissected:
    """One packet, anonymised, with everything the sessioniser needs."""
    key: bytes                  # packed canonical direction-agnostic five-tuple
    frame: bytes                # anonymised bytes, all layers
    ident: bytes                # the IDENT_LEN bytes anonymisation removed
    payload: bytes              # L4 payload (TCP/UDP), for TLS inspection
    is_forward: bool            # True when the packet runs low->high endpoint
    proto: int
    has_payload: bool
    # Where the fields we anonymise and retain actually sit in ``frame``.
    # Recorded rather than assumed, because VLAN tags and IP options move
    # them, and the Step 11 controls need to read them back exactly.
    mac_offset: int             # -1 when the link layer carries no MACs
    ip_offset: int
    ip_version: int
    l4_offset: int
    payload_offset: int         # first application-layer byte; the header/payload cut


def _l3_offset(linktype: int, frame: bytes) -> tuple[int, Optional[int], int]:
    """Return (l3_offset, ethertype, mac_offset) or (-1, None, -1) if unusable.

    ``mac_offset`` is where the 12-byte MAC pair starts, or -1 when the link
    layer carries no MAC addresses.
    """
    if linktype == LINKTYPE_ETHERNET:
        if len(frame) < 14:
            return -1, None, -1
        ethertype = struct.unpack_from(">H", frame, 12)[0]
        off = 14
        # Walk any stack of 802.1Q / 802.1ad VLAN tags.
        while ethertype in (0x8100, 0x88A8, 0x9100) and len(frame) >= off + 4:
            ethertype = struct.unpack_from(">H", frame, off + 2)[0]
            off += 4
        return off, ethertype, 0
    if linktype == LINKTYPE_NULL:
        if len(frame) < 4:
            return -1, None, -1
        family = struct.unpack_from("<I", frame, 0)[0]
        return 4, (0x0800 if family == 2 else 0x86DD), -1
    if linktype in (LINKTYPE_RAW, LINKTYPE_IPV4, LINKTYPE_IPV6):
        if not frame:
            return -1, None, -1
        version = frame[0] >> 4
        return 0, (0x0800 if version == 4 else 0x86DD if version == 6 else None), -1
    if linktype == LINKTYPE_LINUX_SLL:
        if len(frame) < 16:
            return -1, None, -1
        return 16, struct.unpack_from(">H", frame, 14)[0], -1
    if linktype == LINKTYPE_LINUX_SLL2:
        if len(frame) < 20:
            return -1, None, -1
        return 20, struct.unpack_from(">H", frame, 0)[0], -1
    return -1, None, -1


def dissect(frame: bytes, linktype: int,
            directional: bool = False) -> Optional[Dissected]:
    """Parse, anonymise and key one frame. Returns None if it is not IP/TCP/UDP.

    ``directional=True`` keys each direction separately, giving unidirectional
    flows instead of bidirectional sessions. That is the Step 10(iii)
    granularity ablation: the studies under review disagree about which unit a
    classifier should see, and the two halves of a session carry very
    different amounts of signal.
    """
    l3, ethertype, mac_off = _l3_offset(linktype, frame)
    if l3 < 0 or ethertype not in (0x0800, 0x86DD):
        return None

    if ethertype == 0x0800:
        if len(frame) < l3 + 20:
            return None
        ihl = (frame[l3] & 0x0F) * 4
        if ihl < 20 or len(frame) < l3 + ihl:
            return None
        proto = frame[l3 + 9]
        # IPv4 fragments other than the first carry no usable L4 header.
        frag_off = struct.unpack_from(">H", frame, l3 + 6)[0] & 0x1FFF
        src_ip_off, dst_ip_off, addr_len = l3 + 12, l3 + 16, 4
        total_len = struct.unpack_from(">H", frame, l3 + 2)[0]
        l4 = l3 + ihl
        l3_payload_end = min(len(frame), l3 + total_len) if total_len >= ihl else len(frame)
        if frag_off != 0:
            return None
    else:
        if len(frame) < l3 + 40:
            return None
        proto = frame[l3 + 6]
        src_ip_off, dst_ip_off, addr_len = l3 + 8, l3 + 24, 16
        payload_len = struct.unpack_from(">H", frame, l3 + 4)[0]
        l4 = l3 + 40
        l3_payload_end = min(len(frame), l4 + payload_len) if payload_len else len(frame)
        # Skip the common extension headers rather than mis-reading them as L4.
        hops = 0
        while proto in (0, 43, 44, 60) and len(frame) >= l4 + 8 and hops < 8:
            proto, ext_len = frame[l4], (frame[l4 + 1] + 1) * 8
            l4 += ext_len
            hops += 1

    if proto not in (6, 17):  # TCP, UDP
        return None

    if proto == 6:
        if len(frame) < l4 + 20:
            return None
        sport, dport = struct.unpack_from(">HH", frame, l4)
        data_off = ((frame[l4 + 12] >> 4) & 0x0F) * 4
        if data_off < 20:
            return None
        payload_start = l4 + data_off
    else:
        if len(frame) < l4 + 8:
            return None
        sport, dport = struct.unpack_from(">HH", frame, l4)
        udp_len = struct.unpack_from(">H", frame, l4 + 4)[0]
        payload_start = l4 + 8
        if udp_len >= 8:
            l3_payload_end = min(l3_payload_end, l4 + udp_len)

    payload = bytes(frame[payload_start:max(payload_start, l3_payload_end)])

    src_ip = frame[src_ip_off:src_ip_off + addr_len]
    dst_ip = frame[dst_ip_off:dst_ip_off + addr_len]

    # Canonical ordering makes the key direction-agnostic: the endpoint that
    # sorts lower is always first, so both directions land in one session.
    a, b = (src_ip, sport), (dst_ip, dport)
    is_forward = True if directional else (a <= b)
    low, high = (a, b) if is_forward else (b, a)
    # Packed rather than a tuple: a long capture holds millions of these in a
    # dict, and the bytes form costs a fraction of the memory.
    key = low[0] + struct.pack(">H", low[1]) + high[0] + struct.pack(">H", high[1]) + bytes((proto,))

    # Anonymise: zero MACs and IPs in place, retain ports (Step 3c). The
    # removed bytes are kept, ordered canonically, for the leakage probe.
    buf = bytearray(frame)
    ident = bytearray(IDENT_LEN)
    if mac_off >= 0:
        src_mac = bytes(frame[mac_off + 6:mac_off + 12])   # Ethernet puts dst first
        dst_mac = bytes(frame[mac_off:mac_off + 6])
        lo_mac, hi_mac = (src_mac, dst_mac) if is_forward else (dst_mac, src_mac)
        ident[0:6], ident[6:12] = lo_mac, hi_mac
        buf[mac_off:mac_off + 12] = b"\x00" * 12
    lo_ip, hi_ip = (src_ip, dst_ip) if is_forward else (dst_ip, src_ip)
    ident[12:12 + addr_len] = lo_ip
    ident[12 + addr_len:12 + 2 * addr_len] = hi_ip
    buf[src_ip_off:src_ip_off + addr_len] = b"\x00" * addr_len
    buf[dst_ip_off:dst_ip_off + addr_len] = b"\x00" * addr_len

    return Dissected(
        key=key,
        frame=bytes(buf[:l3_payload_end] if l3_payload_end > l3 else buf),
        ident=bytes(ident),
        payload=payload,
        is_forward=is_forward,
        proto=proto,
        has_payload=len(payload) > 0,
        mac_offset=mac_off,
        ip_offset=src_ip_off,
        ip_version=4 if addr_len == 4 else 6,
        l4_offset=l4,
        payload_offset=payload_start,
    )
