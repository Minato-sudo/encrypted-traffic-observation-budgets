"""Build the session store from raw captures (Phase 2 Step 3).

Two passes over each capture. The first pass counts, per session, the exact
packet count and total byte length; the second pass collects the byte prefix
and flushes a session the moment its last packet has been seen. Knowing the
counts in advance is what makes the second pass exact and memory-bounded: no
idle timeout, no LRU eviction, and no session silently cut in half, which
matters because true flow length is one of the independent variables rather
than a nuisance parameter.
"""

from __future__ import annotations

import hashlib
import struct
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from .dissect import dissect
from .encryption import EncryptionProbe
from .pcapio import iter_packets
from .store import BYTE_CEILING, MAX_PACKETS_INDEXED, SessionRecord, StoreWriter


@dataclass
class SessionizeStats:
    files: int = 0
    packets_read: int = 0
    packets_dissected: int = 0
    sessions_seen: int = 0
    sessions_written: int = 0
    dropped_no_payload: int = 0
    dropped_duplicate: int = 0
    stratum_counts: dict = field(default_factory=lambda: defaultdict(int))

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "stratum_counts"}
        d["stratum_counts"] = dict(self.stratum_counts)
        return d


@dataclass(slots=True)
class _Open:
    """A session being assembled during the second pass."""
    prefix: bytearray
    pkt_offsets: list
    pkt_hdr_lens: list
    ident: bytes
    seen: int
    probe: EncryptionProbe
    has_payload: bool
    mac_offset: int
    ip_offset: int
    ip_version: int
    l4_offset: int
    payload_offset: int


def _pass_one(path: Path, directional: bool = False) -> tuple[dict, int, int]:
    """Count packets and bytes per session, and record each session's first timestamp."""
    counts: dict[bytes, list] = {}
    read = dissected = 0
    for ts, linktype, frame in iter_packets(path):
        read += 1
        d = dissect(frame, linktype, directional=directional)
        if d is None:
            continue
        dissected += 1
        entry = counts.get(d.key)
        if entry is None:
            counts[d.key] = [1, len(d.frame), ts]
        else:
            entry[0] += 1
            entry[1] += len(d.frame)
    return counts, read, dissected


def sessionize(
    captures: Iterable[tuple[Path, str]],
    out_path: str | Path,
    label_names: list[str],
    label_of: Callable[[Path, str], int],
    byte_ceiling: int = BYTE_CEILING,
    drop_payloadless: bool = True,
    deduplicate: bool = True,
    directional: bool = False,
    progress: Optional[Callable[[str], None]] = None,
) -> SessionizeStats:
    """Preprocess captures into a single session store.

    ``captures`` yields ``(path, group_name)`` pairs, where the group name is
    the unit the Step 3f split keeps intact -- normally the capture file, so
    no session from one capture can appear on both sides of the split.
    """
    stats = SessionizeStats()
    seen_hashes: set[int] = set()
    say = progress or (lambda _msg: None)

    with StoreWriter(out_path, label_names) as writer:
        for path, group in captures:
            path = Path(path)
            label = label_of(path, group)
            src_file = writer.register_src_file(group)
            say(f"pass 1/2 {path.name}")
            counts, read, dissected = _pass_one(path, directional=directional)
            stats.files += 1
            stats.packets_read += read
            stats.packets_dissected += dissected
            stats.sessions_seen += len(counts)

            say(f"pass 2/2 {path.name} ({len(counts)} sessions)")
            open_sessions: dict[bytes, _Open] = {}

            for _ts, linktype, frame in iter_packets(path):
                d = dissect(frame, linktype, directional=directional)
                if d is None:
                    continue
                st = open_sessions.get(d.key)
                if st is None:
                    st = _Open(
                        prefix=bytearray(), pkt_offsets=[], pkt_hdr_lens=[], ident=d.ident,
                        seen=0, probe=EncryptionProbe(proto=d.proto), has_payload=False,
                        mac_offset=d.mac_offset, ip_offset=d.ip_offset,
                        ip_version=d.ip_version, l4_offset=d.l4_offset,
                        payload_offset=d.payload_offset,
                    )
                    open_sessions[d.key] = st

                st.seen += 1
                st.has_payload = st.has_payload or d.has_payload
                st.probe.feed(d.payload, d.is_forward)
                before = len(st.prefix)
                if before < byte_ceiling:
                    st.prefix.extend(d.frame[: byte_ceiling - before])
                # Index a packet only if it actually put bytes in the prefix.
                # Recording the ones that arrive after the ceiling is reached
                # would make pkt_in_prefix count packets the prefix does not
                # contain, which silently inflates anything that stratifies or
                # slices on it.
                if len(st.prefix) > before and len(st.pkt_offsets) < MAX_PACKETS_INDEXED:
                    st.pkt_offsets.append(len(st.prefix))
                    # Header length of this packet, so Step 10(ii) can cut
                    # headers from payload across the whole prefix rather than
                    # only in the first packet.
                    st.pkt_hdr_lens.append(min(255, max(0, d.payload_offset)))

                total_packets = counts[d.key][0]
                if st.seen >= total_packets:
                    _emit(writer, d.key, st, counts[d.key], label, src_file,
                          stats, seen_hashes, drop_payloadless, deduplicate)
                    del open_sessions[d.key]

            # Any session still open means pass 1 and pass 2 disagreed, which
            # should not happen; emit rather than silently lose the data.
            for key, st in open_sessions.items():
                _emit(writer, key, st, counts[key], label, src_file,
                      stats, seen_hashes, drop_payloadless, deduplicate)

        writer.close(extra_attrs={
            "preprocess_stats": stats.as_dict(),
            "directional": int(directional),
        "drop_payloadless": int(drop_payloadless),
            "deduplicate": int(deduplicate),
        })
    return stats


def _emit(writer, key, st: _Open, count_entry, label, src_file,
          stats, seen_hashes, drop_payloadless, deduplicate) -> None:
    if drop_payloadless and not st.has_payload:
        stats.dropped_no_payload += 1
        return

    # Deduplicate on the anonymised prefix, before splitting, so an identical
    # session cannot straddle train and test (Step 3iv).
    digest = hashlib.sha256(bytes(st.prefix)).digest()
    dedup_hash = struct.unpack("<Q", digest[:8])[0]
    if deduplicate:
        if dedup_hash in seen_hashes:
            stats.dropped_duplicate += 1
            return
        seen_hashes.add(dedup_hash)

    stratum, cipher, version = st.probe.classify()
    entropy = st.probe.payload_entropy()
    stats.stratum_counts[stratum] += 1

    stats.sessions_written += 1
    writer.append(SessionRecord(
        prefix=bytes(st.prefix),
        ident=st.ident,
        pkt_offsets=st.pkt_offsets,
        pkt_hdr_lens=st.pkt_hdr_lens,
        true_length=count_entry[1],
        packet_count=count_entry[0],
        label=label,
        src_file=src_file,
        stratum=stratum,
        cipher_suite=cipher,
        tls_version=version,
        first_ts=count_entry[2],
        dedup_hash=dedup_hash,
        mac_offset=st.mac_offset,
        ip_offset=st.ip_offset,
        ip_version=st.ip_version,
        l4_offset=st.l4_offset,
        payload_offset=st.payload_offset,
        payload_entropy=entropy,
    ))
