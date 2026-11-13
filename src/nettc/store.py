"""The HDF5 session store (Phase 2 Step 3e).

Everything is preprocessed exactly once, at a generous byte ceiling, so that
every budget in the sweep is served by slicing this store rather than by
re-running preprocessing. That is what keeps architecture and training
constant across budgets, which is the whole point of the comparison.

Each session record carries the four independent variables named in Step 1 --
true byte length, packet count, padding fraction (derived from prefix_len)
and encryption stratum -- plus the per-packet offsets that let a *packet*
budget be sliced out of the same bytes, and the identifier bytes that
anonymisation removed, for the Step 11 leakage control.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

BYTE_CEILING = 2048     # Step 3e: every budget in the sweep fits under this
MAX_PACKETS_INDEXED = 32  # covers the largest packet budget we sweep (20)
IDENT_LEN = 48

_FIELDS: dict[str, tuple[str, tuple[int, ...]]] = {
    "bytes":         ("u1", (BYTE_CEILING,)),
    "ident":         ("u1", (IDENT_LEN,)),
    "pkt_offsets":   ("u2", (MAX_PACKETS_INDEXED,)),
    "pkt_hdr_lens":  ("u1", (MAX_PACKETS_INDEXED,)),
    "prefix_len":    ("u2", ()),
    "pkt_in_prefix": ("u1", ()),   # packets with at least one byte in the prefix
    "true_length":   ("u4", ()),
    "packet_count":  ("u4", ()),
    "label":         ("u1", ()),
    "src_file":      ("u2", ()),
    "stratum":       ("u1", ()),
    "cipher_suite":  ("u2", ()),
    "tls_version":   ("u2", ()),
    "first_ts":      ("f8", ()),
    "dedup_hash":    ("u8", ()),
    "mac_offset":    ("i2", ()),
    "ip_offset":     ("u2", ()),
    "ip_version":    ("u1", ()),
    "l4_offset":     ("u2", ()),
    "payload_offset": ("u2", ()),
    "payload_entropy": ("f4", ()),   # bits per byte; 8.0 is uniform random
}


@dataclass(slots=True)
class SessionRecord:
    prefix: bytes
    ident: bytes
    pkt_offsets: list[int]
    pkt_hdr_lens: list[int]
    true_length: int
    packet_count: int
    label: int
    src_file: int
    stratum: int
    cipher_suite: int
    tls_version: int
    first_ts: float
    dedup_hash: int
    mac_offset: int
    ip_offset: int
    ip_version: int
    l4_offset: int
    payload_offset: int
    payload_entropy: float


class StoreWriter:
    """Buffered appender for the session store."""

    def __init__(self, path: str | Path, label_names: list[str], flush_every: int = 4096):
        self.path = Path(path)
        self.label_names = label_names
        self.flush_every = flush_every
        self._buf: list[SessionRecord] = []
        self._n = 0
        self._src_files: list[str] = []
        self._fh = h5py.File(self.path, "w")
        for name, (dtype, shape) in _FIELDS.items():
            self._fh.create_dataset(
                name,
                shape=(0, *shape),
                maxshape=(None, *shape),
                dtype=dtype,
                chunks=(min(1024, flush_every), *shape) if shape else (flush_every,),
                compression="lzf" if name in ("bytes", "ident", "pkt_offsets",
                                              "pkt_hdr_lens") else None,
            )
        self._fh.attrs["label_names"] = json.dumps(label_names)
        self._fh.attrs["byte_ceiling"] = BYTE_CEILING
        self._fh.attrs["max_packets_indexed"] = MAX_PACKETS_INDEXED
        self._fh.attrs["ident_len"] = IDENT_LEN

    def register_src_file(self, name: str) -> int:
        """Register a capture file and return its id, used for group-wise splits."""
        if name in self._src_files:
            return self._src_files.index(name)
        self._src_files.append(name)
        return len(self._src_files) - 1

    def append(self, rec: SessionRecord) -> None:
        self._buf.append(rec)
        if len(self._buf) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        k = len(self._buf)
        start, end = self._n, self._n + k

        bytes_arr = np.zeros((k, BYTE_CEILING), dtype=np.uint8)
        ident_arr = np.zeros((k, IDENT_LEN), dtype=np.uint8)
        offs_arr = np.zeros((k, MAX_PACKETS_INDEXED), dtype=np.uint16)
        hdr_arr = np.zeros((k, MAX_PACKETS_INDEXED), dtype=np.uint8)
        scalars = {name: np.zeros(k, dtype=dtype)
                   for name, (dtype, shape) in _FIELDS.items() if not shape}

        for i, r in enumerate(self._buf):
            n = min(len(r.prefix), BYTE_CEILING)
            bytes_arr[i, :n] = np.frombuffer(r.prefix[:n], dtype=np.uint8)
            ident_arr[i] = np.frombuffer(r.ident[:IDENT_LEN].ljust(IDENT_LEN, b"\x00"), dtype=np.uint8)
            m = min(len(r.pkt_offsets), MAX_PACKETS_INDEXED)
            offs_arr[i, :m] = np.asarray(r.pkt_offsets[:m], dtype=np.uint16)
            hdr_arr[i, :m] = np.asarray(r.pkt_hdr_lens[:m], dtype=np.uint8)
            scalars["prefix_len"][i] = n
            scalars["pkt_in_prefix"][i] = m
            scalars["true_length"][i] = min(r.true_length, np.iinfo(np.uint32).max)
            scalars["packet_count"][i] = min(r.packet_count, np.iinfo(np.uint32).max)
            scalars["label"][i] = r.label
            scalars["src_file"][i] = r.src_file
            scalars["stratum"][i] = r.stratum
            scalars["cipher_suite"][i] = r.cipher_suite
            scalars["tls_version"][i] = r.tls_version
            scalars["first_ts"][i] = r.first_ts
            scalars["dedup_hash"][i] = r.dedup_hash
            scalars["mac_offset"][i] = r.mac_offset
            scalars["ip_offset"][i] = r.ip_offset
            scalars["ip_version"][i] = r.ip_version
            scalars["l4_offset"][i] = r.l4_offset
            scalars["payload_offset"][i] = r.payload_offset
            scalars["payload_entropy"][i] = r.payload_entropy

        for name, arr in (("bytes", bytes_arr), ("ident", ident_arr),
                          ("pkt_offsets", offs_arr), ("pkt_hdr_lens", hdr_arr)):
            ds = self._fh[name]
            ds.resize(end, axis=0)
            ds[start:end] = arr
        for name, arr in scalars.items():
            ds = self._fh[name]
            ds.resize(end, axis=0)
            ds[start:end] = arr

        self._n = end
        self._buf.clear()

    def close(self, extra_attrs: Optional[dict] = None) -> int:
        self.flush()
        self._fh.attrs["src_file_names"] = json.dumps(self._src_files)
        self._fh.attrs["n_sessions"] = self._n
        for key, value in (extra_attrs or {}).items():
            self._fh.attrs[key] = value if isinstance(value, (int, float, str)) else json.dumps(value)
        self._fh.close()
        return self._n

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self._fh:
            self.close()


class Store:
    """Read-side view of the session store, loaded into memory as arrays.

    The byte matrix for a full corpus at a 2048-byte ceiling is a few
    gigabytes, so ``lazy_bytes=True`` keeps it on disk and slices per batch.
    """

    def __init__(self, path: str | Path, lazy_bytes: bool = False):
        self.path = Path(path)
        self._fh = h5py.File(self.path, "r")
        self.label_names: list[str] = json.loads(self._fh.attrs["label_names"])
        self.src_file_names: list[str] = json.loads(self._fh.attrs.get("src_file_names", "[]"))
        self.byte_ceiling = int(self._fh.attrs["byte_ceiling"])
        self.max_packets_indexed = int(self._fh.attrs["max_packets_indexed"])
        self.n = int(self._fh["label"].shape[0])

        self.prefix_len = self._fh["prefix_len"][:].astype(np.int32)
        self.pkt_in_prefix = self._fh["pkt_in_prefix"][:].astype(np.int32)
        self.true_length = self._fh["true_length"][:].astype(np.int64)
        self.packet_count = self._fh["packet_count"][:].astype(np.int64)
        self.label = self._fh["label"][:].astype(np.int64)
        self.src_file = self._fh["src_file"][:].astype(np.int32)
        self.stratum = self._fh["stratum"][:].astype(np.int32)
        self.cipher_suite = self._fh["cipher_suite"][:].astype(np.int32)
        self.tls_version = self._fh["tls_version"][:].astype(np.int32)
        self.first_ts = self._fh["first_ts"][:]
        self.pkt_offsets = self._fh["pkt_offsets"][:].astype(np.int32)
        self.pkt_hdr_lens = self._fh["pkt_hdr_lens"][:].astype(np.int32)
        self.mac_offset = self._fh["mac_offset"][:].astype(np.int32)
        self.ip_offset = self._fh["ip_offset"][:].astype(np.int32)
        self.ip_version = self._fh["ip_version"][:].astype(np.int32)
        self.l4_offset = self._fh["l4_offset"][:].astype(np.int32)
        self.payload_offset = self._fh["payload_offset"][:].astype(np.int32)
        self.payload_entropy = self._fh["payload_entropy"][:].astype(np.float32)

        self._lazy = lazy_bytes
        self.bytes = None if lazy_bytes else self._fh["bytes"][:]
        self.ident = None if lazy_bytes else self._fh["ident"][:]

    def get_bytes(self, idx: np.ndarray) -> np.ndarray:
        if self.bytes is not None:
            return self.bytes[idx]
        order = np.argsort(idx)           # h5py fancy indexing needs increasing order
        out = self._fh["bytes"][idx[order]]
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        return out[inverse]

    def get_ident(self, idx: np.ndarray) -> np.ndarray:
        if self.ident is not None:
            return self.ident[idx]
        order = np.argsort(idx)
        out = self._fh["ident"][idx[order]]
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        return out[inverse]

    @property
    def n_classes(self) -> int:
        return len(self.label_names)

    def close(self) -> None:
        self._fh.close()

    def __len__(self) -> int:
        return self.n
