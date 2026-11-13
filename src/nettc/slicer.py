"""The budget slicer (Phase 2 Step 4) and the padding controls (Step 11ii).

Every budget in the sweep is cut from the same preprocessed store, so the
only thing that changes between runs is how many bytes the model is allowed
to see. Two axes are supported: a byte budget, and a packet budget served by
cutting at the stored per-packet offsets.

Padding is treated as a measured variable rather than an implementation
detail, because at conventional budgets most sessions in these corpora are
padded rather than truncated. Every slice therefore reports the number of
real bytes it contains, and three mitigations are available:

* ``zero``  -- the field's default: pad with 0x00 and say nothing;
* ``embed`` -- a learned byte embedding, but padding is still 0x00 and so
               indistinguishable from a real zero byte. This exists purely so
               that ``mask`` can be attributed: it isolates the change of
               input representation from the change of padding symbol;
* ``mask``  -- a learned embedding *and* a dedicated PAD symbol (256), so
               padding is distinguishable from a real zero byte;
* ``convmask`` -- keep zero padding but hand the model the valid length so
               pooling can exclude padded positions entirely.

The fourth control, a length-balanced resample, is an index-level operation
and lives in :func:`length_balanced_indices`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np

PadMode = Literal["zero", "mask", "convmask", "embed"]
PAD_SYMBOL = 256  # one past the byte range, used only when pad_mode="mask"


@dataclass(frozen=True)
class BudgetSpec:
    """A point in the sweep. Exactly one of ``n_bytes`` / ``n_packets`` is set."""
    n_bytes: Optional[int] = None
    n_packets: Optional[int] = None
    width: Optional[int] = None   # tensor width for a packet budget

    def __post_init__(self):
        if (self.n_bytes is None) == (self.n_packets is None):
            raise ValueError("specify exactly one of n_bytes or n_packets")

    @property
    def axis(self) -> str:
        return "bytes" if self.n_bytes is not None else "packets"

    @property
    def label(self) -> str:
        return f"{self.n_bytes}B" if self.n_bytes is not None else f"{self.n_packets}P"

    def tensor_width(self) -> int:
        if self.n_bytes is not None:
            return self.n_bytes
        if self.width is None:
            raise ValueError("a packet budget needs a width; call resolve_packet_width")
        return self.width


@dataclass
class Slice:
    x: np.ndarray            # (n, width) uint8, or int16 when pad_mode="mask"
    valid_len: np.ndarray    # (n,) real bytes present, before padding
    width: int

    @property
    def padding_fraction(self) -> np.ndarray:
        return 1.0 - np.minimum(self.valid_len, self.width) / float(self.width)

    @property
    def truncated(self) -> np.ndarray:
        return self.valid_len > self.width


def resolve_packet_width(store, spec: BudgetSpec, idx: np.ndarray,
                         quantile: float = 0.95, multiple: int = 64) -> BudgetSpec:
    """Pick a tensor width for a packet budget from the reference split.

    A packet budget does not fix a byte count, so the tensor needs a width.
    It is taken from a quantile of the byte lengths the budget actually
    produces on the *training* split only -- never the test split -- and
    rounded up, so the choice cannot leak test information.
    """
    if spec.n_packets is None:
        return spec
    lengths = _packet_budget_lengths(store, spec.n_packets, idx)
    if lengths.size == 0:
        width = multiple
    else:
        q = float(np.quantile(lengths, quantile))
        width = int(min(store.byte_ceiling, max(multiple, int(np.ceil(q / multiple) * multiple))))
    return BudgetSpec(n_packets=spec.n_packets, width=width)


def _packet_budget_lengths(store, n_packets: int, idx: np.ndarray) -> np.ndarray:
    """Bytes available when a session is cut after its first ``n_packets`` packets."""
    k = min(n_packets, store.max_packets_indexed)
    offsets = store.pkt_offsets[idx]                     # (n, max_packets_indexed)
    in_prefix = store.pkt_in_prefix[idx]
    take = np.minimum(in_prefix, k) - 1                  # index of the last packet we keep
    lengths = np.where(take >= 0, offsets[np.arange(idx.size), np.maximum(take, 0)], 0)
    # If the budget exceeds what the 2048-byte prefix holds, the cut is the
    # whole prefix: the ceiling, not the session, is the binding constraint.
    exhausted = in_prefix < k
    lengths = np.where(exhausted, store.prefix_len[idx], lengths)
    return lengths.astype(np.int64)


def layer_scoped(store, idx: np.ndarray, scope: str) -> tuple:
    """Rebuild each prefix keeping only headers, or only payload (Step 10ii).

    Motivated by two findings in the review that point the same way: P2's best
    configuration uses header-derived attributes and *zero* payload bytes, and
    P5 reports that encrypted payload alone supports accuracies of roughly
    0.12 and 0.30. If headers alone match the full prefix here, then the
    budget being swept is mostly a budget over headers.

    Returns ``(bytes, lengths)`` where each row is the kept bytes packed to
    the left, so the result slots into the normal budget machinery.
    """
    if scope not in ("headers", "payload"):
        raise ValueError(f"scope must be 'headers' or 'payload', not {scope!r}")
    raw = store.get_bytes(idx)
    offsets = store.pkt_offsets[idx]
    hdr_lens = store.pkt_hdr_lens[idx]
    counts = store.pkt_in_prefix[idx]

    out = np.zeros_like(raw)
    lengths = np.zeros(idx.size, dtype=np.int64)
    for row in range(idx.size):
        start, write = 0, 0
        for p in range(int(counts[row])):
            end = int(offsets[row, p])
            cut = min(start + int(hdr_lens[row, p]), end)
            lo, hi = (start, cut) if scope == "headers" else (cut, end)
            if hi > lo:
                out[row, write:write + (hi - lo)] = raw[row, lo:hi]
                write += hi - lo
            start = end
        lengths[row] = write
    return out, lengths


def mask_ports(store, idx: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """Zero the source and destination port bytes in every packet of a prefix.

    Phase 2 Step 3c retained ports, citing P1's ablation that they carry
    signal, and promised to test the consequence in Step 11. On USTC-TFC2016
    that test returns 0.751 accuracy from the four port bytes alone against a
    0.136 majority-class rate -- three quarters of the gain over chance that
    the full 784-byte model achieves. A port-masked arm is therefore not
    optional: without it the sweep cannot distinguish "how many bytes of
    traffic a classifier needs" from "how early the port number appears".

    Ports sit at a fixed offset within each packet for a given encapsulation,
    so the per-packet start offsets locate every copy, not just the first.
    """
    out = raw.copy()
    starts = store.pkt_offsets[idx]
    counts = store.pkt_in_prefix[idx]
    # Offset of the L4 header within a frame, taken from the first packet.
    # Constant for a given encapsulation; IP options would move it, and those
    # sessions are left unmasked rather than mis-masked.
    l4_rel = store.l4_offset[idx]
    width = raw.shape[1]
    for row in range(idx.size):
        rel = int(l4_rel[row])
        packet_start = 0
        for p in range(int(counts[row])):
            lo = packet_start + rel
            if lo + 4 <= width:
                out[row, lo:lo + 4] = 0
            packet_start = int(starts[row, p])
    return out


def take(store, idx: np.ndarray, spec: BudgetSpec,
         pad_mode: PadMode = "zero", source: str = "bytes") -> Slice:
    """Cut ``idx`` sessions down to ``spec`` and pad to a fixed width.

    ``source`` selects what the model is shown, which is how the Step 11
    controls are built. See :func:`_positional_source` for the alternatives
    to the session bytes.
    """
    idx = np.asarray(idx)
    if source in ("headers", "payload"):
        packed, avail = layer_scoped(store, idx, source)
        width = spec.tensor_width()
        keep = np.minimum(avail, width)
        out = np.zeros((idx.size, width), dtype=np.uint8)
        cut = packed[:, :width]
        region = np.arange(width)[None, :] < keep[:, None]
        out[region] = cut[region]
        if pad_mode in ("mask", "embed"):
            x = out.astype(np.int16)
            if pad_mode == "mask":
                x[~region] = PAD_SYMBOL
        else:
            x = out
        return Slice(x=x, valid_len=avail, width=width)

    if source not in ("bytes", "no_ports"):
        raw = _positional_source(store, idx, source)
        width = raw.shape[1]
        valid = np.full(idx.size, width, dtype=np.int64)
        x = raw.astype(np.int16) if pad_mode in ("mask", "embed") else raw
        return Slice(x=x, valid_len=valid, width=width)

    raw = store.get_bytes(idx)                            # (n, byte_ceiling)
    if source == "no_ports":
        raw = mask_ports(store, idx, raw)
    width = spec.tensor_width()

    if spec.n_bytes is not None:
        avail = store.prefix_len[idx].astype(np.int64)
    else:
        avail = _packet_budget_lengths(store, spec.n_packets, idx)

    keep = np.minimum(avail, width)
    out = np.zeros((idx.size, width), dtype=np.uint8)
    cut = raw[:, :width]
    # Column-wise mask beats a per-row Python loop by a wide margin here, and
    # the sweep calls this for every budget and every seed.
    col = np.arange(width)[None, :]
    region = col < keep[:, None]
    out[region] = cut[region]

    if pad_mode in ("mask", "embed"):
        x = out.astype(np.int16)
        if pad_mode == "mask":
            x[~region] = PAD_SYMBOL
    else:
        x = out
    return Slice(x=x, valid_len=avail, width=width)


def _gather(raw: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
    """Read ``length`` bytes from each row of ``raw`` starting at ``starts``."""
    cols = starts[:, None] + np.arange(length)[None, :]
    valid = (cols >= 0) & (cols < raw.shape[1])
    out = np.zeros((raw.shape[0], length), dtype=raw.dtype)
    rows = np.repeat(np.arange(raw.shape[0])[:, None], length, axis=1)
    out[valid] = raw[rows[valid], cols[valid]]
    return out


def _positional_source(store, idx: np.ndarray, source: str) -> np.ndarray:
    """Inputs for the Step 11 leakage controls, read from recorded offsets.

    ``ident``        the MAC and IP bytes as they were *before* anonymisation.
                     Training on these measures how strong the identifier
                     shortcut would have been, which is the quantity P5
                     measured across 348 occlusion experiments. It is context,
                     not a gate: it says what anonymisation had to destroy.

    ``id_positions`` the same byte positions read out of the *stored*,
                     anonymised prefix. This is the gate. If anonymisation
                     worked these are all zero and a model trained on them
                     cannot beat the majority class; anything better means the
                     zeroing did not land where it was supposed to.

    ``ports``        the four port bytes, which Step 3c deliberately retains
                     because P1's ablation shows they carry signal. If ports
                     alone classify well, that is a leakage finding about our
                     own pipeline and has to be reported next to the headline
                     numbers rather than discovered later.
    """
    if source == "ident":
        return store.get_ident(idx)

    raw = store.get_bytes(idx)
    if source == "ports":
        return _gather(raw, store.l4_offset[idx], 4)
    if source == "id_positions":
        mac = _gather(raw, np.maximum(store.mac_offset[idx], 0), 12)
        mac[store.mac_offset[idx] < 0] = 0
        addr_len = np.where(store.ip_version[idx] == 4, 4, 16)
        ips = np.zeros((idx.size, 32), dtype=raw.dtype)
        for alen in (4, 16):
            m = addr_len == alen
            if not m.any():
                continue
            pair = _gather(raw[m], store.ip_offset[idx][m], 2 * alen)
            ips[np.flatnonzero(m)[:, None], np.arange(2 * alen)[None, :]] = pair
        return np.concatenate([mac, ips], axis=1)
    raise ValueError(f"unknown source {source!r}")


def length_balanced_indices(store, idx: np.ndarray, n_bins: int = 8,
                            seed: int = 0, by: str = "true_length",
                            edges: Optional[np.ndarray] = None,
                            classes: Optional[Sequence[int]] = None,
                            report: Optional[dict] = None) -> np.ndarray:
    """Resample ``idx`` so every retained class shares one flow-length distribution.

    The third padding control of Step 11ii. If a class is separable only
    because its flows are a characteristic length, that advantage disappears
    here, which is the point: it isolates length from content.

    **Pass ``edges`` and ``classes`` from the training split when balancing
    validation or test.** Derived independently per split they will disagree:
    on ISCX-VPN the train split retains {chat, email, file, voip, vpn_file,
    vpn_voip} and the test split {chat, email, streaming, vpn_chat, vpn_p2p,
    vpn_voip}, overlapping in three of six. A model trained on one label set
    and scored against another collapses to chance, which reads exactly like
    the dramatic finding this control is meant to detect and is instead a
    broken experiment. :func:`length_balance_plan` derives both from train.

    Balancing is exact within the classes it retains. Bins that some retained
    class never occupies are dropped whole, because taking only the classes
    present would reintroduce the length-label correlation being removed.
    """
    rng = np.random.default_rng(seed)
    values = getattr(store, by)[idx]
    labels = store.label[idx]
    if idx.size == 0:
        return idx

    if edges is None:
        edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
        edges[-1] = np.nextafter(edges[-1], np.inf)
    edges = np.asarray(edges, dtype=np.float64)
    bins = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)
    n_bins_eff = len(edges) - 1

    occupied = [b for b in range(n_bins_eff) if (bins == b).any()]
    if classes is None:
        all_classes = np.unique(labels)
        classes = [c for c in all_classes
                   if all(((bins == b) & (labels == c)).any() for b in occupied)]
    classes = list(classes)

    info = {"n_bins": n_bins_eff, "classes_total": int(np.unique(labels).size),
            "classes_spanning_all_bins": [store.label_names[c] for c in classes],
            "classes_excluded": [store.label_names[c] for c in np.unique(labels)
                                 if c not in classes],
            "edges": [float(e) for e in edges]}
    if len(classes) < 2:
        info["outcome"] = ("fewer than two classes span every length bin; the "
                           "length-balanced control cannot be formed")
        if report is not None:
            report.update(info)
        return idx[:0]

    chosen = []
    for b in occupied:
        per_class = {c: np.flatnonzero((bins == b) & (labels == c)) for c in classes}
        quota = min(v.size for v in per_class.values())
        if quota == 0:
            continue
        for c in classes:
            chosen.append(rng.choice(per_class[c], size=quota, replace=False))
    if not chosen:
        info["outcome"] = "no bin holds every retained class; nothing could be balanced"
        if report is not None:
            report.update(info)
        return idx[:0]

    picked = np.concatenate(chosen)
    info["outcome"] = (f"balanced {len(classes)} of {info['classes_total']} classes "
                       f"across {len(occupied)} length bins")
    info["n_retained"] = int(picked.size)
    info["fraction_retained"] = float(picked.size / idx.size)
    if report is not None:
        report.update(info)
    return idx[np.sort(picked)]


def length_balance_plan(store, train_idx: np.ndarray, n_bins: int = 8,
                        by: str = "true_length") -> tuple:
    """Derive bin edges and the retained class set from the training split.

    Returns ``(edges, classes)`` to be passed to
    :func:`length_balanced_indices` for every split, so train, validation and
    test are balanced over the same bins and the same labels.
    """
    values = getattr(store, by)[train_idx]
    labels = store.label[train_idx]
    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    bins = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, n_bins - 1)
    occupied = [b for b in range(n_bins) if (bins == b).any()]
    classes = [int(c) for c in np.unique(labels)
               if all(((bins == b) & (labels == c)).any() for b in occupied)]
    return edges, classes
