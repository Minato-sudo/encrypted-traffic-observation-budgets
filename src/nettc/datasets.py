"""Corpus adapters for the two benchmarks of Phase 2 Step 2.

Each adapter answers two questions: which capture files belong to the corpus,
and what class a given file's sessions carry. Labelling is kept here, out of
the preprocessing code, because the two corpora differ in a way that matters
downstream: USTC-TFC2016 is one pcap per class, so a group-wise split by
capture file is impossible there and ``split.make_split`` falls back to a
temporal cut. ISCX VPN-nonVPN has many files per class and gets a real
group-wise split. Which applies is recorded in the split manifest.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

CAPTURE_SUFFIXES = (".pcap", ".pcapng", ".cap")


@dataclass
class Corpus:
    name: str
    captures: list          # [(path, group_name)]
    label_names: list
    label_of: Callable      # (path, group) -> int
    notes: str = ""

    def summary(self) -> str:
        from collections import Counter
        counts = Counter(self.label_names[self.label_of(p, g)] for p, g in self.captures)
        lines = [f"{self.name}: {len(self.captures)} captures, {len(self.label_names)} classes"]
        for name in self.label_names:
            lines.append(f"  {name:<28} {counts.get(name, 0):>3} capture file(s)")
        if self.notes:
            lines.append(f"  note: {self.notes}")
        return "\n".join(lines)


def find_captures(root: str | Path) -> list:
    root = Path(root)
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in CAPTURE_SUFFIXES)


# --------------------------------------------------------------- USTC-TFC2016

# Ten malware and ten benign classes, one pcap per class.
USTC_MALWARE = ["Cridex", "Geodo", "Htbot", "Miuref", "Neris", "Nsis-ay",
                "Shifu", "Tinba", "Virut", "Zeus"]
USTC_BENIGN = ["BitTorrent", "Facetime", "FTP", "Gmail", "MySQL", "Outlook",
               "Skype", "SMB", "Weibo", "WorldOfWarcraft"]


def ustc_tfc2016(root: str | Path, binary: bool = False) -> Corpus:
    """USTC-TFC2016. ``binary=True`` collapses to malware against benign."""
    captures = find_captures(root)
    if not captures:
        raise FileNotFoundError(f"no capture files under {root}")

    canonical = {n.lower().replace("-", "").replace("_", ""): n
                 for n in USTC_MALWARE + USTC_BENIGN}

    def resolve(path: Path) -> str:
        stem = path.stem.lower().replace("-", "").replace("_", "")
        if stem in canonical:
            return canonical[stem]
        # Redistributed copies prefix or suffix the class name.
        for key, name in canonical.items():
            if key in stem:
                return name
        raise KeyError(f"cannot map {path.name!r} to a USTC-TFC2016 class")

    if binary:
        label_names = ["benign", "malware"]

        def label_of(path: Path, _group: str) -> int:
            return 1 if resolve(path) in USTC_MALWARE else 0
    else:
        label_names = USTC_MALWARE + USTC_BENIGN

        def label_of(path: Path, _group: str) -> int:
            return label_names.index(resolve(path))

    return Corpus(
        name="USTC-TFC2016" + (" (binary)" if binary else ""),
        captures=[(p, p.stem) for p in captures],
        label_names=label_names,
        label_of=label_of,
        notes="one capture per class, so the split falls back to a temporal cut "
              "rather than a group-wise one",
    )


# ----------------------------------------------------------- ISCX VPN-nonVPN

# The filenames encode an application and whether the flow was tunnelled.
# Two label schemes are in use across the literature and both are offered,
# because results are not comparable between them: the 6-class "traffic
# category" scheme and the finer per-application scheme.
# Order matters: the first match wins, so the more specific application
# patterns must precede the broader ones. Two traps in the real filenames:
# "gmailchat" is Gmail's *chat* feature, not mail, and would otherwise be
# caught by the "gmail" in the email rule; and the separator is inconsistent,
# with both "facebook_chat_4a.pcap" and "facebookchat1.pcapng" present, so
# every multi-word application needs an optional underscore.
ISCX_CATEGORY_PATTERNS = [
    ("chat",      r"(aim|icq|facebook_?chat|hangouts?_?chat|skype_?chat|gmail_?chat)"),
    ("email",     r"(email|gmail|imap|pop3|smtp)"),
    ("file",      r"(ftps?|sftp|scp|skype_?files?|dropbox)"),
    ("p2p",       r"(torrent|bittorrent|p2p)"),
    ("streaming", r"(netflix|spotify|vimeo|youtube|hangouts?_?video|facebook_?video|skype_?video)"),
    ("voip",      r"(voip|voipbuster|hangouts?_?audio|facebook_?audio|skype_?audio)"),
]


def _iscx_category(stem: str) -> Optional[str]:
    s = stem.lower()
    for name, pattern in ISCX_CATEGORY_PATTERNS:
        if re.search(pattern, s):
            return name
    return None


def iscx_vpn(root: str | Path, scheme: str = "category",
             vpn_aware: bool = True) -> Corpus:
    """ISCX VPN-nonVPN 2016.

    ``scheme="category"`` gives the 6-class traffic-category scheme;
    ``scheme="app"`` labels by application. ``vpn_aware`` splits each class
    into tunnelled and non-tunnelled, which is the 12- or 14-class setting
    most of the reviewed papers actually report.

    The application scheme is heuristic and cannot be fully correct, because
    ISCX's filenames are not systematic: some encode a transfer direction
    (``ftps_down_1a`` against ``vpn_ftps_A``) that others omit, so the two
    land in different classes. ``build_store.py`` prints the resulting class
    map before it preprocesses anything -- read it before committing to a
    long run, and prefer the category scheme unless the finer labels are
    genuinely needed.
    """
    captures = find_captures(root)
    if not captures:
        raise FileNotFoundError(f"no capture files under {root}")

    def base_label(path: Path) -> str:
        stem = path.stem
        if scheme == "category":
            cat = _iscx_category(stem)
            if cat is None:
                raise KeyError(f"cannot map {path.name!r} to an ISCX traffic category")
            return cat
        # Per-application: strip the vpn_ prefix and the trailing capture
        # index. ISCX indexes captures inconsistently -- aim_chat_3a,
        # vpn_aim_chat1a, netflix1, vpn_netflix_A -- so the suffix is peeled
        # repeatedly rather than once.
        name = re.sub(r"^vpn[_-]", "", stem, flags=re.I).lower()
        for _ in range(3):
            stripped = re.sub(r"[_-]?\d+[a-z]?$", "", name)
            stripped = re.sub(r"[_-][a-z]$", "", stripped)
            if stripped == name or not stripped:
                break
            name = stripped
        return name

    def full_label(path: Path) -> str:
        label = base_label(path)
        if vpn_aware and re.match(r"^vpn[_-]", path.stem, flags=re.I):
            return f"vpn_{label}"
        return label

    label_names = sorted({full_label(p) for p in captures})
    return Corpus(
        name=f"ISCX-VPN-nonVPN ({scheme}{', vpn-aware' if vpn_aware else ''})",
        captures=[(p, p.stem) for p in captures],
        label_names=label_names,
        label_of=lambda path, _group: label_names.index(full_label(path)),
        notes="several captures per class, so the split is group-wise by capture file",
    )


# ------------------------------------------------------------------ integrity

def checksum(path: str | Path, algorithm: str = "sha256", chunk: int = 1 << 20) -> str:
    h = hashlib.new(algorithm)
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def manifest(captures: Iterable, algorithm: str = "sha256") -> dict:
    """Checksum every capture, so a corpus can be verified before use (Step 2)."""
    out = {}
    for path, group in captures:
        path = Path(path)
        out[str(path)] = {"group": group, "bytes": path.stat().st_size,
                          algorithm: checksum(path, algorithm)}
    return out


REGISTRY = {
    "ustc": ustc_tfc2016,
    "ustc-binary": lambda root: ustc_tfc2016(root, binary=True),
    "iscx": iscx_vpn,
    "iscx-app": lambda root: iscx_vpn(root, scheme="app"),
}


def load_corpus(kind: str, root: str | Path) -> Corpus:
    if kind not in REGISTRY:
        raise KeyError(f"unknown corpus {kind!r}; have {sorted(REGISTRY)}")
    return REGISTRY[kind](root)
