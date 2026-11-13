"""Check the distributed archives against what was extracted and labelled.

A class can be absent from a corpus for two very different reasons: the
download went wrong, or the release does not contain it. Those call for
opposite responses -- fetch it again, or report the corpus as incomplete --
and the class list alone cannot tell them apart. This script settles it by
reading the archive directories themselves, so the limitation the reports
carry is a verified statement about the distribution rather than an inference
from what happened to be parsed.

For each archive it records the capture files it declares; it then compares
that set against the files on disk, and searches both for the categories the
label taxonomy expects. Writes ``results/<corpus>/archive_audit.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nettc.metrics import save_json

SUFFIXES = (".pcap", ".pcapng")

# How a capture file name betrays its category. Only the categories that have
# ever been reported missing need an entry; the rest are resolved by the
# dataset adapter, which is the authority on labelling.
CATEGORY_HINTS = {
    "p2p": r"torrent|p2p",
    "chat": r"chat|aim|icq|hangout",
    "email": r"email|gmail",
    "file": r"ftps|sftp|scp|_file",
    "streaming": r"netflix|spotify|vimeo|youtube",
    "voip": r"audio|voipbuster|video",
}


def archive_members(path: Path) -> list[str]:
    """Capture file names an archive declares, without extracting it."""
    try:
        out = subprocess.run(["unzip", "-Z1", str(path)],
                             capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired):
        return []
    names = []
    for line in out.stdout.splitlines():
        line = line.strip()
        if line and not line.endswith("/"):
            base = line.split("/")[-1]
            if base.lower().endswith(SUFFIXES):
                names.append(base)
    return names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archives", required=True,
                    help="directory holding the distributed .zip archives")
    ap.add_argument("--extracted", required=True,
                    help="directory the archives were extracted into")
    ap.add_argument("--out", required=True, help="results directory")
    ap.add_argument("--corpus", default=None)
    args = ap.parse_args()

    adir, edir = Path(args.archives), Path(args.extracted)
    archives = sorted(p for p in adir.glob("*.zip"))
    declared: dict[str, list[str]] = {a.name: archive_members(a) for a in archives}
    all_declared = {n for v in declared.values() for n in v}
    on_disk = {p.name for p in edir.rglob("*") if p.suffix.lower() in SUFFIXES}

    hits = {}
    for cat, pattern in CATEGORY_HINTS.items():
        rx = re.compile(pattern, re.I)
        found = sorted(n for n in all_declared if rx.search(n))
        hits[cat] = {
            "in_archives": found,
            "vpn_only": bool(found) and all(n.lower().startswith("vpn") for n in found),
        }

    audit = {
        "corpus": args.corpus or edir.name,
        "archives": {k: len(v) for k, v in sorted(declared.items())},
        "n_archives": len(archives),
        "n_declared": len(all_declared),
        "n_extracted": len(on_disk),
        "declared_but_not_extracted": sorted(all_declared - on_disk),
        "extracted_but_in_no_archive": sorted(on_disk - all_declared),
        "extraction_complete": not (all_declared - on_disk),
        "categories": hits,
    }
    save_json(audit, Path(args.out) / "archive_audit.json")

    print(f"{len(archives)} archives declare {len(all_declared)} capture files; "
          f"{len(on_disk)} are extracted")
    print(f"extraction complete: {audit['extraction_complete']}")
    for cat, cell in hits.items():
        n = len(cell["in_archives"])
        note = "  (VPN side only)" if cell["vpn_only"] else ""
        print(f"  {cat:<10} {n:>3} capture(s){note}")
    print(f"\nwrote {Path(args.out) / 'archive_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
