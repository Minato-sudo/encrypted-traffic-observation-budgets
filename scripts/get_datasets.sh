#!/usr/bin/env bash
# Fetch and verify the two corpora of Phase 2 Step 2.
#
# This script is deliberately not run as part of the build. It downloads tens
# of gigabytes and one of the two corpora requires you to accept the
# provider's terms yourself, which is not something a script should do on
# your behalf. Read it, then run the parts you want.
#
#   bash scripts/get_datasets.sh check            # disk + what is already here
#   bash scripts/get_datasets.sh ustc             # primary corpus (~3.7 GB)
#   bash scripts/get_datasets.sh iscx             # instructions for the secondary corpus
#   bash scripts/get_datasets.sh checksums ustc   # record SHA-256 for every capture
#
set -euo pipefail

DATA_DIR="${DATA_DIR:-$(cd "$(dirname "$0")/.." && pwd)/data}"
USTC_DIR="$DATA_DIR/USTC-TFC2016"
ISCX_DIR="$DATA_DIR/ISCX-VPN-nonVPN"

# USTC-TFC2016 is distributed as a public git repository, part .pcap and
# part .7z. Verified complete (20/20 classes) on 2026-10-22.
# ISCX VPN-nonVPN 2016 is distributed by the Canadian Institute for
# Cybersecurity behind a terms-of-use page and has no stable direct link.
USTC_REPO="${USTC_REPO:-https://github.com/yungshenglu/USTC-TFC2016}"
ISCX_PAGE="https://www.unb.ca/cic/datasets/vpn.html"

say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
warn() { printf '  ! %s\n' "$*" >&2; }

cmd_check() {
    say "disk"
    df -h "$DATA_DIR" 2>/dev/null || df -h "$(dirname "$DATA_DIR")"
    local avail_gb
    avail_gb=$(df -BG --output=avail "$(dirname "$DATA_DIR")" | tail -1 | tr -dc '0-9')
    echo
    echo "  USTC-TFC2016 needs about 4 GB of captures, plus roughly 2-4 GB for its store."
    echo "  ISCX VPN-nonVPN needs about 26 GB of captures, plus a store of its own."
    if [ "$avail_gb" -lt 40 ]; then
        warn "only ${avail_gb} GB free. That fits USTC comfortably but is tight for both."
        warn "Consider building the ISCX store and then deleting its raw captures:"
        warn "  the store is self-contained, so the pcaps are not needed again."
    fi

    say "what is already here"
    for dir in "$USTC_DIR" "$ISCX_DIR"; do
        if [ -d "$dir" ]; then
            local n sz
            n=$(find "$dir" -type f \( -name '*.pcap' -o -name '*.pcapng' -o -name '*.cap' \) | wc -l)
            sz=$(du -sh "$dir" 2>/dev/null | cut -f1)
            printf '  %-46s %4s capture files, %s\n' "$dir" "$n" "$sz"
        else
            printf '  %-46s absent\n' "$dir"
        fi
    done
}

cmd_ustc() {
    say "USTC-TFC2016 -> $USTC_DIR"
    echo "  source: $USTC_REPO"
    echo "  Verified 2026-10-22: 20 of 20 classes present (10 benign, 10 malware),"
    echo "  0.39 GB in the repository, expanding to roughly 3.7 GB. Eleven classes"
    echo "  are stored as .pcap and nine as .7z, so the archives are expanded below."
    echo "  Credited upstream to echowei/DeepTraffic (Wei Wang), MPL-2.0."

    command -v 7z >/dev/null 2>&1 || { warn "7z not found; install p7zip-full"; exit 1; }
    mkdir -p "$DATA_DIR"
    if [ -d "$USTC_DIR/.git" ]; then
        echo "  already cloned; pulling"
        git -C "$USTC_DIR" pull --ff-only
    else
        git clone --depth 1 "$USTC_REPO" "$USTC_DIR"
    fi

    say "expanding archives"
    # Expand in place, next to the .pcap files. The corpus adapter globs
    # recursively and maps a class from the filename, so Benign/ and Malware/
    # staying as separate subtrees is fine.
    find "$USTC_DIR" -name '*.7z' -print0 | while IFS= read -r -d '' archive; do
        local dir base
        dir=$(dirname "$archive")
        base=$(basename "$archive" .7z)
        if find "$dir" -maxdepth 1 -iname "$base.pcap" | grep -q .; then
            echo "  $base already expanded"
            continue
        fi
        echo "  expanding $(basename "$archive")"
        7z x -y -o"$dir" "$archive" >/dev/null || warn "failed to expand $archive"
    done

    local n sz
    n=$(find "$USTC_DIR" -type f \( -iname '*.pcap' -o -iname '*.pcapng' \) | wc -l)
    sz=$(du -sh "$USTC_DIR" 2>/dev/null | cut -f1)
    echo
    echo "  $n capture files, $sz on disk"
    if [ "$n" -lt 20 ]; then
        warn "expected 20 captures (10 malware + 10 benign) but found $n."
        warn "Check which classes are missing before building a store:"
        warn "  find $USTC_DIR -iname '*.pcap' -printf '%f\\n' | sort"
    else
        echo "  all 20 classes present"
    fi
    echo
    echo "  next:"
    echo "    bash $0 checksums ustc"
    echo "    PYTHONPATH=src python3 scripts/build_store.py --corpus ustc \\"
    echo "        --root \"$USTC_DIR\" --out stores/ustc.h5 --checksums"
}

cmd_iscx() {
    say "ISCX VPN-nonVPN 2016 -> $ISCX_DIR"
    cat <<TXT
  This one has to be fetched by hand. The Canadian Institute for
  Cybersecurity distributes it behind a terms-of-use page:

      $ISCX_PAGE

  Accept their terms, download the pcap archives, and unpack them into:

      $ISCX_DIR

  Keep the original filenames. The corpus adapter reads the class and the
  VPN flag out of them (vpn_youtube_A.pcap, skype_chat1a.pcap, and so on),
  so renaming them will silently change your labels.

  Then:
      python3 scripts/build_store.py --corpus iscx --root "$ISCX_DIR" \\
          --out stores/iscx.h5 --checksums
TXT
    mkdir -p "$ISCX_DIR"
}

cmd_checksums() {
    local which="${1:-ustc}"
    local dir="$USTC_DIR"
    [ "$which" = "iscx" ] && dir="$ISCX_DIR"
    say "SHA-256 for every capture under $dir"
    find "$dir" -type f \( -name '*.pcap' -o -name '*.pcapng' -o -name '*.cap' \) \
        -print0 | sort -z | xargs -0 sha256sum | tee "$DATA_DIR/${which}.sha256"
    echo "  wrote $DATA_DIR/${which}.sha256"
}

case "${1:-check}" in
    check)     cmd_check ;;
    ustc)      cmd_ustc ;;
    iscx)      cmd_iscx ;;
    checksums) shift; cmd_checksums "$@" ;;
    *) echo "usage: $0 {check|ustc|iscx|checksums [ustc|iscx]}" >&2; exit 2 ;;
esac
