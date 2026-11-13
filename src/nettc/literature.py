"""Published results from the twelve reviewed studies, for Step 12's comparison.

Figures are transcribed from the Phase 1 and Phase 2 comparison tables, which
were read from the full texts. Each entry carries the *settings* alongside the
number, because that is what decides whether a comparison is meaningful: the
observation budget, the label scheme, the preprocessing, and whether the paper
reports a metric we also report.

The honest position, which this module encodes rather than hides, is that
almost none of these numbers are directly comparable to ours. They differ in
class scheme (6 against 12 against 20), in preprocessing (P2 discards DNS and
SSDP, collapsing ISCX-VPN from 13,520 sessions to 4,511), in split policy, and
in whether identifier fields were removed at all -- P5 measured that removing
them costs ET-BERT 0.45 accuracy, so a paper that kept them is reporting a
different quantity. :func:`comparability_notes` states the obstacle for each
entry, and the comparison table prints it next to the number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class PublishedResult:
    paper: str
    year: int
    venue: str
    corpus: str
    classes: Optional[int]
    accuracy: Optional[float]
    macro_f1: Optional[float]
    budget: str                  # the observation budget, in the paper's own terms
    budget_bytes: Optional[int]  # our best estimate in bytes, where one exists
    budget_packets: Optional[int]
    anonymised_identifiers: Optional[bool]
    note: str = ""


# Ordered as in the Phase 2 extended review.
RESULTS: tuple[PublishedResult, ...] = (
    PublishedResult("P1 S2F-GNN", 2026, "Digital Communications and Networks (Q1)",
                    "ISCX-VPN", None, 0.9326, None, "20 packets, 10 x 0.1 s windows",
                    None, 20, None,
                    "accuracy at 20% labels; the headline claim is few-shot, not full-label"),
    PublishedResult("P1 S2F-GNN", 2026, "Digital Communications and Networks (Q1)",
                    "ISCX-VPN", None, None, 0.8754, "20 packets, 5% labels",
                    None, 20, None, "F1 at 5% labels"),
    PublishedResult("P2 CLET", 2026, "J. King Saud Univ. CIS (Q1)",
                    "ISCX-VPN", 14, 0.8861, 0.8280, "16 packets, header descriptor",
                    None, 16, None,
                    "DNS and SSDP removed, leaving 4,511 of 13,520 sessions; "
                    "AIM 0% and SFTP 50% inside this macro-F1"),
    PublishedResult("P3 OHCE", 2026, "J. King Saud Univ. CIS (Q1)",
                    "USTC-TFC2016", 20, 0.9438, None, "5 packets",
                    None, 5, True,
                    "open-set setting; flows under 5 packets discarded entirely"),
    PublishedResult("P4 MH-Net", 2025, "AAAI (A*)",
                    "ISCX-VPN", 6, 0.9942, 0.9941, "15 packets, flow level",
                    None, 15, True,
                    "flow level; removing the 8-bit view collapses F1 to 0.1702"),
    PublishedResult("P4 MH-Net", 2025, "AAAI (A*)",
                    "ISCX-VPN", 6, 0.9768, 0.9766, "15 packets, packet level",
                    None, 15, True, "packet level"),
    PublishedResult("P5 SoK (ET-BERT)", 2025, "IEEE S&P (A*)",
                    "ISCX-VPN", None, 0.96, None, "varies by occlusion",
                    None, None, False,
                    "0.96 with identifier fields present, 0.51 with them removed"),
    PublishedResult("P5 SoK (YaTC)", 2025, "IEEE S&P (A*)",
                    "ISCX-VPN", None, 0.90, None, "varies by occlusion",
                    None, None, False,
                    "0.90 with identifiers, 0.62 without; encrypted payload alone 0.30"),
    PublishedResult("P7 MLETC", 2025, "IEEE Access (Q1)",
                    "ISCX-VPN", 6, 0.9875, 0.9880, "5 packets x 100 bytes",
                    500, 5, True,
                    "budget chosen to fit a 512-token ceiling, by the authors' own account"),
    PublishedResult("P7 MLETC", 2025, "IEEE Access (Q1)",
                    "USTC-TFC2016", 20, 0.9902, 0.9926, "5 packets x 100 bytes",
                    500, 5, True, ""),
    PublishedResult("P8 MFF", 2025, "Electronics (Q2)",
                    "ISCX-VPN", 12, 0.9961, None, "1024 bytes", 1024, None, True,
                    "about 87% of ISCX-VPN sessions fall below this budget and are padded"),
    PublishedResult("P8 MFF", 2025, "Electronics (Q2)",
                    "USTC-TFC2016", 20, 0.9999, None, "1024 bytes", 1024, None, True,
                    "about 70% of USTC-TFC sessions are padded at this budget"),
    PublishedResult("P9 Dual Embedding + GNN", 2025, "Electronics (Q2)",
                    "ISCX-VPN", None, 0.967, 0.960, "fixed packet count",
                    None, None, False,
                    "retains source and destination IP addresses and ports as inputs"),
    PublishedResult("P10 TransECA-Net", 2025, "Applied Sciences (Q2)",
                    "ISCX-VPN", 12, 0.9765, None, "784 bytes", 784, None, True,
                    "per-class far weaker than the headline: File 93.56%, VoIP 92.20%"),
    PublishedResult("P10 TransECA-Net", 2025, "Applied Sciences (Q2)",
                    "USTC-TFC2016", 10, 0.9825, None, "784 bytes", 784, None, True,
                    "ten benign classes only, used for generalisation"),
    PublishedResult("P11 CL-MetaFlow", 2025, "Electronics (Q2)",
                    "ISCX-VPN + ISCX-Tor", 17, None, 0.620, "no payload used",
                    0, None, True,
                    "5-way 5-shot; far lower than closed-world figures, by design"),
    PublishedResult("P12 Stacked ensembles", 2025, "Scientific Reports (Q1)",
                    "ISCX-VPN", 6, 0.9930, None, "88 flow-level features",
                    None, None, None,
                    "flow statistics, not raw bytes; fitting the scaler on all data "
                    "inflates accuracy from 0.9949 to 0.9960"),
)


def for_corpus(corpus: str) -> list:
    key = corpus.lower().replace("-", "").replace("_", "").replace(" ", "")
    return [r for r in RESULTS
            if key in r.corpus.lower().replace("-", "").replace("_", "").replace(" ", "")]


def comparability_notes(result: PublishedResult, our_classes: int) -> list:
    """Why this number is or is not comparable to ours. Stated, not buried."""
    obstacles = []
    if result.classes is not None and result.classes != our_classes:
        obstacles.append(f"different class scheme ({result.classes} against {our_classes})")
    if result.anonymised_identifiers is False:
        obstacles.append("identifier fields retained, which P5 shows is worth up to "
                         "0.45 accuracy on its own")
    if result.anonymised_identifiers is None:
        obstacles.append("does not state whether identifier fields were removed")
    if "removed" in result.note or "discards" in result.note or "discarded" in result.note:
        obstacles.append("different session filtering")
    if result.macro_f1 is None:
        obstacles.append("reports accuracy but not macro-F1")
    return obstacles


def is_comparable(result: PublishedResult, our_classes: int) -> bool:
    """True when nothing structural stands in the way of a direct comparison."""
    return not comparability_notes(result, our_classes)


def comparison_table(ours: dict, corpus: str, our_classes: int,
                     our_label: str = "this work") -> str:
    """A table placing our result beside the published ones, with the caveats.

    ``ours`` is a cell from the budget sweep: it needs ``macro_f1``,
    ``accuracy`` and a budget label.
    """
    def trim(text: str, width: int) -> str:
        return text if len(text) <= width else text[: width - 1] + "\u2026"

    rows = [
        f"Comparison on {corpus}",
        "",
        f"{'method':<24}{'year':>5}  {'budget':<26}{'acc':>8}{'macroF1':>9}",
        "-" * 74,
    ]
    for r in for_corpus(corpus):
        acc = f"{r.accuracy:.4f}" if r.accuracy is not None else "--"
        mf1 = f"{r.macro_f1:.4f}" if r.macro_f1 is not None else "--"
        rows.append(f"{trim(r.paper, 24):<24}{r.year:>5}  {trim(r.budget, 26):<26}"
                    f"{acc:>8}{mf1:>9}")
        obstacles = comparability_notes(r, our_classes)
        if obstacles:
            for obstacle in obstacles:
                rows.append(f"{'':<6}not comparable: {obstacle}")
        else:
            rows.append(f"{'':<6}broadly comparable: budget, class scheme and "
                        f"anonymisation align; preprocessing and splits still differ")
        if r.note:
            rows.append(f"{'':<6}{r.note}")
        rows.append("")
    rows.append("-" * 74)
    acc = ours.get("accuracy", {}).get("mean")
    mf1 = ours.get("macro_f1", {}).get("mean")
    rows.append(f"{trim(our_label, 24):<24}{2026:>5}  {trim(str(ours.get('budget', '?')), 26):<26}"
                f"{'--' if acc is None else f'{acc:.4f}':>8}"
                f"{'--' if mf1 is None else f'{mf1:.4f}':>9}")
    rows.append(f"{'':<6}identifiers removed, and the removal verified by a positive control")
    rows.append("")
    rows.append("Read with care. These figures were produced under different "
                "preprocessing, different class schemes and different splits, and")
    rows.append("several of the papers do not say whether identifier fields were "
                "removed. P5 measured that removing them costs ET-BERT 0.45")
    rows.append("accuracy, so a retained-identifier number and an anonymised number "
                "are not measuring the same task. The budgets, by contrast,")
    rows.append("are directly comparable, and locating them on our own sweep is the "
                "point of the exercise.")
    return "\n".join(rows)


def budget_marks(axis: str = "packets") -> dict:
    """The reviewed budgets, for annotating the budget profile."""
    out: dict = {}
    for r in RESULTS:
        value = r.budget_packets if axis == "packets" else r.budget_bytes
        if not value:
            continue
        out.setdefault(value, r.paper.split()[0])
    return out
