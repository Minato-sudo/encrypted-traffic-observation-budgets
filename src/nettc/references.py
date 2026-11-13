"""Full bibliographic entries for everything this study cites.

The rest of the project refers to the reviewed studies by the short labels P1
to P12, because a label that carries its own number is easier to check against
the Phase 1 and Phase 2 comparison tables than an author-year string. That is
fine inside the project and useless to a reader, who cannot look any of them
up. This module is the mapping back: one entry per label, transcribed from the
"Full Citations" blocks of the Phase 1 and Phase 2 submissions, so the three
documents cite the same papers in the same words.

:func:`render` emits the reference list grouped by kind. :func:`missing` takes
a rendered document and names any label it cites without an entry here, which
is what keeps the list from falling behind the prose: the label is written by
hand in the prose but the entry is not, so a new citation shows up as a
missing-reference error rather than as a silent omission.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Reference:
    label: str      # the short label used in the prose, e.g. "P4"
    authors: str
    year: int
    title: str
    venue: str      # journal or proceedings, with volume and pages
    doi: str = ""
    note: str = ""  # anything a reader needs in order to find it


# The twelve reviewed studies, in the order Phase 1 and Phase 2 introduced
# them. P1 to P3 come from the Phase 1 "Full Citations" block, P4 to P12 from
# the Phase 2 "Full Citations (Extended Set)" block.
REVIEWED: tuple[Reference, ...] = (
    Reference(
        "P1", "Zhai, J., Shen, L., Wu, S., Chen, K., & Liu, G.", 2026,
        "S²F-GNN: A self-supervised robust encrypted traffic classification "
        "method based on dynamic heterogeneous graphs",
        "Digital Communications and Networks",
        "10.1016/j.dcan.2026.05.002", "accepted 12 May 2026"),
    Reference(
        "P2", "Gu, J., Zhong, Y., & Yu, X.", 2026,
        "Bridging packet and session: Cross-level dual-attention networks for "
        "encrypted traffic classification",
        "Journal of King Saud University, Computer and Information "
        "Sciences, 38:79",
        "10.1007/s44443-026-00470-7", "published online 20 January 2026"),
    Reference(
        "P3", "Yang, Q., Chen, M., Zhang, B., Han, H., He, W., Wu, F., & Ji, Y.",
        2026,
        "Adaptive encrypted traffic classification via online hash center "
        "evolution",
        "Journal of King Saud University, Computer and Information "
        "Sciences, 38:558",
        "10.1007/s44443-026-00949-3", "published online 23 June 2026"),
    Reference(
        "P4", "Zhang, H., Yue, H., Xiao, X., Yu, L., Li, Q., Ling, Z., & "
        "Zhang, Y.", 2025,
        "Revolutionizing encrypted traffic classification with MH-Net: A "
        "multi-view heterogeneous graph model",
        "Proceedings of the AAAI Conference on Artificial Intelligence, "
        "39(1), 1048-1056",
        "10.1609/aaai.v39i1.32091"),
    Reference(
        "P5", "Wickramasinghe, N., Shaghaghi, A., Tsudik, G., & Jha, S.", 2025,
        "SoK: Decoding the enigma of encrypted network traffic classifiers",
        "IEEE Symposium on Security and Privacy (SP), 1825-1843"),
    Reference(
        "P6", "Zhou, G., Guo, X., Liu, Z., Li, T., Li, Q., & Xu, K.", 2025,
        "TrafficFormer: An efficient pre-trained model for traffic data",
        "IEEE Symposium on Security and Privacy (SP), 1844-1860"),
    Reference(
        "P7", "Park, J.-T., Choi, Y.-S., Cho, B.-S., Kim, S.-H., & Kim, M.-S.",
        2025,
        "Multi-level pre-training for encrypted network traffic classification",
        "IEEE Access, 13, 68643-68659",
        "10.1109/ACCESS.2025.3559068"),
    Reference(
        "P8", "Huang, H., Zhou, Y., Jiang, F., Zhou, X., & Jiang, Q.", 2025,
        "MFF: A multimodal feature fusion approach for encrypted traffic "
        "classification",
        "Electronics, 14(13), 2584", "10.3390/electronics14132584"),
    Reference(
        "P9", "Liu, Z., Wei, Q., Song, Q., & Duan, C.", 2025,
        "Fine-grained encrypted traffic classification using dual embedding "
        "and graph neural networks",
        "Electronics, 14(4), 778", "10.3390/electronics14040778"),
    Reference(
        "P10", "Liu, Z., Xie, Y., Luo, Y., Wang, Y., & Ji, X.", 2025,
        "TransECA-Net: A transformer-based model for encrypted traffic "
        "classification",
        "Applied Sciences, 15(6), 2977", "10.3390/app15062977"),
    Reference(
        "P11", "Li, Z., Wang, J., Song, Y.-F., & Yue, S.-H.", 2025,
        "Unlocking few-shot encrypted traffic classification: A "
        "contrastive-driven meta-learning approach",
        "Electronics, 14(21), 4245", "10.3390/electronics14214245"),
    Reference(
        "P12", "Elshewey, A. M., & Osman, A. M.", 2025,
        "Enhancing encrypted HTTPS traffic classification based on stacked "
        "deep ensembles models",
        "Scientific Reports, 15, 35230", "10.1038/s41598-025-21261-6"),
)


# The two corpora. Neither Phase 1 nor Phase 2 cited the papers that released
# them, which is an omission worth repairing here: every number in this study
# is a number about these two releases specifically, and the release matters
# (the ISCX distribution used here is missing the non-VPN p2p captures).
CORPORA: tuple[Reference, ...] = (
    Reference(
        "USTC-TFC2016", "Wang, W., Zhu, M., Zeng, X., Ye, X., & Sheng, Y.", 2017,
        "Malware traffic classification using convolutional neural network "
        "for representation learning",
        "International Conference on Information Networking (ICOIN), "
        "712-717",
        "10.1109/ICOIN.2017.7899588",
        "the paper that assembled the corpus: 10 benign classes captured at "
        "USTC and 10 malware classes from the CTU-13 traces. Obtained for this "
        "study from github.com/yungshenglu/USTC-TFC2016"),
    Reference(
        "ISCX VPN-nonVPN",
        "Draper-Gil, G., Lashkari, A. H., Mamun, M. S. I., & Ghorbani, A. A.",
        2016,
        "Characterization of encrypted and VPN traffic using time-related "
        "features",
        "Proceedings of the 2nd International Conference on Information "
        "Systems Security and Privacy (ICISSP), 407-414",
        "10.5220/0005740704070414",
        "distributed by the Canadian Institute for Cybersecurity as "
        "ISCXVPN2016"),
)


# Methods and tooling. Only what the study actually depends on: the pcap
# parser, the session assembler and every metric in this project are written
# against the raw formats rather than taken from a library, so there is no
# scapy or scikit-learn entry to make.
METHODS: tuple[Reference, ...] = (
    Reference(
        "McNemar", "McNemar, Q.", 1947,
        "Note on the sampling error of the difference between correlated "
        "proportions or percentages",
        "Psychometrika, 12(2), 153-157", "10.1007/BF02295996",
        "the paired test used in §3; this study uses the exact binomial "
        "form of it rather than the chi-squared approximation"),
    Reference(
        "Dietterich", "Dietterich, T. G.", 1998,
        "Approximate statistical tests for comparing supervised "
        "classification learning algorithms",
        "Neural Computation, 10(7), 1895-1923",
        "10.1162/089976698300017197",
        "why a paired test on the same test set and a comparison of means "
        "across seeds answer different questions, which §3 relies on"),
    Reference(
        "van der Oord", "van den Oord, A., Dieleman, S., Zen, H., Simonyan, K., "
        "Vinyals, O., Graves, A., Kalchbrenner, N., Senior, A., & "
        "Kavukcuoglu, K.", 2016,
        "WaveNet: A generative model for raw audio",
        "arXiv:1609.03499", "10.48550/arXiv.1609.03499",
        "the causal dilated convolution stack the early-exit encoder is built "
        "from; causality is what makes a prefix-budget head honest"),
    Reference(
        "PyTorch", "Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., "
        "Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., "
        "Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., "
        "Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., & "
        "Chintala, S.", 2019,
        "PyTorch: An imperative style, high-performance deep learning library",
        "Advances in Neural Information Processing Systems 32, "
        "8024-8035",
        note="version 2.11.0+cu130 in the runs reported here"),
    Reference(
        "HDF5", "The HDF Group", 2024,
        "Hierarchical Data Format, version 5",
        "hdfgroup.org/HDF5", note="via h5py 3.16.0; the session store format"),
)

GROUPS: tuple[tuple[str, tuple[Reference, ...]], ...] = (
    ("Reviewed studies", REVIEWED),
    ("Corpora", CORPORA),
    ("Methods and tooling", METHODS),
)


def _one(r: Reference) -> str:
    bits = [f"**[{r.label}]** {r.authors} ({r.year}). {r.title}. *{r.venue}*."]
    if r.doi:
        bits.append(f" DOI: {r.doi}.")
    if r.note:
        bits.append(f" {r.note}.")
    return "".join(bits)


def render(heading: str = "## References") -> list[str]:
    """The reference list, as Markdown lines."""
    out = [heading, "",
           "The reviewed studies are cited in the text by the labels P1 to "
           "P12, carried over unchanged from the Phase 1 and Phase 2 "
           "submissions so that the three documents refer to the same papers "
           "by the same names.", ""]
    for title, group in GROUPS:
        out += [f"### {title}", ""]
        for r in group:
            out += [_one(r), ""]
    return out


def missing(document: str) -> list[str]:
    """Labels the document cites that have no entry here.

    Guards against the prose and the reference list drifting apart. Only the
    P-labels are checked, because those are the ones written by hand.
    """
    known = {r.label for r in REVIEWED}
    cited = set(re.findall(r"\bP(?:1[0-2]|[1-9])\b", document))
    return sorted(cited - known, key=lambda s: int(s[1:]))


def unused(document: str) -> list[str]:
    """Entries in the reviewed set the document never cites."""
    cited = set(re.findall(r"\bP(?:1[0-2]|[1-9])\b", document))
    return [r.label for r in REVIEWED if r.label not in cited]
