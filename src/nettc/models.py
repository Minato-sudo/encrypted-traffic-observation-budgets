"""Models: the reproduced baselines (Step 6) and the adaptive classifier (Step 5).

The baselines exist to anchor the study to published work and to check the
preprocessing pipeline, not to compete. The contribution is
:class:`AdaptiveByteCNN`, whose convolutions are *causal* along the byte
axis: the output at position i depends only on bytes 0..i. That property is
what makes the early-exit heads honest. A head placed at byte 128 computes
exactly the same logits whether it is fed 128 bytes or 2048, so a single
forward pass trains every budget simultaneously and the reported
accuracy-per-byte frontier reflects bytes genuinely read rather than bytes
read and then masked.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .slicer import PAD_SYMBOL

DEFAULT_EXITS: tuple[int, ...] = (64, 128, 256, 512, 1024)


class ByteFrontEnd(nn.Module):
    """Turns a batch of raw byte rows into (B, C, L) features.

    Four input modes, so the padding controls of Step 11ii are separable:

    * ``zero``     -- bytes scaled to [0, 1] on one channel, padding is 0x00.
                      What the byte-image and 1D-CNN baselines do.
    * ``convmask`` -- the same scalar input, but the model is handed the valid
                      length so pooling can exclude padded positions.
    * ``embed``    -- a learned per-byte embedding, padding still 0x00 and
                      therefore indistinguishable from a real zero byte.
    * ``mask``     -- a learned embedding *and* a dedicated PAD symbol, so
                      padding is distinguishable from a real zero byte.
    """

    def __init__(self, pad_mode: str = "zero", embed_dim: int = 16):
        super().__init__()
        self.pad_mode = pad_mode
        # "embed" exists so that "mask" can be attributed. Moving from zero to
        # mask changes two things at once -- a scalar input becomes a learned
        # embedding, *and* padding gains its own symbol -- and the smoke run
        # showed that confound is worth more than half a point of macro-F1.
        # "embed" holds the representation change and drops the padding
        # change, so embed-minus-zero isolates the representation and
        # mask-minus-embed isolates the padding symbol.
        if pad_mode in ("mask", "embed"):
            self.embed = nn.Embedding(PAD_SYMBOL + 1, embed_dim, padding_idx=None)
            self.out_channels = embed_dim
        else:
            self.embed = None
            self.out_channels = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.embed is not None:
            return self.embed(x.long()).transpose(1, 2)
        return (x.float() / 255.0).unsqueeze(1)


def _valid_mask(valid_len: Optional[torch.Tensor], length: int,
                device, dtype=torch.float32) -> Optional[torch.Tensor]:
    """(B, 1, L) mask of positions holding a real byte."""
    if valid_len is None:
        return None
    pos = torch.arange(length, device=device)[None, :]
    return (pos < valid_len.clamp(max=length)[:, None]).to(dtype).unsqueeze(1)


def _masked_pool(h: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Concatenated masked mean and max pooling over the length axis."""
    if mask is None:
        return torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
    denom = mask.sum(dim=2).clamp(min=1.0)
    mean = (h * mask).sum(dim=2) / denom
    very_negative = torch.finfo(h.dtype).min
    mx = h.masked_fill(mask == 0, very_negative).amax(dim=2)
    # A row with no valid bytes at all would otherwise pool to -inf.
    mx = torch.where(mask.sum(dim=2) > 0, mx, torch.zeros_like(mx))
    return torch.cat([mean, mx], dim=1)


class Wang2DCNN(nn.Module):
    """Byte-image CNN after Wang et al., ICOIN 2017.

    Retained as the reproducible reference point and as a check on the
    preprocessing pipeline (Step 6i), not as a contribution: the 2026
    literature treats this representation as a baseline.
    """

    def __init__(self, n_classes: int, n_bytes: int = 784, pad_mode: str = "zero",
                 side: int = 28):
        super().__init__()
        if side * side != n_bytes:
            raise ValueError(f"the 2D form needs a square budget; {n_bytes} is not {side}^2")
        self.side = side
        self.front = ByteFrontEnd(pad_mode)
        c = self.front.out_channels
        self.features = nn.Sequential(
            nn.Conv2d(c, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 1024), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(1024, n_classes),
        )

    def forward(self, x, valid_len: Optional[torch.Tensor] = None):
        h = self.front(x)                               # (B, C, L)
        b, c, _ = h.shape
        h = h.reshape(b, c, self.side, self.side)
        return self.classifier(self.features(h))


class Wang1DCNN(nn.Module):
    """1D byte-sequence CNN after Wang et al., ISI 2017 (Step 6ii).

    Global pooling rather than a flatten keeps the parameter count identical
    at every budget, so a difference between budgets cannot be explained by
    one variant simply having more capacity.
    """

    def __init__(self, n_classes: int, pad_mode: str = "zero"):
        super().__init__()
        self.front = ByteFrontEnd(pad_mode)
        c = self.front.out_channels
        self.conv1 = nn.Conv1d(c, 32, kernel_size=25, padding=12)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=25, padding=12)
        self.head = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n_classes))

    def forward(self, x, valid_len: Optional[torch.Tensor] = None):
        h = self.front(x)
        mask = _valid_mask(valid_len, h.shape[2], h.device, h.dtype)
        h = F.relu(self.conv1(h))
        h = F.max_pool1d(h, 3, ceil_mode=True)
        h = F.relu(self.conv2(h))
        h = F.max_pool1d(h, 3, ceil_mode=True)
        if mask is not None:
            mask = F.max_pool1d(F.max_pool1d(mask, 3, ceil_mode=True), 3, ceil_mode=True)
            mask = mask[:, :, :h.shape[2]]
        return self.head(_masked_pool(h, mask))


class DeepPacket(nn.Module):
    """CNN after Deep Packet (Lotfollahi et al., 2020), Step 6iii."""

    def __init__(self, n_classes: int, pad_mode: str = "zero"):
        super().__init__()
        self.front = ByteFrontEnd(pad_mode)
        c = self.front.out_channels
        self.conv1 = nn.Conv1d(c, 200, kernel_size=4, stride=3)
        self.conv2 = nn.Conv1d(200, 200, kernel_size=5, stride=1)
        self.head = nn.Sequential(
            nn.Linear(400, 200), nn.ReLU(), nn.Dropout(0.05),
            nn.Linear(200, 100), nn.ReLU(), nn.Dropout(0.05),
            nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, n_classes),
        )

    def forward(self, x, valid_len: Optional[torch.Tensor] = None):
        h = F.relu(self.conv1(self.front(x)))
        h = F.relu(self.conv2(h))
        return self.head(_masked_pool(h, None))


class ChannelNorm1d(nn.Module):
    """Normalise over channels at each position independently.

    Deliberately not GroupNorm or BatchNorm: both compute statistics across
    the length axis, which would let a byte at position 900 shift the
    activation at position 10 and silently destroy the causality the
    early-exit heads rely on. Normalising within a position cannot.
    """

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1))
        self.eps = eps

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        mean = h.mean(dim=1, keepdim=True)
        var = h.var(dim=1, keepdim=True, unbiased=False)
        return (h - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class CausalBlock(nn.Module):
    """Dilated causal residual block. No downsampling, so depth never mixes
    a later byte into an earlier position's output."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm1 = ChannelNorm1d(channels)
        self.norm2 = ChannelNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.norm1(self.conv1(F.pad(h, (self.left_pad, 0)))))
        y = self.drop(y)
        y = F.relu(self.norm2(self.conv2(F.pad(y, (self.left_pad, 0)))))
        return h + self.drop(y)


class AdaptiveByteCNN(nn.Module):
    """Shared causal encoder with an auxiliary classification head per budget.

    Step 5 of the methodology. Head k pools only positions [0, exits[k]) of a
    causal feature map, so its logits are a function of the first
    ``exits[k]`` bytes alone. At inference the smallest head is read first
    and deeper heads are consulted only while confidence is below the
    threshold; sweeping the threshold traces the accuracy-per-byte frontier.
    """

    def __init__(self, n_classes: int, exits: Sequence[int] = DEFAULT_EXITS,
                 channels: int = 96, kernel_size: int = 5,
                 dilations: Sequence[int] = (1, 2, 4, 8, 16, 32),
                 pad_mode: str = "zero", dropout: float = 0.1):
        super().__init__()
        self.exits = tuple(sorted(exits))
        self.front = ByteFrontEnd(pad_mode)
        self.stem = nn.Conv1d(self.front.out_channels, channels, 1)
        self.blocks = nn.ModuleList(
            [CausalBlock(channels, kernel_size, d, dropout) for d in dilations])
        self.heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(2 * channels, 128), nn.ReLU(),
                           nn.Dropout(dropout), nn.Linear(128, n_classes))
             for _ in self.exits])

    def reachable_exits(self, length: int) -> tuple:
        """The exits an input of ``length`` bytes can actually reach.

        ``forward`` stops at the first head whose budget covers the whole
        input, so a model built with deeper exits than the ceiling it is run
        at returns fewer logits than it has heads. Callers that index heads
        by position must ask here rather than assume ``self.exits``.
        """
        out = []
        for budget in self.exits:
            out.append(budget)
            if budget >= length:
                break
        return tuple(out)

    def encode(self, x, valid_len: Optional[torch.Tensor] = None):
        h = self.stem(self.front(x))
        for block in self.blocks:
            h = block(h)
        return h

    def forward(self, x, valid_len: Optional[torch.Tensor] = None) -> list[torch.Tensor]:
        """Logits from every head whose budget the input is long enough to reach."""
        h = self.encode(x, valid_len)
        length = h.shape[2]
        logits = []
        for budget, head in zip(self.exits, self.heads):
            cut = min(budget, length)
            region = h[:, :, :cut]
            mask = _valid_mask(valid_len, cut, h.device, h.dtype) if valid_len is not None else None
            logits.append(head(_masked_pool(region, mask)))
            if budget >= length:
                break
        return logits

    @torch.no_grad()
    def early_exit(self, x, valid_len: Optional[torch.Tensor], threshold: float):
        """Run the exit policy and report what it cost.

        Returns ``(prediction, bytes_read, exit_index)``. ``bytes_read`` is
        the budget of the head that actually decided, capped by the bytes the
        session really holds -- a 40-byte flow does not cost 64 bytes to read.
        """
        all_logits = self.forward(x, valid_len)
        n = x.shape[0]
        device = x.device
        decided = torch.zeros(n, dtype=torch.bool, device=device)
        pred = torch.zeros(n, dtype=torch.long, device=device)
        exit_idx = torch.full((n,), len(all_logits) - 1, dtype=torch.long, device=device)
        budget_used = torch.full((n,), float(self.exits[len(all_logits) - 1]), device=device)

        for k, logits in enumerate(all_logits):
            probs = F.softmax(logits.float(), dim=1)
            conf, hat = probs.max(dim=1)
            take = (~decided) & ((conf >= threshold) | (k == len(all_logits) - 1))
            pred[take] = hat[take]
            exit_idx[take] = k
            budget_used[take] = float(self.exits[k])
            decided |= take
            if bool(decided.all()):
                break

        if valid_len is not None:
            budget_used = torch.minimum(budget_used, valid_len.to(budget_used.dtype))
        return pred, budget_used, exit_idx


class ByteTransformer(nn.Module):
    """A byte-level transformer encoder, as an architecture-family control.

    The three reproduced baselines are convolutional and date from 2017-2020,
    while every study reviewed in Phase 1 is a transformer, a graph network or
    an ensemble from 2025-2026. That leaves an obvious question open: is the
    budget saturation this study measures a property of *the data*, or just of
    what a local convolution can extract? A model that mixes information
    globally at every layer rather than through a growing receptive field
    answers it, because the two families fail differently when the budget is
    cut.

    Bytes are grouped into ``patch``-byte tokens by a strided convolution
    before attention, which is what the reviewed transformers do as well --
    ET-BERT tokenises hex pairs, YaTC takes 2D patches -- and which keeps
    attention quadratic in budget/patch rather than in budget. The model still
    reads every byte inside the budget; only the token boundary changes.

    This is trained from scratch on the same data, schedule and seeds as the
    other baselines. It is therefore a comparison of *architectures at matched
    budget and matched supervision*, not a reproduction of the published
    pretrained systems, whose results come mostly from pre-training on large
    unlabelled corpora this study does not use. Read it as evidence about the
    shape of the budget curve, not as a reimplementation of P5 or P6.
    """

    def __init__(self, n_classes: int, pad_mode: str = "zero", patch: int = 4,
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 4,
                 dropout: float = 0.1, max_tokens: int = 512):
        super().__init__()
        self.front = ByteFrontEnd(pad_mode)
        self.patch = patch
        self.embed = nn.Conv1d(self.front.out_channels, d_model,
                               kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, max_tokens, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        # norm_first disables the nested-tensor fast path anyway; saying so
        # explicitly keeps a warning out of every training log.
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, n_classes))

    def forward(self, x, valid_len: Optional[torch.Tensor] = None):
        h = self.embed(self.front(x))                      # (B, d, T)
        tokens = h.shape[2]
        if tokens > self.pos.shape[1]:
            raise ValueError(
                f"{tokens} tokens exceeds max_tokens={self.pos.shape[1]}; "
                f"raise max_tokens or the patch size")
        h = h.transpose(1, 2) + self.pos[:, :tokens]       # (B, T, d)

        # A token counts as padding only when every byte in it is padding, so
        # the partially filled token at the boundary is still attended to.
        pad_mask = None
        if valid_len is not None:
            tok_pos = torch.arange(tokens, device=h.device)[None, :]
            n_valid = (valid_len.clamp(min=0) + self.patch - 1) // self.patch
            pad_mask = tok_pos >= n_valid.clamp(max=tokens)[:, None]
            # An all-padding row would make every key masked and produce NaNs.
            pad_mask[:, 0] = False

        h = self.norm(self.encoder(h, src_key_padding_mask=pad_mask))
        keep = None if pad_mask is None else (~pad_mask).to(h.dtype).unsqueeze(1)
        return self.head(_masked_pool(h.transpose(1, 2), keep))


MODEL_REGISTRY = {
    "wang2d": Wang2DCNN,
    "wang1d": Wang1DCNN,
    "deeppacket": DeepPacket,
    "adaptive": AdaptiveByteCNN,
    "transformer": ByteTransformer,
}


def build_model(name: str, n_classes: int, pad_mode: str = "zero", **kwargs) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {name!r}; have {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](n_classes=n_classes, pad_mode=pad_mode, **kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
