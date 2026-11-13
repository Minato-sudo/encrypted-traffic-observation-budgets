"""Training loop (Phase 2 Steps 5-7, 9).

One loop serves every model and every budget, because the comparison is only
valid if architecture and training are held constant while the budget moves.
Anything that could differ between budgets -- optimiser, schedule, early
stopping criterion, seed -- is fixed here and recorded in the returned
history, so a run can be reproduced from its own record.

For :class:`~nettc.models.AdaptiveByteCNN` the loss sums cross-entropy over
every early-exit head, so a single forward pass trains all budgets at once.
That is only sound because the heads are causal; see the note in models.py.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import Result, evaluate
from .models import build_model, count_parameters
from .slicer import BudgetSpec, PadMode, take


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 6
    class_weighted: bool = False
    pad_mode: PadMode = "zero"
    amp: bool = True
    device: Optional[str] = None
    num_workers: int = 0           # slicing is vectorised; no loader processes needed
    grad_clip: float = 5.0
    label_smoothing: float = 0.0
    head_loss_weights: Optional[Sequence[float]] = None
    min_epochs: int = 3
    verbose: bool = False

    def resolved_device(self) -> str:
        return self.device or ("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainHistory:
    model: str
    budget: str
    seed: int
    epochs_run: int
    best_epoch: int
    best_val_macro_f1: float
    train_loss: list = field(default_factory=list)
    val_macro_f1: list = field(default_factory=list)
    seconds: float = 0.0
    params: int = 0
    config: dict = field(default_factory=dict)
    n_train: int = 0
    n_val: int = 0


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SlicedView:
    """Materialised (x, valid_len, y) for one split at one budget.

    Slicing once per split rather than once per epoch matters: the sweep
    trains the same data at eight budgets across three seeds, and the slicer
    would otherwise dominate the runtime.
    """

    def __init__(self, store, idx: np.ndarray, spec: BudgetSpec, pad_mode: PadMode,
                 source: str = "bytes"):
        self.idx = np.asarray(idx)
        sl = take(store, self.idx, spec, pad_mode=pad_mode, source=source)
        self.x = sl.x
        self.valid_len = np.minimum(sl.valid_len, sl.width).astype(np.int64)
        self.true_valid = sl.valid_len.astype(np.int64)
        self.y = store.label[self.idx]
        self.width = sl.width
        self.padding_fraction = sl.padding_fraction

    def __len__(self) -> int:
        return self.idx.size


def _batches(n: int, batch_size: int, shuffle: bool, rng: Optional[np.random.Generator] = None):
    order = rng.permutation(n) if (shuffle and rng is not None) else np.arange(n)
    for start in range(0, n, batch_size):
        yield order[start:start + batch_size]


def _to_device(view: SlicedView, sel: np.ndarray, device: str, needs_len: bool):
    x = torch.from_numpy(np.ascontiguousarray(view.x[sel])).to(device, non_blocking=True)
    y = torch.from_numpy(view.y[sel]).to(device, non_blocking=True)
    vl = (torch.from_numpy(view.valid_len[sel]).to(device, non_blocking=True)
          if needs_len else None)
    return x, y, vl


def _loss(outputs, y, weight, cfg: TrainConfig) -> torch.Tensor:
    """Cross-entropy, summed over heads when the model has several."""
    if isinstance(outputs, (list, tuple)):
        weights = cfg.head_loss_weights or [1.0] * len(outputs)
        total = outputs[0].new_zeros(())
        for logits, w in zip(outputs, weights):
            total = total + w * F.cross_entropy(
                logits, y, weight=weight, label_smoothing=cfg.label_smoothing)
        return total / max(sum(weights[:len(outputs)]), 1e-9)
    return F.cross_entropy(outputs, y, weight=weight, label_smoothing=cfg.label_smoothing)


def _final_logits(outputs):
    return outputs[-1] if isinstance(outputs, (list, tuple)) else outputs


@torch.no_grad()
def predict(model: nn.Module, view: SlicedView, cfg: TrainConfig,
            head: int = -1) -> np.ndarray:
    """Predicted labels. ``head`` selects an early-exit head on multi-head models."""
    device = cfg.resolved_device()
    needs_len = cfg.pad_mode == "convmask"
    model.eval()
    preds = np.empty(len(view), dtype=np.int64)
    for sel in _batches(len(view), cfg.batch_size, shuffle=False):
        x, _y, vl = _to_device(view, sel, device, needs_len)
        with torch.autocast("cuda", enabled=cfg.amp and device == "cuda"):
            out = model(x, vl)
        logits = out[head] if isinstance(out, (list, tuple)) else out
        preds[sel] = logits.float().argmax(dim=1).cpu().numpy()
    return preds


def train_model(store, train_idx: np.ndarray, val_idx: np.ndarray,
                spec: BudgetSpec, model_name: str, cfg: TrainConfig,
                seed: int = 0, model_kwargs: Optional[dict] = None,
                source: str = "bytes") -> tuple[nn.Module, TrainHistory, SlicedView, SlicedView]:
    """Train one model at one budget with one seed.

    Retries at a halved batch size if the GPU runs out of memory. The sweep
    runs the same configuration across budgets from 64 to 1500 bytes and
    across models whose activation footprints differ by an order of
    magnitude: the adaptive encoder does no downsampling, so at a 1024-byte
    ceiling it holds far more activations than the 1D baseline at 64 bytes.
    One batch size cannot fit all of them, and the alternative to backing off
    is choosing a batch small enough for the worst case and paying it
    everywhere. The batch actually used is recorded in the history.
    """
    attempt = cfg
    while True:
        try:
            return _train_once(store, train_idx, val_idx, spec, model_name,
                               attempt, seed, model_kwargs, source)
        except torch.OutOfMemoryError:
            if attempt.batch_size <= 16:
                raise
            torch.cuda.empty_cache()
            attempt = dataclasses.replace(
                attempt, batch_size=max(16, attempt.batch_size // 2))
            print(f"    out of GPU memory; retrying at batch size "
                  f"{attempt.batch_size}", flush=True)


def _train_once(store, train_idx: np.ndarray, val_idx: np.ndarray,
                spec: BudgetSpec, model_name: str, cfg: TrainConfig,
                seed: int, model_kwargs: Optional[dict],
                source: str) -> tuple[nn.Module, TrainHistory, SlicedView, SlicedView]:
    """One training attempt at a fixed batch size."""
    set_seed(seed)
    device = cfg.resolved_device()
    needs_len = cfg.pad_mode == "convmask"

    tr = SlicedView(store, train_idx, spec, cfg.pad_mode, source=source)
    va = SlicedView(store, val_idx, spec, cfg.pad_mode, source=source)

    kwargs = dict(model_kwargs or {})
    if model_name == "wang2d":
        kwargs.setdefault("n_bytes", tr.width)
    model = build_model(model_name, store.n_classes, pad_mode=cfg.pad_mode, **kwargs).to(device)

    weight = None
    if cfg.class_weighted:
        counts = np.bincount(tr.y, minlength=store.n_classes).astype(np.float64)
        inv = np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0)
        inv = inv / inv[inv > 0].mean()
        weight = torch.tensor(inv, dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device == "cuda")
    rng = np.random.default_rng(seed)

    hist = TrainHistory(model=model_name, budget=spec.label, seed=seed, epochs_run=0,
                        best_epoch=-1, best_val_macro_f1=-1.0,
                        params=count_parameters(model), config=asdict(cfg),
                        n_train=len(tr), n_val=len(va))
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    t0 = time.time()
    stale = 0

    for epoch in range(cfg.epochs):
        model.train()
        running, seen = 0.0, 0
        for sel in _batches(len(tr), cfg.batch_size, shuffle=True, rng=rng):
            x, y, vl = _to_device(tr, sel, device, needs_len)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=cfg.amp and device == "cuda"):
                loss = _loss(model(x, vl), y, weight, cfg)
            scaler.scale(loss).backward()
            if cfg.grad_clip:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            running += float(loss.detach()) * sel.size
            seen += sel.size
        sched.step()

        val_pred = predict(model, va, cfg)
        val_res = evaluate(va.y, val_pred, store.label_names, with_confusion=False)
        hist.train_loss.append(running / max(seen, 1))
        hist.val_macro_f1.append(val_res.macro_f1)
        hist.epochs_run = epoch + 1

        if val_res.macro_f1 > hist.best_val_macro_f1:
            hist.best_val_macro_f1 = val_res.macro_f1
            hist.best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if cfg.verbose:
            print(f"    epoch {epoch:>3} loss {hist.train_loss[-1]:.4f} "
                  f"val macroF1 {val_res.macro_f1:.4f}")
        if stale >= cfg.patience and epoch + 1 >= cfg.min_epochs:
            break

    model.load_state_dict(best_state)
    hist.seconds = time.time() - t0
    return model, hist, tr, va


def evaluate_model(model: nn.Module, store, test_idx: np.ndarray, spec: BudgetSpec,
                   cfg: TrainConfig, head: int = -1, source: str = "bytes"
                   ) -> tuple[Result, np.ndarray, SlicedView]:
    te = SlicedView(store, test_idx, spec, cfg.pad_mode, source=source)
    pred = predict(model, te, cfg, head=head)
    res = evaluate(te.y, pred, store.label_names,
                   meta={"budget": spec.label, "width": te.width,
                         "mean_padding_fraction": float(te.padding_fraction.mean())})
    return res, pred, te


@torch.no_grad()
def early_exit_sweep(model, store, test_idx: np.ndarray, spec: BudgetSpec,
                     cfg: TrainConfig, thresholds: Sequence[float]) -> list[dict]:
    """Run the exit policy at each threshold and record accuracy against cost.

    Step 7 E4. The cost reported is mean bytes genuinely read, capped by each
    session's real length, which is the x-axis of the Step 12 frontier.
    """
    device = cfg.resolved_device()
    needs_len = cfg.pad_mode == "convmask"
    te = SlicedView(store, test_idx, spec, cfg.pad_mode)
    model.eval()
    rows = []
    for tau in thresholds:
        preds = np.empty(len(te), dtype=np.int64)
        bytes_read = np.empty(len(te), dtype=np.float64)
        exits = np.empty(len(te), dtype=np.int64)
        for sel in _batches(len(te), cfg.batch_size, shuffle=False):
            x, _y, vl = _to_device(te, sel, device, needs_len)
            if vl is None:
                vl = torch.from_numpy(te.true_valid[sel]).to(device)
            p, used, eidx = model.early_exit(x, vl, float(tau))
            preds[sel] = p.cpu().numpy()
            bytes_read[sel] = used.float().cpu().numpy()
            exits[sel] = eidx.cpu().numpy()
        res = evaluate(te.y, preds, store.label_names, with_confusion=False)
        rows.append({
            "threshold": float(tau),
            "accuracy": res.accuracy,
            "macro_f1": res.macro_f1,
            "weighted_f1": res.weighted_f1,
            "mean_bytes_read": float(bytes_read.mean()),
            "median_bytes_read": float(np.median(bytes_read)),
            "exit_histogram": np.bincount(exits, minlength=len(model.exits)).tolist(),
        })
    return rows
