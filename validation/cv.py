# validation/cv.py
"""
Cross-validation splitters that are safe for financial time series.

Two families:

* PurgedKfold / PurgedWalkForward — classic K-fold or rolling-origin splits with
  (a) purge: drop train samples whose *label window* overlaps the test window,
  (b) embargo: drop a small % of samples immediately *after* the test window.

* CombinatorialPurgedKFold — for PBO. Splits the series into S chunks and
  enumerates all C(S, S//2) train/test partitions.

All splitters operate on integer index ranges [0, T). Label windows are provided
by the caller as `label_end_idx[i]` — the last index whose future the label at i
depends on. For simple next-bar labels, `label_end_idx[i] = i + 1`.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterator, Sequence

import numpy as np


@dataclass(frozen=True)
class Split:
    train_idx: np.ndarray
    test_idx: np.ndarray


def _make_label_ends(T: int, horizon: int, label_end_idx: Sequence[int] | None) -> np.ndarray:
    if label_end_idx is None:
        # default: label at i depends on next `horizon` bars
        end = np.arange(T, dtype=np.int64) + int(horizon)
        return np.minimum(end, T - 1)
    arr = np.asarray(label_end_idx, dtype=np.int64)
    if arr.shape != (T,):
        raise ValueError(f"label_end_idx must have shape ({T},), got {arr.shape}")
    return arr


def _purged_train_mask(
    T: int, train_idx: np.ndarray, test_idx: np.ndarray,
    label_end: np.ndarray, embargo_pct: float,
) -> np.ndarray:
    """
    Return a boolean mask over train_idx selecting samples to KEEP.

    Purge: sample i is dropped if its [i, label_end[i]] window overlaps any test
    index. Embargo: sample i is dropped if i is within `embargo_pct * T` indices
    *after* the test window ends (to prevent micro-leakage via overlapping features).
    """
    keep = np.ones(len(train_idx), dtype=bool)
    test_min = int(test_idx.min()) if len(test_idx) else 0
    test_max = int(test_idx.max()) if len(test_idx) else -1
    embargo = int(np.floor(embargo_pct * T))
    embargo_start = test_max + 1
    embargo_end = min(T - 1, test_max + embargo)

    for k, i in enumerate(train_idx):
        li = int(label_end[i])
        # Purge: overlap with any test index
        if li >= test_min and i <= test_max:
            keep[k] = False
            continue
        # Embargo: inside the embargo window after test
        if embargo and embargo_start <= i <= embargo_end:
            keep[k] = False
    return keep


class PurgedWalkForward:
    """
    Rolling-origin walk-forward with purge + embargo.

    Parameters
    ----------
    n_splits : int
    train_pct : float   fraction of total T used for the initial train window
    test_pct  : float   fraction of total T used for each test window
    step_pct  : float   how far to advance the window each split
    embargo_pct : float fraction of T to embargo after each test window
    horizon : int       label horizon in bars (for default label_end_idx)
    label_end_idx : optional, explicit label end per sample
    """

    def __init__(
        self, n_splits: int = 5, train_pct: float = 0.5, test_pct: float = 0.1,
        step_pct: float = 0.1, embargo_pct: float = 0.01, horizon: int = 1,
        label_end_idx: Sequence[int] | None = None,
    ) -> None:
        self.n_splits = int(n_splits)
        self.train_pct = float(train_pct)
        self.test_pct = float(test_pct)
        self.step_pct = float(step_pct)
        self.embargo_pct = float(embargo_pct)
        self.horizon = int(horizon)
        self.label_end_idx = label_end_idx

    def split(self, T: int) -> Iterator[Split]:
        train_len = max(2, int(np.floor(self.train_pct * T)))
        test_len = max(1, int(np.floor(self.test_pct * T)))
        step = max(1, int(np.floor(self.step_pct * T)))
        label_end = _make_label_ends(T, self.horizon, self.label_end_idx)

        produced = 0
        train_start = 0
        while produced < self.n_splits:
            train_idx = np.arange(train_start, train_start + train_len)
            test_start = train_start + train_len
            test_end = test_start + test_len
            if test_end > T:
                break
            test_idx = np.arange(test_start, test_end)
            keep = _purged_train_mask(T, train_idx, test_idx, label_end, self.embargo_pct)
            train_idx = train_idx[keep]
            if len(train_idx) >= 2 and len(test_idx) >= 1:
                yield Split(train_idx=train_idx, test_idx=test_idx)
                produced += 1
            train_start += step


class CombinatorialPurgedKFold:
    """
    Enumerate all C(S, S//2) train/test partitions, purged + embargoed.

    Used to compute PBO (Bailey et al.). Yields (train_idx, test_idx) pairs;
    test_idx is S//2 *consecutive* chunks (complement of the train chunks).
    """

    def __init__(
        self, n_chunks: int = 8, embargo_pct: float = 0.01, horizon: int = 1,
        label_end_idx: Sequence[int] | None = None,
        max_splits: int | None = None,
    ) -> None:
        if n_chunks % 2 != 0:
            raise ValueError("n_chunks must be even")
        self.n_chunks = int(n_chunks)
        self.embargo_pct = float(embargo_pct)
        self.horizon = int(horizon)
        self.label_end_idx = label_end_idx
        self.max_splits = max_splits

    def split(self, T: int) -> Iterator[Split]:
        S = self.n_chunks
        # S equal-size chunks; last chunk takes the remainder.
        bounds = [(i * T) // S for i in range(S + 1)]
        label_end = _make_label_ends(T, self.horizon, self.label_end_idx)
        all_chunks = list(range(S))

        emitted = 0
        for test_chunks in combinations(all_chunks, S // 2):
            test_idx = np.concatenate([
                np.arange(bounds[c], bounds[c + 1]) for c in test_chunks
            ])
            train_chunks = [c for c in all_chunks if c not in test_chunks]
            raw_train = np.concatenate([
                np.arange(bounds[c], bounds[c + 1]) for c in train_chunks
            ])
            keep = _purged_train_mask(
                T, raw_train, test_idx, label_end, self.embargo_pct,
            )
            train_idx = raw_train[keep]
            if len(train_idx) < 2 or len(test_idx) < 1:
                continue
            yield Split(train_idx=train_idx, test_idx=test_idx)
            emitted += 1
            if self.max_splits is not None and emitted >= self.max_splits:
                return
