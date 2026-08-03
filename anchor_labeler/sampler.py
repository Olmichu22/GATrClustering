"""
Anchor proposal strategies.

``nhits_band`` is the port of the current automatic method
(``src/convert/mark_anchors.py``): per class, keep the tagged events whose hit
count sits within ``window_std`` sigmas of the class mean and draw
``n_per_class`` of them reproducibly; if the band is too narrow it widens to the
N closest to the mean. The labeler adds what the batch script could not do:

  * ``exclude`` -- events the user already resolved (kept anchors) or
    explicitly ignored. Ignored events can NEVER come back.
  * ``seen``    -- events already shown but left undecided. They stay eligible,
    but a fresh round prefers never-shown candidates so re-sampling actually
    brings new material (``prefer_unseen``).

Strategies are looked up in ``STRATEGIES``; adding one is a function plus an
entry, so a new detector can propose anchors by any rule it likes.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

from .config import ClassSpec

# One proposal: (event index, proposed class index)
Proposal = Tuple[int, int]


def _draw(pool: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if pool.size <= n:
        return pool
    return rng.choice(pool, size=n, replace=False)


def _split_unseen(cands: np.ndarray, seen: Set[int]) -> Tuple[np.ndarray, np.ndarray]:
    if not seen:
        return cands, np.empty(0, dtype=np.int64)
    is_seen = np.fromiter((int(c) in seen for c in cands), dtype=bool, count=cands.size)
    return cands[~is_seen], cands[is_seen]


def nhits_band(dataset, cands: np.ndarray, n: int, rng: np.random.Generator,
               params: Dict[str, Any], seen: Set[int]) -> np.ndarray:
    """Events near the class-mean hit count, unseen ones first."""
    if cands.size == 0:
        return cands
    window_std = float(params.get("window_std", 1.0))
    prefer_unseen = bool(params.get("prefer_unseen", True))
    nh = dataset.nhits()

    def band_of(sub: np.ndarray) -> np.ndarray:
        if sub.size == 0:
            return sub
        vals = nh[sub]
        mean, std = vals.mean(), max(vals.std(), 1e-6)
        inside = sub[np.abs(vals - mean) <= window_std * std]
        if inside.size < n:      # auto-widen: the n closest to the mean
            order = np.argsort(np.abs(vals - mean))
            inside = sub[order[: max(n, inside.size)]]
        return inside

    # The band is defined on the FULL candidate pool so the physics window does
    # not drift as events get labelled; only the draw honours the preference.
    band = set(int(x) for x in band_of(cands))
    pool = np.array(sorted(band), dtype=np.int64)
    if not prefer_unseen:
        return _draw(pool, n, rng)

    unseen, already = _split_unseen(pool, seen)
    picked = _draw(unseen, n, rng)
    if picked.size < n and already.size:
        picked = np.concatenate([picked, _draw(already, n - picked.size, rng)])
    return picked


def random_sample(dataset, cands: np.ndarray, n: int, rng: np.random.Generator,
                  params: Dict[str, Any], seen: Set[int]) -> np.ndarray:
    """Uniform draw over the candidate pool (no hit-count window)."""
    if cands.size == 0 or not bool(params.get("prefer_unseen", True)):
        return _draw(cands, n, rng)
    unseen, already = _split_unseen(cands, seen)
    picked = _draw(unseen, n, rng)
    if picked.size < n and already.size:
        picked = np.concatenate([picked, _draw(already, n - picked.size, rng)])
    return picked


STRATEGIES = {"nhits_band": nhits_band, "random": random_sample}


def sample_round(dataset, classes: Sequence[ClassSpec], params: Dict[str, Any],
                 exclude: Iterable[int] = (), seen: Iterable[int] = (),
                 round_index: int = 0) -> Tuple[List[Proposal], List[Dict[str, Any]]]:
    """Propose one round of anchors.

    Returns ``(proposals, stats)`` where stats carries per-class diagnostics for
    the UI (pool size, drawn count, hit-count range).
    """
    strategy = str(params.get("strategy", "nhits_band"))
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown sampling strategy '{strategy}' (known: {sorted(STRATEGIES)})")
    fn = STRATEGIES[strategy]

    n_per = int(params.get("n_per_class", 25))
    # Round index enters the seed so a re-sample genuinely re-draws while each
    # (seed, round) pair stays reproducible.
    rng = np.random.default_rng(int(params.get("seed", 42)) + 10_007 * int(round_index))

    exclude_set = {int(e) for e in exclude}
    seen_set = {int(s) for s in seen}
    nh = dataset.nhits()

    proposals: List[Proposal] = []
    stats: List[Dict[str, Any]] = []
    taken: Set[int] = set()

    for cls in classes:
        if not cls.samplable:
            continue
        mask = dataset.mask_for_source(cls.source)
        cands = np.flatnonzero(mask).astype(np.int64)
        if exclude_set or taken:
            drop = exclude_set | taken
            keep = np.fromiter((int(c) not in drop for c in cands), dtype=bool, count=cands.size)
            cands = cands[keep]
        sel = np.asarray(fn(dataset, cands, n_per, rng, params, seen_set), dtype=np.int64)
        taken.update(int(s) for s in sel)
        proposals.extend((int(i), cls.index) for i in sel)
        stats.append({
            "class_index": cls.index,
            "class_name": cls.name,
            "pool": int(cands.size),
            "drawn": int(sel.size),
            "nhits_mean": float(nh[sel].mean()) if sel.size else None,
            "nhits_min": float(nh[sel].min()) if sel.size else None,
            "nhits_max": float(nh[sel].max()) if sel.size else None,
        })

    # Interleave classes so the reviewer does not see 25 muons in a row.
    by_class: Dict[int, List[Proposal]] = {}
    for p in proposals:
        by_class.setdefault(p[1], []).append(p)
    interleaved: List[Proposal] = []
    while any(by_class.values()):
        for ci in sorted(by_class):
            if by_class[ci]:
                interleaved.append(by_class[ci].pop(0))
    return interleaved, stats
