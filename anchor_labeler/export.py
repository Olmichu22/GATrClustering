"""
Export the curated anchors.

Only events the user KEPT (confirmed proposal or corrected label) are exported;
skipped and ignored events never leave the session. Three artefacts, so the
result drops into the existing pipeline without glue code:

  anchors_<stamp>.json  full record (label, whether it was corrected, round)
  anchors_<stamp>.csv   event_index,label,label_name,confirmed
  anchors_<stamp>.yml   ``mode: manual`` config readable by
                        ``src/convert/mark_anchors.py``

and, optionally, either

  * a copy of the dataset with ``anchor_label`` filled in (-1 everywhere else),
    for a PRIMARY file that is also the training file, or
  * one small h5 PER CLASS holding only the kept events of that class
    (``write_subsets``), for a file used purely as an anchor source. That is the
    shape ``data.anchor_datasets`` expects: it forces one label on every event
    of the file, so the file must contain nothing else.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import time
from typing import Any, Dict, List, Optional

from .config import LabelerConfig
from .session import SessionStore


def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def export_session(cfg: LabelerConfig, store: SessionStore,
                   write_h5: bool = False, h5_out: Optional[str] = None,
                   anchor_field: str = "anchor_label",
                   write_subsets: bool = False) -> Dict[str, Any]:
    store.save()
    kept = store.kept_anchors()
    os.makedirs(cfg.export_dir, exist_ok=True)
    stamp = _stamp()
    base = os.path.join(cfg.export_dir, f"anchors_{stamp}")
    names = {c.index: c.name for c in cfg.classes}

    rows: List[Dict[str, Any]] = []
    for idx in sorted(kept):
        d = store.decision(idx) or {}
        rows.append({
            "event_index": int(idx),
            "label": int(kept[idx]),
            "label_name": names.get(int(kept[idx]), str(kept[idx])),
            "proposed": d.get("proposed"),
            "confirmed": d.get("proposed") == kept[idx],
            "round": d.get("round"),
            "ts": d.get("ts"),
        })

    json_path = base + ".json"
    with open(json_path, "w") as fh:
        json.dump({
            "config_name": cfg.name,
            "dataset_path": cfg.dataset.path,
            "exported": time.time(),
            "classes": [{"index": c.index, "name": c.name} for c in cfg.classes],
            "summary": store.summary(),
            "anchors": rows,
        }, fh, indent=2)

    csv_path = base + ".csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["event_index", "label", "label_name", "confirmed"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})

    # mark_anchors.py-compatible manual config
    manual: Dict[int, List[int]] = {c.index: [] for c in cfg.classes}
    for r in rows:
        manual.setdefault(r["label"], []).append(r["event_index"])
    yml_path = base + ".yml"
    with open(yml_path, "w") as fh:
        fh.write("# Hand-curated anchors exported by the anchor labeler.\n")
        fh.write(f"# dataset: {cfg.dataset.path}\n")
        fh.write("mode: manual\n")
        fh.write(f"nhits_field: {cfg.dataset.nhits_field or 'null'}\n")
        fh.write("manual:\n")
        for ci in sorted(manual):
            fh.write(f"  {ci}: {manual[ci]}   # {names.get(ci, ci)}\n")

    out: Dict[str, Any] = {
        "json": json_path, "csv": csv_path, "yaml": yml_path,
        "n_anchors": len(rows),
        "per_class": {names.get(ci, str(ci)): len(v) for ci, v in sorted(manual.items())},
    }

    if write_h5:
        out["h5"] = _write_h5(cfg, kept, h5_out, anchor_field, stamp)
    if write_subsets:
        out["subsets"] = _write_subsets(cfg, kept, anchor_field, stamp)
    return out


def _write_subsets(cfg: LabelerConfig, kept: Dict[int, int], anchor_field: str,
                   stamp: str) -> Dict[str, str]:
    """One h5 per class with ONLY that class's kept events (CSR rebuilt).

    Every per-hit array is re-sliced event by event and ``offsets`` recomputed,
    so the output is a valid flat/CSR file the training loader reads unchanged.
    Per-event arrays are indexed with the kept list. Arrays are classified by
    length against the source (n_hits vs n_events), so no field list is
    hardcoded and a different detector exports the same way.
    """
    import h5py
    import numpy as np

    src = cfg.dataset.path
    names = {c.index: c.name for c in cfg.classes}
    by_class: Dict[int, List[int]] = {}
    for idx, lab in kept.items():
        by_class.setdefault(int(lab), []).append(int(idx))

    stem = os.path.splitext(os.path.basename(src))[0]
    os.makedirs(cfg.export_dir, exist_ok=True)
    written: Dict[str, str] = {}

    with h5py.File(src, "r") as f:
        off = np.asarray(f[cfg.dataset.offsets])
        n_events, n_hits = len(off) - 1, int(off[-1])
        hit_keys, event_keys = [], []
        for k in f:
            if k == cfg.dataset.offsets:
                continue
            n = f[k].shape[0]
            if n == n_hits and n != n_events:
                hit_keys.append(k)
            elif n == n_events:
                event_keys.append(k)

        for ci in sorted(by_class):
            sel = np.array(sorted(i for i in by_class[ci] if 0 <= i < n_events), dtype=np.int64)
            if sel.size == 0:
                continue
            counts = (off[sel + 1] - off[sel]).astype(np.int64)
            new_off = np.zeros(sel.size + 1, dtype=np.int64)
            np.cumsum(counts, out=new_off[1:])
            # hit rows of the kept events, in order
            rows = np.concatenate([np.arange(off[i], off[i + 1], dtype=np.int64)
                                   for i in sel]) if sel.size else np.zeros(0, dtype=np.int64)

            name = names.get(ci, str(ci))
            path = os.path.join(cfg.export_dir, f"{stem}_manual_{name}_{stamp}.h5")
            with h5py.File(path, "w") as g:
                g.create_dataset(cfg.dataset.offsets, data=new_off, compression="lzf")
                for k in hit_keys:
                    g.create_dataset(k, data=np.asarray(f[k])[rows], compression="lzf")
                for k in event_keys:
                    if k == anchor_field:
                        continue
                    g.create_dataset(k, data=np.asarray(f[k])[sel], compression="lzf")
                g.create_dataset(anchor_field,
                                 data=np.full(sel.size, int(ci), dtype=np.int64),
                                 compression="lzf")
                g.attrs["source_file"] = src
                g.attrs["source_indices"] = sel
                g.attrs["anchor_label"] = int(ci)
                g.attrs["exported"] = stamp
            written[name] = path
    return written


def _write_h5(cfg: LabelerConfig, kept: Dict[int, int], h5_out: Optional[str],
              anchor_field: str, stamp: str) -> str:
    import h5py
    import numpy as np

    src = cfg.dataset.path
    if not h5_out:
        stem = os.path.splitext(os.path.basename(src))[0]
        h5_out = os.path.join(cfg.export_dir, f"{stem}_manualanchors_{stamp}.h5")
    h5_out = os.path.abspath(h5_out)
    if os.path.abspath(src) != h5_out:
        os.makedirs(os.path.dirname(h5_out), exist_ok=True)
        shutil.copyfile(src, h5_out)
    with h5py.File(h5_out, "r+") as f:
        n_events = len(f[cfg.dataset.offsets]) - 1
        anchor = -np.ones(n_events, dtype=np.int64)
        for idx, lab in kept.items():
            if 0 <= idx < n_events:
                anchor[idx] = int(lab)
        if anchor_field in f:
            del f[anchor_field]
        f.create_dataset(anchor_field, data=anchor, compression="lzf")
    return h5_out
