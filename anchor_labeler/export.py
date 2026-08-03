"""
Export the curated anchors.

Only events the user KEPT (confirmed proposal or corrected label) are exported;
skipped and ignored events never leave the session. Three artefacts, so the
result drops into the existing pipeline without glue code:

  anchors_<stamp>.json  full record (label, whether it was corrected, round)
  anchors_<stamp>.csv   event_index,label,label_name,confirmed
  anchors_<stamp>.yml   ``mode: manual`` config readable by
                        ``src/convert/mark_anchors.py``

and, optionally, a copy of the dataset with ``anchor_label`` filled in
(-1 everywhere else) which the training configs can consume directly.
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
                   anchor_field: str = "anchor_label") -> Dict[str, Any]:
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
    return out


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
