"""
Session state: what the user has decided, and crash-proof persistence.

Two files are kept side by side:

  * ``<session>.json``      canonical state, written atomically (tmp + rename)
    by the autosave thread every ``autosave_seconds`` and on every export/exit.
  * ``<session>.json.tmp``  a snapshot rewritten on EVERY single decision.

The tmp file is the "no perdemos el estado" guarantee: it costs one small
JSON dump per click, and on startup it is preferred over the canonical file
whenever it is newer (i.e. the process died between autosaves), so at most the
click in flight can be lost.

Decision statuses
  kept     -- the event IS an anchor of ``label`` (``label == proposed`` means
              the automatic proposal was confirmed, otherwise it was corrected)
  ignored  -- rejected for good; never proposed again in any future round
  skipped  -- seen but undecided; may come back in a later round
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional

STATUS_KEPT = "kept"
STATUS_IGNORED = "ignored"
STATUS_SKIPPED = "skipped"
VALID_STATUS = {STATUS_KEPT, STATUS_IGNORED, STATUS_SKIPPED}

SCHEMA_VERSION = 1


def _atomic_write(path: str, payload: str) -> None:
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".sess-", suffix=".json")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class SessionStore:
    def __init__(self, path: str, config_name: str, dataset_path: str,
                 autosave_seconds: float = 5.0):
        self.path = os.path.abspath(path)
        self.tmp_path = self.path + ".tmp"
        self.autosave_seconds = max(0.5, float(autosave_seconds))
        self._lock = threading.RLock()
        self._dirty = False
        self._stop = threading.Event()

        self.state: Dict[str, Any] = self._load() or {
            "version": SCHEMA_VERSION,
            "config_name": config_name,
            "dataset_path": dataset_path,
            "created": time.time(),
            "updated": time.time(),
            "round_index": 0,
            "rounds": [],
            "queue": [],
            "cursor": 0,
            "decisions": {},
        }
        self.state.setdefault("config_name", config_name)
        self.state.setdefault("dataset_path", dataset_path)
        self._thread = threading.Thread(target=self._autosave_loop, daemon=True)
        self._thread.start()

    # ---- persistence ---------------------------------------------------

    def _load(self) -> Optional[Dict[str, Any]]:
        """Prefer whichever of canonical/tmp is newer and parses."""
        cands: List[str] = []
        for p in (self.tmp_path, self.path):
            if os.path.exists(p):
                cands.append(p)
        cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        for p in cands:
            try:
                with open(p) as fh:
                    data = json.load(fh)
                if p == self.tmp_path:
                    print(f"[session] recovered newer autosave snapshot: {p}")
                else:
                    print(f"[session] resumed: {p}")
                return data
            except Exception as exc:      # corrupt half-write -> try the other
                print(f"[session] could not read {p}: {exc}")
        return None

    def _snapshot(self) -> str:
        with self._lock:
            self.state["updated"] = time.time()
            return json.dumps(self.state, separators=(",", ":"))

    def flush_tmp(self) -> None:
        """Cheap per-click snapshot (no fsync, no rename)."""
        payload = self._snapshot()
        try:
            with open(self.tmp_path, "w") as fh:
                fh.write(payload)
        except OSError as exc:
            print(f"[session] tmp write failed: {exc}")

    def save(self) -> None:
        """Durable atomic save of the canonical file."""
        payload = self._snapshot()
        _atomic_write(self.path, payload)
        with self._lock:
            self._dirty = False
        try:
            # tmp is now redundant but kept in sync so mtime ordering stays sane
            with open(self.tmp_path, "w") as fh:
                fh.write(payload)
        except OSError:
            pass

    def _touch(self) -> None:
        with self._lock:
            self._dirty = True
        self.flush_tmp()

    def _autosave_loop(self) -> None:
        while not self._stop.wait(self.autosave_seconds):
            try:
                if self._dirty:
                    self.save()
            except Exception as exc:
                print(f"[session] autosave failed: {exc}")

    def close(self) -> None:
        self._stop.set()
        try:
            self.save()
        except Exception:
            pass

    # ---- queries -------------------------------------------------------

    def decision(self, index: int) -> Optional[Dict[str, Any]]:
        return self.state["decisions"].get(str(int(index)))

    def indices_with_status(self, status: str) -> List[int]:
        return [int(k) for k, v in self.state["decisions"].items() if v.get("status") == status]

    def resolved_indices(self) -> List[int]:
        """Never propose these again: kept anchors + explicit ignores."""
        return [int(k) for k, v in self.state["decisions"].items()
                if v.get("status") in (STATUS_KEPT, STATUS_IGNORED)]

    def seen_indices(self) -> List[int]:
        return [int(k) for k in self.state["decisions"]]

    def kept_anchors(self) -> Dict[int, int]:
        return {int(k): int(v["label"]) for k, v in self.state["decisions"].items()
                if v.get("status") == STATUS_KEPT and v.get("label") is not None}

    def summary(self) -> Dict[str, Any]:
        dec = self.state["decisions"].values()
        kept = [d for d in dec if d.get("status") == STATUS_KEPT]
        per_class: Dict[str, int] = {}
        for d in kept:
            per_class[str(d["label"])] = per_class.get(str(d["label"]), 0) + 1
        queue = self.state["queue"]
        pending = [i for i in queue if (self.decision(i) or {}).get("status") is None]
        return {
            "round_index": self.state["round_index"],
            "queue_len": len(queue),
            "cursor": self.state["cursor"],
            "pending_in_queue": len(pending),
            "kept": len(kept),
            "kept_confirmed": sum(1 for d in kept if d.get("label") == d.get("proposed")),
            "kept_corrected": sum(1 for d in kept if d.get("label") != d.get("proposed")),
            "ignored": sum(1 for d in dec if d.get("status") == STATUS_IGNORED),
            "skipped": sum(1 for d in dec if d.get("status") == STATUS_SKIPPED),
            "kept_per_class": per_class,
            "session_path": self.path,
            "updated": self.state.get("updated"),
        }

    # ---- mutations -----------------------------------------------------

    def start_round(self, proposals: List[Any], params: Dict[str, Any],
                    stats: List[Dict[str, Any]], append: bool = False) -> None:
        with self._lock:
            # A fresh sampling round supersedes review mode: the queue it builds
            # replaces whatever was parked, so keeping the stash would restore a
            # queue the user has already moved on from.
            self.state["mode"] = None
            self.state.pop("queue_stash", None)
            self.state.pop("review_params", None)
            ri = int(self.state["round_index"]) + 1
            self.state["round_index"] = ri
            self.state["rounds"].append({
                "index": ri, "params": params, "stats": stats,
                "n_proposals": len(proposals), "created": time.time(),
            })
            new_queue = [int(i) for i, _ in proposals]
            if append:
                have = set(self.state["queue"])
                self.state["queue"] = self.state["queue"] + [i for i in new_queue if i not in have]
            else:
                self.state["queue"] = new_queue
                self.state["cursor"] = 0
            props = self.state.setdefault("proposed_label", {})
            for i, ci in proposals:
                props[str(int(i))] = int(ci)
                d = self.state["decisions"].get(str(int(i)))
                if d is not None and d.get("status") == STATUS_SKIPPED:
                    # a skipped event coming back starts fresh in this round
                    d["round"] = ri
        self._touch()

    # ---- review mode ---------------------------------------------------
    #
    # Reviewing already-saved anchors is a QUEUE operation and nothing else:
    # `decisions` and `proposed_label` are never touched here, so re-reviewing
    # cannot lose or rewrite what was labeled. The sampling queue you were in
    # the middle of is parked in `queue_stash` and comes back with
    # `end_review()`, which is why entering review costs no progress.

    def in_review(self) -> bool:
        return self.state.get("mode") == "review"

    def begin_review(self, indices: List[int], params: Dict[str, Any]) -> None:
        with self._lock:
            if not self.in_review():
                # Only stash the SAMPLING queue: entering review twice in a row
                # must not overwrite the parked queue with a review queue.
                self.state["queue_stash"] = {
                    "queue": list(self.state["queue"]),
                    "cursor": int(self.state["cursor"]),
                }
            self.state["mode"] = "review"
            self.state["review_params"] = dict(params)
            self.state["queue"] = [int(i) for i in indices]
            self.state["cursor"] = 0
        self._touch()

    def end_review(self) -> None:
        with self._lock:
            stash = self.state.pop("queue_stash", None) or {}
            self.state["mode"] = None
            self.state.pop("review_params", None)
            self.state["queue"] = [int(i) for i in stash.get("queue", [])]
            n = len(self.state["queue"])
            self.state["cursor"] = max(0, min(int(stash.get("cursor", 0)), max(0, n - 1)))
        self._touch()

    def kept_indices(self, labels: Optional[List[int]] = None,
                     order: str = "class") -> List[int]:
        """Saved anchors, optionally restricted to some classes.

        order 'class' groups by label then index (review one class at a time);
        'index' keeps file order (see the events as they sit in the h5).
        """
        wanted = None if not labels else {int(x) for x in labels}
        pairs = [(int(k), int(v["label"]))
                 for k, v in self.state["decisions"].items()
                 if v.get("status") == STATUS_KEPT and v.get("label") is not None
                 and (wanted is None or int(v["label"]) in wanted)]
        if order == "index":
            pairs.sort(key=lambda p: p[0])
        else:
            pairs.sort(key=lambda p: (p[1], p[0]))
        return [i for i, _ in pairs]

    def set_cursor(self, cursor: int) -> None:
        with self._lock:
            n = len(self.state["queue"])
            self.state["cursor"] = max(0, min(int(cursor), max(0, n - 1)))
        self._touch()

    def proposed_label(self, index: int) -> Optional[int]:
        v = self.state.get("proposed_label", {}).get(str(int(index)))
        return None if v is None else int(v)

    def record(self, index: int, status: str, label: Optional[int] = None,
               note: str = "") -> Dict[str, Any]:
        if status not in VALID_STATUS:
            raise ValueError(f"invalid status '{status}'")
        if status == STATUS_KEPT and label is None:
            raise ValueError("status 'kept' needs a label")
        with self._lock:
            entry = {
                "status": status,
                "label": None if label is None else int(label),
                "proposed": self.proposed_label(index),
                "round": self.state["round_index"],
                "ts": time.time(),
            }
            if note:
                entry["note"] = note
            self.state["decisions"][str(int(index))] = entry
        self._touch()
        return entry

    def undo(self, index: int) -> None:
        with self._lock:
            self.state["decisions"].pop(str(int(index)), None)
        self._touch()
