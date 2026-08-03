"""
Flask backend of the manual anchor labeler.

    python -m anchor_labeler.server --config configs/labeler_sdhcal.yml --port 8060

The server owns three objects: the ``LabelerConfig`` (what a detector looks
like), the dataset backend (read-only) and the ``SessionStore`` (what the user
decided, autosaved). The browser only talks JSON to /api/*, so the same UI works
for any dataset the config can describe.
"""

from __future__ import annotations

import argparse
import atexit
import os
import threading
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request, send_file, send_from_directory

from .config import LabelerConfig, load_config
from .datasets import open_dataset
from .export import export_session
from .sampler import sample_round
from .session import STATUS_IGNORED, STATUS_KEPT, STATUS_SKIPPED, SessionStore

HERE = os.path.dirname(os.path.abspath(__file__))


class Labeler:
    """Everything the request handlers need, behind one lock."""

    def __init__(self, cfg: LabelerConfig, session_path: Optional[str] = None):
        self.cfg = cfg
        self.dataset = open_dataset(cfg.dataset)
        self.store = SessionStore(
            session_path or cfg.session_path,
            config_name=cfg.name,
            dataset_path=cfg.dataset.path,
            autosave_seconds=cfg.autosave_seconds,
        )
        self.lock = threading.RLock()

    # ---- helpers -------------------------------------------------------

    def queue_view(self) -> list:
        st = self.store.state
        out = []
        for i in st["queue"]:
            d = self.store.decision(i) or {}
            out.append({
                "index": int(i),
                "proposed": self.store.proposed_label(i),
                "status": d.get("status"),
                "label": d.get("label"),
            })
        return out

    def state_payload(self) -> Dict[str, Any]:
        return {
            "config": self.cfg.public(),
            "n_events": self.dataset.n_events,
            "summary": self.store.summary(),
            "queue": self.queue_view(),
            "cursor": self.store.state["cursor"],
            "rounds": self.store.state["rounds"][-10:],
        }

    def new_round(self, params: Dict[str, Any], append: bool = False) -> Dict[str, Any]:
        merged = dict(self.cfg.sampling.as_dict())
        merged.update({k: v for k, v in params.items() if v is not None})
        classes = self.cfg.classes
        only = params.get("classes")
        if only:
            wanted = {int(c) for c in only}
            classes = [c for c in classes if c.index in wanted]
        proposals, stats = sample_round(
            self.dataset, classes, merged,
            exclude=self.store.resolved_indices(),
            seen=self.store.seen_indices(),
            round_index=self.store.state["round_index"],
        )
        self.store.start_round(proposals, merged, stats, append=append)
        return {"n_proposals": len(proposals), "stats": stats, "params": merged}

    def close(self) -> None:
        self.store.close()
        self.dataset.close()


def create_app(cfg: LabelerConfig, session_path: Optional[str] = None) -> Flask:
    app = Flask(__name__, static_folder=os.path.join(HERE, "static"),
                template_folder=os.path.join(HERE, "templates"))
    lab = Labeler(cfg, session_path)
    app.config["LABELER"] = lab
    atexit.register(lab.close)

    @app.get("/")
    def index():
        return send_from_directory(app.template_folder, "index.html")

    @app.get("/vendor/plotly.min.js")
    def plotly_js():
        """Serve plotly.js from the installed python package (no CDN needed)."""
        try:
            import plotly
            p = os.path.join(os.path.dirname(plotly.__file__), "package_data", "plotly.min.js")
            if os.path.exists(p):
                return send_file(p, mimetype="application/javascript")
        except Exception:
            pass
        local = os.path.join(HERE, "static", "plotly.min.js")
        if os.path.exists(local):
            return send_file(local, mimetype="application/javascript")
        return ("plotly.min.js not found: pip install plotly, or drop the file in "
                "anchor_labeler/static/", 404)

    @app.get("/api/state")
    def api_state():
        with lab.lock:
            return jsonify(lab.state_payload())

    @app.get("/api/event/<int:index>")
    def api_event(index: int):
        with lab.lock:
            try:
                ev = lab.dataset.event(index)
            except IndexError as exc:
                return jsonify({"error": str(exc)}), 404
            ev["proposed"] = lab.store.proposed_label(index)
            ev["decision"] = lab.store.decision(index)
            return jsonify(ev)

    @app.post("/api/sample")
    def api_sample():
        body = request.get_json(silent=True) or {}
        params = {k: body.get(k) for k in
                  ("strategy", "n_per_class", "seed", "window_std", "prefer_unseen", "classes")}
        with lab.lock:
            try:
                info = lab.new_round(params, append=bool(body.get("append")))
            except (KeyError, ValueError) as exc:
                return jsonify({"error": str(exc)}), 400
            payload = lab.state_payload()
            payload["round_info"] = info
            return jsonify(payload)

    @app.post("/api/decision")
    def api_decision():
        body = request.get_json(silent=True) or {}
        try:
            index = int(body["index"])
            status = str(body["status"])
        except (KeyError, TypeError, ValueError):
            return jsonify({"error": "index and status are required"}), 400
        label = body.get("label")
        if status == STATUS_KEPT and label is None:
            label = lab.store.proposed_label(index)
        if status == STATUS_KEPT:
            if label is None:
                return jsonify({"error": "no label and no proposal for this event"}), 400
            if lab.cfg.class_by_index(int(label)) is None:
                return jsonify({"error": f"unknown class {label}"}), 400
        with lab.lock:
            try:
                entry = lab.store.record(index, status,
                                         None if label is None else int(label),
                                         note=str(body.get("note", "")))
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400
            if body.get("advance", True):
                q = lab.store.state["queue"]
                if index in q:
                    lab.store.set_cursor(q.index(index) + 1)
            return jsonify({"decision": entry, "summary": lab.store.summary(),
                            "cursor": lab.store.state["cursor"]})

    @app.post("/api/undo")
    def api_undo():
        body = request.get_json(silent=True) or {}
        with lab.lock:
            lab.store.undo(int(body.get("index", -1)))
            return jsonify({"summary": lab.store.summary()})

    @app.post("/api/cursor")
    def api_cursor():
        body = request.get_json(silent=True) or {}
        with lab.lock:
            lab.store.set_cursor(int(body.get("cursor", 0)))
            return jsonify({"cursor": lab.store.state["cursor"]})

    @app.post("/api/export")
    def api_export():
        body = request.get_json(silent=True) or {}
        with lab.lock:
            try:
                res = export_session(lab.cfg, lab.store,
                                     write_h5=bool(body.get("write_h5")),
                                     h5_out=body.get("h5_out") or None,
                                     anchor_field=str(body.get("anchor_field", "anchor_label")))
            except Exception as exc:
                return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500
            return jsonify(res)

    @app.post("/api/save")
    def api_save():
        with lab.lock:
            lab.store.save()
            return jsonify({"saved": lab.store.path, "summary": lab.store.summary()})

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="Manual anchor labeler web app")
    ap.add_argument("--config", required=True, help="labeler yaml (see configs/labeler_*.yml)")
    ap.add_argument("--session", default=None, help="override session json path")
    ap.add_argument("--dataset", default=None,
                    help="override dataset.path (same field layout as the config)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8060)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.dataset:
        cfg.dataset.path = os.path.abspath(args.dataset)
    app = create_app(cfg, args.session)
    lab: Labeler = app.config["LABELER"]
    print(f"[labeler] config   : {args.config} ({cfg.name})")
    print(f"[labeler] dataset  : {cfg.dataset.path}  ({lab.dataset.n_events} events)")
    print(f"[labeler] session  : {lab.store.path}  (autosave {cfg.autosave_seconds}s + per-click tmp)")
    print(f"[labeler] open     : http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
