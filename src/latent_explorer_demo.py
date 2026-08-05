#!/usr/bin/env python
"""
latent_explorer_demo.py  —  interactive explorer of the clustering latent space.

Runs OUTSIDE the Apptainer/GATr container: it needs NO PyTorch, GATr nor GPU,
only a plain Python env with:

    pip install dash plotly scipy numpy

The .npz it loads is produced by ``evaluate_clustering.py`` (inside the
container), e.g.:

    python -m src.evaluate_clustering --ckpt results/.../best-*.ckpt \\
        --data_path data/E70GeV_2016.h5 --out_dir results/eval
    # -> results/eval/latent_explorer.npz

Then, on a machine with a browser:

    python src/latent_explorer_demo.py --data results/eval/latent_explorer.npz
    # open http://localhost:8050

Left panel : 2D projection of the event latent z, colored by ASSIGNED CLUSTER;
             the learnable prototypes are co-embedded and overlaid as stars, and
             the (held-out) anchors are drawn as black-edged diamonds. Hover
             tells you whether a point is an anchor and its true anchor class.
             A radio switches between every projection stored in the npz --
             typically PCA and the prototype-plane projection (see
             src/projections.py for why t-SNE is a bad fit here and is off by
             default).
Right panel: the RAW 3D shower of the hovered event (color = threshold).
"""

import argparse
import os
import sys

import numpy as np
from scipy.spatial import KDTree
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State

# tab10, matching src/plots.py::cluster_colors so this view and the training
# PCA png agree on cluster -> color.
CLUSTER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

_DARK_BG = "#1a1a2e"
_CARD_BG = "#16213e"
_PLOT_BG = "#0f3460"
_TEXT = "#e0e0e0"
_MUTED = "#888888"
_ACCENT = "#e94560"

_PROTO_IDX = -1   # customdata sentinel: a prototype point (not a real event)

# Axis/menu labels come from src/projections.py when it is importable (running
# from the repo); the fallback keeps this file usable as a standalone script
# copied next to an npz.
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.projections import METHOD_AXES, METHOD_LABELS
except Exception:  # noqa: BLE001
    METHOD_AXES = {"pca": ("PC 1", "PC 2"),
                   "proto": ("prototype-plane 1", "prototype-plane 2"),
                   "tsne": ("t-SNE 1", "t-SNE 2"),
                   "umap": ("UMAP 1", "UMAP 2")}
    METHOD_LABELS = {"pca": "PCA (linear)", "proto": "Prototype plane",
                     "tsne": "t-SNE", "umap": "UMAP"}


def cluster_color(k):
    return CLUSTER_COLORS[int(k) % len(CLUSTER_COLORS)]


# ─── Data ─────────────────────────────────────────────────────────────────────

def load_data(npz_path):
    d = np.load(npz_path, allow_pickle=False)
    data = {k: d[k] for k in d.files}
    data["K"] = int(data["K"][0]) if "K" in data else int(data["cluster"].max() + 1)
    data["class_names"] = [str(s) for s in data["class_names"]]

    # Projections: new npz files carry emb2d_<method>/proto2d_<method> for each
    # method in emb_methods; older ones only have the single emb_2d/proto_2d
    # pair (whose method name is in emb_method). Normalize both into
    # data["projections"] = {method: {"emb": ..., "proto": ...}}.
    methods = [str(s) for s in data.get("emb_methods", [])]
    if not methods:
        methods = [str(data["emb_method"][0]) if "emb_method" in data else "pca"]
        data[f"emb2d_{methods[0]}"] = data["emb_2d"]
        data[f"proto2d_{methods[0]}"] = data.get("proto_2d")
    data["projections"] = {
        m: {"emb": data[f"emb2d_{m}"], "proto": data.get(f"proto2d_{m}")}
        for m in methods if f"emb2d_{m}" in data
    }
    data["methods"] = list(data["projections"])
    return data


def method_label(m):
    return METHOD_LABELS.get(m, m)


def method_axes(m):
    return METHOD_AXES.get(m, ("dim 1", "dim 2"))


def cluster_name(data, k):
    names = data["class_names"]
    return names[k] if 0 <= k < len(names) else f"cluster {k}"


def get_shower(data, idx):
    """(xyz, thr, n) for event idx from the padded raw-hit arrays."""
    n = int(data["hits_len"][idx])
    xyz = data["hits_xyz"][idx, :n]
    thr = data["hits_thr"][idx, :n]
    return xyz, thr, n


def event_info(data, idx):
    k = int(data["cluster"][idx])
    a = int(data["anchor"][idx])
    e = float(data["energies"][idx])
    n = int(data["hits_len"][idx])
    anc = f"anchor · true={cluster_name(data, a)}" if a >= 0 else "not an anchor"
    e_txt = f"E={e:.1f} GeV  ·  " if e > 0 else ""
    return f"Event {idx}  |  cluster={cluster_name(data, k)}  |  {anc}  |  {e_txt}N_hits={n}"


# ─── Figures ──────────────────────────────────────────────────────────────────

_SCATTER_LAYOUT = dict(
    paper_bgcolor=_DARK_BG,
    plot_bgcolor=_PLOT_BG,
    font=dict(color=_TEXT, size=11),
    margin=dict(l=40, r=20, t=30, b=40),
    hovermode="closest",
    height=520,
)


def make_fig_2d(data, method=None):
    method = method or data["methods"][0]
    proj = data["projections"][method]
    pca = proj["emb"]
    xlab, ylab = method_axes(method)
    cluster = data["cluster"]
    anchor = data["anchor"]
    is_anchor = data["is_anchor"].astype(bool)
    K = data["K"]

    traces = []
    # non-anchor events, one trace per cluster (so the legend names each cluster)
    for k in range(K):
        m = (cluster == k) & (~is_anchor)
        if not m.any():
            continue
        idx = np.flatnonzero(m)
        traces.append(go.Scatter(
            x=pca[m, 0], y=pca[m, 1], mode="markers",
            name=cluster_name(data, k),
            marker=dict(size=7, color=cluster_color(k), opacity=0.65, line=dict(width=0)),
            customdata=idx.reshape(-1, 1),
            hovertemplate=("%{customdata[0]}<br>"
                           f"{method_label(method)}"
                           " (%{x:.2f}, %{y:.2f})"
                           f"<br>cluster={cluster_name(data, k)}<extra></extra>"),
            legendgroup=f"c{k}",
        ))

    # anchors: black-edged diamonds, colored by ASSIGNED cluster
    am = np.flatnonzero(is_anchor)
    if am.size:
        traces.append(go.Scatter(
            x=pca[am, 0], y=pca[am, 1], mode="markers",
            name="anchors",
            marker=dict(
                size=11, symbol="diamond",
                color=[cluster_color(cluster[i]) for i in am],
                line=dict(width=1.3, color="black"),
            ),
            customdata=np.stack([am, anchor[am]], axis=1),
            hovertemplate=("%{customdata[0]}  ANCHOR"
                           "<br>true class idx=%{customdata[1]}<extra></extra>"),
        ))

    # prototypes as stars (per cluster color), not linked to any event
    proto = proj.get("proto")
    if proto is not None and len(proto):
        traces.append(go.Scatter(
            x=proto[:, 0], y=proto[:, 1], mode="markers",
            name="prototypes",
            marker=dict(size=20, symbol="star",
                        color=[cluster_color(k) for k in range(len(proto))],
                        line=dict(width=1.4, color="black")),
            customdata=np.full((len(proto), 1), _PROTO_IDX),
            hovertemplate="prototype %{pointNumber}<extra></extra>",
        ))

    layout = go.Layout(
        **_SCATTER_LAYOUT,
        xaxis=dict(title=xlab, gridcolor="#224", zerolinecolor="#446"),
        yaxis=dict(title=ylab, gridcolor="#224", zerolinecolor="#446"),
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        # per-method uirevision: keep zoom while hovering, reset it when the
        # projection changes (the old viewport means nothing in the new map).
        uirevision=f"scatter2d-{method}",
    )
    return go.Figure(data=traces, layout=layout)


def make_fig_3d(xyz, thr, n, title=""):
    if xyz is None or n == 0:
        return go.Figure(layout=go.Layout(
            paper_bgcolor=_DARK_BG, scene=dict(bgcolor=_DARK_BG),
            font=dict(color=_MUTED), margin=dict(l=0, r=0, t=30, b=0), height=520,
            title=dict(text="← Hover over a point", font=dict(color=_MUTED, size=12)),
        ))

    scatter3d = go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
        marker=dict(
            size=3.5, color=thr, colorscale="RdYlBu_r",
            cmin=1, cmax=3,
            colorbar=dict(title=dict(text="thr", font=dict(size=9, color=_TEXT)),
                          thickness=12, len=0.55, tickfont=dict(size=8, color=_TEXT)),
            opacity=0.85, line=dict(width=0),
        ),
        hovertemplate="x=%{x:.1f} y=%{y:.1f} z=%{z:.1f}<extra></extra>",
    )
    axis_style = dict(backgroundcolor=_PLOT_BG, gridcolor="#334",
                      showbackground=True, tickfont=dict(size=9, color=_TEXT))
    layout = go.Layout(
        paper_bgcolor=_DARK_BG,
        scene=dict(xaxis=dict(title="x", **axis_style),
                   yaxis=dict(title="y", **axis_style),
                   zaxis=dict(title="z (depth)", **axis_style),
                   bgcolor=_PLOT_BG, aspectmode="data"),
        title=dict(text=title, font=dict(size=12, color=_TEXT)),
        margin=dict(l=0, r=0, t=35, b=0), height=520,
        uirevision="shower3d",   # keep camera across hover updates
    )
    return go.Figure(data=[scatter3d], layout=layout)


# ─── App ──────────────────────────────────────────────────────────────────────

def build_app(data):
    methods = data["methods"]
    default_method = methods[0]
    # one KDTree per projection: hovering a prototype falls back to the nearest
    # real event, and "nearest" depends on which map is on screen.
    trees = {m: KDTree(data["projections"][m]["emb"]) for m in methods}
    n_ev = len(data["cluster"])
    n_anc = int(data["is_anchor"].sum())

    card = {"backgroundColor": _CARD_BG, "border": "1px solid #334",
            "borderRadius": "8px", "padding": "12px"}

    app = dash.Dash(__name__)
    app.title = "SDHCAL Clustering · Latent Explorer"
    app.layout = html.Div(
        style={"backgroundColor": _DARK_BG, "minHeight": "100vh",
               "fontFamily": "monospace", "padding": "16px"},
        children=[
            html.H4("SDHCAL · Clustering Latent Explorer",
                    style={"color": _ACCENT, "letterSpacing": "3px",
                           "textAlign": "center", "marginBottom": "4px"}),
            html.P(f"{n_ev} events · {n_anc} anchors · 2D projection of latent z",
                   style={"color": _MUTED, "fontSize": "11px",
                          "textAlign": "center", "marginBottom": "16px"}),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
                children=[
                    html.Div(style=card, children=[
                        html.Div(id="scatter-2d-title",
                                 children=(f"Latent space ({method_label(default_method)}) "
                                           "· color = assigned cluster"),
                                 style={"color": _MUTED, "fontSize": "11px", "marginBottom": "6px"}),
                        dcc.RadioItems(
                            id="proj-method",
                            options=[{"label": method_label(m), "value": m} for m in methods],
                            value=default_method,
                            inline=True,
                            style={"color": _TEXT, "fontSize": "11px", "marginBottom": "6px"},
                            inputStyle={"marginRight": "4px", "marginLeft": "10px"},
                        ),
                        dcc.Graph(id="scatter-2d", figure=make_fig_2d(data, default_method),
                                  config={"displayModeBar": False}),
                    ]),
                    html.Div(style=card, children=[
                        html.Div("Hadronic shower (raw 3D hits)",
                                 style={"color": _MUTED, "fontSize": "11px", "marginBottom": "6px"}),
                        dcc.Graph(id="scatter-3d", figure=make_fig_3d(None, None, 0),
                                  config={"displayModeBar": True}),
                        html.Div(id="event-info",
                                 style={"color": _MUTED, "fontSize": "11px",
                                        "marginTop": "6px", "minHeight": "18px"}),
                    ]),
                ],
            ),
        ],
    )

    @app.callback(
        Output("scatter-2d", "figure"),
        Output("scatter-2d-title", "children"),
        Input("proj-method", "value"),
    )
    def on_method(method):
        method = method or default_method
        return (make_fig_2d(data, method),
                f"Latent space ({method_label(method)}) · color = assigned cluster")

    @app.callback(
        Output("scatter-3d", "figure"),
        Output("event-info", "children"),
        Input("scatter-2d", "hoverData"),
        Input("scatter-2d", "clickData"),
        State("proj-method", "value"),
        prevent_initial_call=True,
    )
    def on_hover(hover_data, click_data, method):
        ev = hover_data or click_data      # click fallback (hover flaky over SSH)
        if ev is None:
            return dash.no_update, dash.no_update
        pt = ev["points"][0]
        idx = int(pt.get("customdata", [_PROTO_IDX])[0]) if "customdata" in pt else _PROTO_IDX
        if idx == _PROTO_IDX:
            # hovered a prototype: fall back to the nearest real event in the
            # projection currently on screen
            tree = trees.get(method or default_method, trees[default_method])
            _, idx = tree.query([float(pt["x"]), float(pt["y"])])
            idx = int(idx)
        xyz, thr, n = get_shower(data, idx)
        k = int(data["cluster"][idx])
        a = int(data["anchor"][idx])
        tag = " · ANCHOR" if a >= 0 else ""
        title = f"{cluster_name(data, k)}  ·  {n} hits{tag}"
        return make_fig_3d(xyz, thr, n, title), event_info(data, idx)

    return app


def parse_args():
    p = argparse.ArgumentParser(description="Interactive clustering latent explorer (no torch).")
    p.add_argument("--data", default="results/eval/latent_explorer.npz")
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Loading {args.data} ...")
    data = load_data(args.data)
    print(f"  {len(data['cluster'])} events, {int(data['is_anchor'].sum())} anchors, "
          f"K={data['K']}, projections: {', '.join(data['methods'])}")
    app = build_app(data)
    print(f"\nServer at  http://{args.host}:{args.port}   (Ctrl+C to stop)\n")
    app.run(debug=args.debug, port=args.port, host=args.host)


if __name__ == "__main__":
    main()
