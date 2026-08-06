"""Cross-tab of cluster assignment vs particle_type on the 30 GeV eval set.
Reads the npz dumps from src.infer_eval (results/eval30/<tag>_30gev.npz) and
prints, per run, the electron/pion/muon -> cluster confusion restricted to
status==1 & non-anchor events. Identifies the electron cluster as the one that
best captures particle_type==0 and reports its recall & precision.

Usage (key4hep, CPU): python -m src.analyze_eval30 [results/eval30/*.npz]
"""
from __future__ import annotations
import glob, sys
import numpy as np

NAMES = {0: "e", 1: "pi", 2: "mu"}


def analyze(path):
    d = np.load(path)
    cl, pt = d["cluster"], d["particle_type"]
    anc = d["anchor"]; fs = d["filter_status"]; nh = d["nhits"]
    K = int(cl.max()) + 1
    m = (fs == 1) & (anc < 0) & np.isin(pt, [0, 1, 2])
    cl, pt, nh = cl[m], pt[m], nh[m]
    tag = path.split("/")[-1].replace("_30gev.npz", "")
    print(f"\n===== {tag}  (status==1, non-anchor, labeled: N={len(cl)}) =====")

    # confusion: rows = true particle, cols = cluster
    print("  true\\cluster " + "  ".join(f"c{k:>6}" for k in range(K)) + "   N")
    for p in [0, 1, 2]:
        sel = pt == p
        row = [int(((cl == k) & sel).sum()) for k in range(K)]
        frac = "  ".join(f"{r/max(sel.sum(),1):6.3f}" for r in row)
        print(f"  {NAMES[p]:>10}  {frac}   {int(sel.sum())}")

    # electron cluster = cluster with highest electron recall
    e_sel = pt == 0
    if e_sel.sum() == 0:
        print("  (no labeled electrons)"); return
    rec_by_c = [((cl == k) & e_sel).sum() / e_sel.sum() for k in range(K)]
    ec = int(np.argmax(rec_by_c))
    in_ec = cl == ec
    recall = ((in_ec) & e_sel).sum() / e_sel.sum()
    precision = ((in_ec) & e_sel).sum() / max(in_ec.sum(), 1)
    # what contaminates the e-cluster
    comp = {NAMES[p]: int(((cl == ec) & (pt == p)).sum()) for p in [0, 1, 2]}
    nh_ec = nh[in_ec]
    print(f"  -> electron cluster = c{ec}:  recall(e)={recall:.3f}  precision(e)={precision:.3f}")
    print(f"     e-cluster composition {comp}  | nHits med={np.median(nh_ec):.0f}  %<150={np.mean(nh_ec<150):.3f}")


if __name__ == "__main__":
    paths = sys.argv[1:] or sorted(glob.glob("results/eval30/*_30gev.npz"))
    if not paths:
        print("no npz found in results/eval30/"); sys.exit(1)
    for p in paths:
        analyze(p)
