# GATrClustering

Unsupervised, anchor-guided clustering of SDHCAL shower events (e/π/μ) on graphs
of hits, with a **GATr** backbone. Forked from `../GATrAutoencoder`.

The whole pipeline is **modular and parametric** (see `configs/poc.yml`): feature
routing (multivector point / geometric scalar / GATr scalar input), per-feature
scaling, and switchable aggregation — so switching detector/conditions means
editing the YAML, not the code.

## Layout

```
configs/poc.yml                 experiment config (data + features + scaling + model + loss + train)
src/data/flat_h5_reader.py      fixed CSR contract + remappable field_map
src/data/scaling.py             per-feature z_norm/minmax/log/none; online/file/dataset stats
src/data/dataset.py             PyG dataset + make_clustering_splits (train-only stats)
src/data/feature_routing.py     point / geometric-scalar / GATr-scalar routing; thr ordinal|one_hot
src/models/gatr_module.py       forked GATr encoder (generalized geometric-scalar embedding)
src/models/attention_pooling.py forked PMA pooling
src/models/aggregation.py       mean | attention | token (asymmetric aggregation token)
src/models/prototype_head.py    projection to z + cosine logits vs learnable prototypes
src/models/clustering_model.py  encoder -> aggregation -> prototype head
src/losses/swap_loss.py         SwAV-style swap loss (no Sinkhorn)
src/losses/vicreg.py            variance + covariance regularization
src/augment.py                  per-hit dropout (two views)
src/train_clustering.py         training + anchor-based prototype init
src/evaluate_clustering.py      nHits-by-cluster (primary), t-SNE, confusion
```

## Data contract

Flat CSR HDF5 (or npz): an `offsets` array (`len n_events+1`), per-hit arrays
(`len n_hits`) and per-event arrays (`len n_events`). Branch names are remapped
via `data.field_map` in the config. Anchors are a per-event field
(`anchor_label`: `-1` = no anchor, `0..K-1` = class) that the user marks.

### Simulation-as-anchors (multi-dataset)

Instead of marking anchors inside one file, you can supply anchors from separate
files (e.g. **simulation** as labels, **test-beam** fully unlabeled). See
`configs/sim_anchors.yml`:

- `data.path` is the primary (unlabeled) file; `data.force_anchor: -1` blanks its
  own `anchor_label` field so it enters training with no labels.
- `data.anchor_datasets` is a list of extra files, each with an `anchor_label`
  (one particle class per file, applied to all its events), an optional
  per-source `field_map` override (defaults to `data.field_map` — same hit keys),
  and an optional `max_events` (reproducible subsample cap to balance anchors).

All files are concatenated into one dataset (`MultiFlatEventReader`); scaling
stats are computed over the combined train split. Everything downstream (splits,
CE, prototype init, held-out anchor accuracy, evaluation) is unchanged.

### Anchor strategy: dedicated e-beam wins once the thr swap is fixed (s9)

The winning configuration is **`configs/sim_anchors_s9.yml`**: on the 2012
test-beam primary, the π/μ anchors are the primary's own hand-marked 75 π + 75 μ
(`src/convert/mark_anchors.py`, **no simulation**), and the **electron** anchor is
the dedicated 70 GeV electron-beam TB run (`data/El70GeV_2012_TBanchors_v2_thrfix.h5`,
500 ev) **with the `thr1`/`thr2` swap corrected** (that swap was the hidden driver
of the old latent island — see below). Run it via `condor/train.sh` (forces
`--data_path data/E70GeV_2012.h5`).

Its predecessor **`configs/sim_anchors_s6.yml`** used 129 **in-domain** electron
showers extracted from the primary instead (depth-purified EM:
`is_electron & nHits∈[250,900) & max-k≤38`, also no simulation). s6 was the best
recipe until s9; the 3-seed comparison below explains why s9 supersedes it.

This was reached by elimination; each earlier variant failed a specific way:

| Symptom | Root cause | Fix |
|---|---|---|
| electron ≡ muon collapse | attention/mean pooling averages hits → blind to nHits | `aggregation.type: attention_card` (sum + `log1p(nHits)` branch) |
| electron cluster stuck ~0% | dedicated e-beam anchors overfit into an isolated latent island TB electrons never reach | in-domain electron anchors from the primary (can't fly off) |
| overfitting (val/loss ≫ train) | out-of-domain **simulation** π/μ anchors → model learns "sim vs TB" shortcut | drop sim anchors; go fully in-domain (s6) |

Result (s6 vs the sim-anchor s5): `val/loss` 2.94 → **0.38**, held-out anchor
acc 0.50 → 0.97, cluster proportions `[e,π,μ] = [0.02, 0.10, 0.88]` matching the
true 2012 flag composition, and three clean clusters with correct physics
(μ ~70 hits, π ~1000, e in the compact intermediate band).

**Why s9 beats s6 — a precision↔recall trade, both characterized across 3 seeds**
(`sim_anchors_s6{,b,c}` and `sim_anchors_s9{,b,c}`, seeds 42/123/777). In both
recipes the π/μ clusters are perfectly resolved every seed (π med ~950–1050, μ
med ~76, proportions matching the true 2012 flag composition ~[0.02, 0.08, 0.89]).
All the difficulty lives in the electron minority wedged between the μ (low-nHits)
and π (high-nHits) clouds. Electron cluster (c0) per seed:

| seed | s6 recall / %<150 / nHits med | s9 recall / %<150 / nHits med |
|---|---|---|
| 42  | 1.00 / 0.20 / 436 | 0.85 / 0.14 / 510 |
| 123 | 1.00 / 0.41 / 261 | 0.84 / 0.45 / 441 |
| 777 | 1.00 / 0.71 / **45** ⚠ | 0.80 / **0.00** / 532 |

- **s6 (in-domain e-anchors): perfect recall, fragile precision.** Recalls real
  EM showers 100% every seed, but the loosely-pinned electron prototype drifts by
  seed and leaks the low-nHits muon tail — seed 777 **collapsed** (cluster 0
  became a muon-tail dump, nHits med 45, 71% below 150). `val/heldout_anchor_acc`
  swings 0.97 / 0.59 / 0.34.
- **s9 (dedicated e-beam, thr-swap fixed): robust precision, ~15–20% recall loss.**
  The tight, self-similar dedicated-beam anchor blob **pins** the prototype so it
  never drifts into the muon cloud — nHits med stays 441–532 (real compact
  showers) and `%<150` stays clean (0.00–0.45) **every seed**; s9 seed 777 is the
  cleanest single electron cluster in the project (0% contamination). The price is
  recall ≈ 0.80–0.85: being beam-specific, it misses the ~15–20% of the primary's
  EM showers that differ from the beam signature. `val/heldout_anchor_acc`
  0.99 / 1.00 / 0.99 (flat).

**Neither dominates, but s9 is the better default:** for a *pure* electron sample
(physics: purity ≫ completeness) s9's robust ~85%-complete clean cluster wins over
s6's complete-but-variably-contaminated one. s9 is reproducible; s6 seed 42 was a
lucky draw.

**s10 — density pooling (`configs/sim_anchors_s10{,b,c}.yml`, seeds 42/123/777).**
The pooling scalar is swapped from `log1p(nHits)` to **DENSITY = nHits /
n_active_layers** (`aggregation.type: attention_density`), to make the event
descriptor agnostic to detector depth (a muon spreads few hits over many layers →
low density; a compact EM shower → high density).

> ⚠️ **CORRECTION — the table below was measured with anchor events counted, and
> is therefore misleading. Struck through, kept visible so the mistake is on
> record.** The eval `.npz` is the *validation split of the full concatenated
> dataset* (primary 2012 **+** the dedicated e-beam anchor file), and the metric
> script did **not** exclude `is_anchor` events. Of the ~53 "true EM" events it
> scored, **~38–41 were the electron anchors themselves** (guaranteed electrons
> pinned to the prototype) — only **~12–15 were real 2012 electrons**. See the
> honest, anchor-excluded numbers further down.

> ~~3-seed comparison (seeds 42/123/777, real padded-3D EM tag):~~
> ~~| seed | s9 recall / %<150 | s10 recall / %<150 / precision |~~
> ~~| 42   | 0.85 / 0.136 | 0.91 / 0.014 / 0.69 |~~
> ~~| 123  | 0.84 / ~0.45 | 0.77 / 0.047 / 0.69 |~~
> ~~| 777  | 0.80 / 0.00  | 0.79 / 0.000 / 0.98 |~~
> ~~- Muon leakage is eliminated in every seed (%<150 0.00–0.047 vs s9's up to 0.45).~~
> ~~- Recall is preserved (~0.82 mean vs s9's 0.83).~~
> ~~- The pion dip in s10 seed 42 (0.084→0.047) was seed noise; s10b/s10c match s9.~~
> ~~Density is a net win… `sim_anchors_s10` is the current best recipe.~~

**Honest metrics, anchor events EXCLUDED** (val split, real 2012 only, ~12–15
true-EM candidates — *small-N, per-seed recall is noisy*):

| seed | s9 recall / %<150 | s10 recall / %<150 | e-cluster share of real data |
|------|-------------------|--------------------|------------------------------|
| 42   | 0.47 / 0.35       | **0.67** / 0.03    | ~1.0 % |
| 123  | —                 | 0.07 / 0.15        | ~0.7 % |
| 777  | —                 | 0.08 / 0.00        | ~0.0 % |

What actually holds up once anchors are removed:
- **This TB sample has very few real electrons** (~12–15 EM in a 2944-event val
  split; it is largely a pion beam), so the electron cluster is genuinely tiny
  (0–1 % of real events) and per-seed recall is a small-count statistic.
- **We DO isolate some of the important ones**: s10 seed 42 recovers ~2/3 of the
  real EM candidates into a clean cluster (`%<150 = 0.03`, i.e. almost no muon
  contamination) — a real, if small, win. Density still lowers muon leakage vs s9
  (real `%<150` 0.35 → 0.03).
- The earlier "robust across all seeds" claim was an **anchor artifact**: s10b/s10c
  recover almost no *real* primary electrons (recall ~0.07–0.08); they mostly just
  pin their own anchors.
- **Next: evaluate on the FULL primary (domain==0, anchor file excluded)** for
  ~60–75 real EM candidates and trustworthy statistics before ranking s9 vs s10.

Two reweighting attempts to fix the s6 fragility *without* the dedicated anchors
both failed, which is why the fix came from the anchor **source**, not a loss lever:
- **s7** (`sim_anchors_s7.yml`, more electron anchors 129→184 + electron CE
  upweight `[3,1,1]`): **did not help** — acts on the anchor loss, but recall was
  already 100%; precision lives in the unlabeled-assignment term.
- **s8** (`sim_anchors_s8.yml`, prior-side): tighter electron prior (0.02→0.012),
  `lambda_prior` 2→4. **Worse** — `%<150` 0.20→0.69 and recall dropped 1.00→0.90.

Both reweighting levers fail for the same **geometric** reason: the latent space is
a *continuous* muon(low-nHits)→electron(compact shower) manifold with no gap, so
any argmax boundary cuts a populated bridge. s9 sidesteps it by pinning the
prototype off the bridge with a tight anchor, rather than trying to move the cut.
A structural alternative remains open (**two-stage clustering**: MIP-vs-shower
first, then e-vs-π within showers) if the ~15% recall loss ever needs recovering.

Note: the cardinality signal is already in the model (`attention_card` pooling
concatenates `log1p(nHits)`); the fragility was an under-constrained minority
prototype on a continuous manifold, not a missing feature — so re-injecting nHits
as an explicit input scalar is deliberately avoided (topography should be learned
from geometry).

> The dedicated 2012 electron-beam TB file had `thr1`/`thr2` swapped (never fixed
> in preprocessing). Because `thr` is a model feature (`mv_scalar:[thr]`), that
> inverted signature was a give-away the encoder used to fly the anchors into an
> isolated latent island (the original reason it was abandoned). **s9 fixes the
> swap** (`scratchpad/fix_thr.py` → `El70GeV_2012_TBanchors_v2_thrfix.h5`,
> `{1:0.70, 2:0.22, 3:0.08}` matching the primary) — the island is gone and the
> dedicated anchors become the most *stable* anchor source we have.

> Metric gotcha: `hits_xyz` in `eval/latent_explorer.npz` is **padded 3D**
> `[N, maxhits, 3]`, not flat-CSR — index event `i` as
> `hits_xyz[i, :hits_len[i], 2]`. The real EM tag used for recall is
> `nHits∈[250,900] & k95≤20` with `k95 = pctile((z−226.5)/28, 95)`.

## Run (inside the Apptainer image)

```bash
apptainer exec --nv -B /nfs/cms/arqolmo/GPU_train/mlpf/extlib \
  -B /nfs:/nfs --pwd "$PWD" /nfs/cms/arqolmo/GPU_train/mlpf/gatr_v9.sif bash

# train  (W&B on by default; logs losses, lr, held-out anchor acc, latent PCA)
export WANDB_API_KEY=...        # or:  --wandb_mode offline  /  --no_wandb
python -m src.train_clustering --cfg configs/poc.yml --data_path /path/to/dataset.h5

# evaluate  (also writes results/eval/latent_explorer.npz for the explorer below)
python -m src.evaluate_clustering --ckpt results/poc_run1/last.ckpt \
  --data_path /path/to/dataset.h5 --out_dir results/eval
```

## Interactive latent explorer

`evaluate_clustering.py` dumps `latent_explorer.npz` (t-SNE of the event latent
`z` + co-embedded prototypes, plus the raw hits of each event). The viewer is a
small Dash app that runs **outside** the container — no PyTorch/GATr/GPU, just:

```bash
pip install dash plotly scipy numpy         # a plain Python env, not the .sif
python src/latent_explorer_demo.py --data results/eval/latent_explorer.npz
# open http://localhost:8050
```

Left: 2D t-SNE colored by assigned cluster, prototypes as stars, held-out
anchors as black-edged diamonds (hover shows anchor flag + true class).
Right: the raw 3D shower of the hovered event (color = threshold).

## SiW-ECAL TB2026 (separate detector line)

Second detector, **kept strictly separate from all SDHCAL work**: configs live
under `configs/siwecal/`, converted data under
`/nfs/cms/arqolmo/SiWECALTB2026/Converted/`, labeler ports **8060+** (8050–8052
are SDHCAL), and labeler sessions/exports carry a `siwecal_` prefix. Class
indices keep the SDHCAL convention (0=e, 1=π, 2=μ, 3=multip, 4=ruido) so labels
never need remapping.

Source files are the `*.valtree.root` under `/nfs/cms/arqolmo/SiWECALTB2026/Data/`
(tree `ecal`; the two ROOT cycles are autosave snapshots, uproot picks the
latest). Format notes, all encoded in `configs/siwecal/export_valtree.yml`:

- 15 slabs, `hit_slab` already 0-based → logical `k`. **z is inverted**: the
  beam enters at slab 0 = z=0 mm and depth grows toward *negative* z
  (slab 14 = −225 mm), with a 30 mm gap between slabs 10 and 11.
- x, y in mm (±86.325, 32×32 pads of 5.53 mm) — no `hit_affine` needed for
  display; whether training shares a frame with SDHCAL is a later decision.
- Per-hit amplitude is **analog** (`e` = `hit_energy` in MIPs, 0.5 cut); there
  is no SDHCAL-style `thr`, so `normalize_thr: false` and the future training
  feature map must route `e` instead of `thr`.
- Rich per-event reco scalars are exported (`nHits_total`, `energy_sum`,
  `energy_reco`, `mip_likeness`, `is_shower`, `shower_start/max/length`, …);
  with no truth labels in TB data, these drive the labeler's proposal pools.

### 1. Convert ROOT → flat CSR h5

```bash
source /cvmfs/sw.hsf.org/key4hep/setup.sh    # provides uproot
python -m src.convert.root_to_flat_h5 \
    --config configs/siwecal/export_valtree.yml \
    --inputs /nfs/cms/arqolmo/SiWECALTB2026/Data/ecal_TB2026CERN_run_000044.valtree.root \
    --output /nfs/cms/arqolmo/SiWECALTB2026/Converted/run44_34GeV.h5
```

No converter code changes were needed — the config remaps everything. For a new
run, copy the command with the new file; if branch names differ, only the YAML
changes. `anchor_label` is written as −1 (labels come from the labeler).

### 2. Label anchors

```bash
bash run_labeler.sh --config configs/siwecal/labeler_run44.yml --port 8060
```

The run-44 labeler colors hits by analog `e` (continuous colorbar) and proposes
from reco discriminants: `is_shower` → electron pool (99.3% of the 34 GeV run —
electron-dominated beam), `mip_likeness > 0.30` → muon pool (only 379 events).
Pion/multip/ruido are manual-only until we know what the run contains. For a new
run, copy `labeler_run44.yml`, update `dataset.path`, `session_path` and the
`name`, and pick a fresh 806x port.

### 3. (Pending) Training adaptations

Before training on SiW-ECAL: route `e` (not `thr`) in `data.field_map` /
`feature_routing`, and revisit depth-dependent pieces (density pooling divides
by active layers — only 15 here; z sign is inverted vs SDHCAL).
