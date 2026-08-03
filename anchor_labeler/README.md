# Anchor labeler

Web app to curate anchors **by hand**. It takes the current automatic method
(`src/convert/mark_anchors.py`, `nhits_sampling`), shows each proposed event as
a rotatable 3D shower, and lets you confirm it, relabel it or discard it. Only
what you confirm or relabel is exported.

Motivation: the automatic anchors are only as good as the flag that tagged the
event, and that flag is not the labeled particle — mislabeled anchors poison the
CE term. Reviewing them one by one removes that failure mode.

## Install & run

```bash
bash install_labeler.sh          # once: flask + h5py + numpy + pyyaml + plotly in .venv_viewer
bash run_labeler.sh              # http://127.0.0.1:8060
bash run_labeler.sh --config configs/labeler_other.yml --port 8070
```

Working over ssh? Forward the port from your laptop and open it locally:

```bash
ssh -L 8060:localhost:8060 <user>@<host>
```

No internet access is needed: `plotly.min.js` is served from the installed
python package, not from a CDN.

## Reviewing

The centre panel shows the event in 3D with the **beam axis pointing right**
(`dataset.beam_axis`, free rotation with the mouse, `r` recentres). Hits are
coloured by `dataset.hit_color_field` (threshold for SDHCAL).

| key | action |
|-----|--------|
| `Enter` / `Space` | confirm the proposed class and advance |
| class key (`0`,`1`,`2`, …) | keep the event with **that** class instead |
| `i` | ignore — rejected for good, never proposed again |
| `s` | skip — undecided, may come back in a later round |
| `←` / `→` | previous / next event |
| `u` | undo the decision on the current event |
| `r` | reset the 3D camera |

The left panel edits the sampling parameters (`strategy`, `n_per_class`,
`seed`, `window_std`, prefer-unseen, which classes to draw) and launches a new
round. A new round **replaces** the queue unless *añadir a la cola actual* is
ticked. Re-sampling can bring back events you have not decided on (skipped or
never shown), but **ignored events never return** and kept anchors are not
proposed twice.

`Exportar anchors` writes, into `export_dir`:

* `anchors_<stamp>.json` — full record (label, corrected or confirmed, round)
* `anchors_<stamp>.csv`
* `anchors_<stamp>.yml` — `mode: manual` config consumable by
  `python -m src.convert.mark_anchors --h5 <file> --config <this yml>`
* optionally a **copy of the dataset** with `anchor_label` filled in (-1
  elsewhere), ready for the training configs.

## State is never lost

* `<session>.json.tmp` is rewritten on **every single decision**.
* `<session>.json` is written atomically (tmp + `os.replace`) every
  `autosave_seconds` (default 5 s) and on export/exit.
* On startup the newer of the two is loaded, so a crash costs at most the click
  in flight. Restarting the app resumes the queue, the cursor and all decisions.

## Adding a different detector / dataset

Nothing is hardcoded: write a new `configs/labeler_*.yml` describing the file,
the field names, the classes and how candidates are proposed
(`flag` / `value` / `range` / `all`). If the data is not a flat/CSR HDF5, add a
class with `n_events`, `nhits()`, `mask_for_source()` and `event(i)` to
`datasets.py::BACKENDS`; a new proposal rule is one function in
`sampler.py::STRATEGIES`. The UI reads classes, colours, axes and parameters
from the server, so it needs no change.

## Layout

```
anchor_labeler/
  config.py     yaml -> typed config (dataset, classes, sampling)
  datasets.py   backends; hdf5_flat reads the CSR layout event by event
  sampler.py    proposal strategies (nhits_band = current method, random)
  session.py    decisions + atomic autosave / crash recovery
  export.py     json / csv / mark_anchors yml / h5 with anchor_label
  server.py     Flask API (/api/state, /sample, /event, /decision, /export)
  templates/, static/   single-page UI (plotly.js 3D)
```
