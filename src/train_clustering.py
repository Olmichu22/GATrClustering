"""
Training entry point for the clustering POC (PyTorch Lightning).

    L = loss.swap.weight * L_swap(two dropout views)
      + lambda_ce        * CE(anchors)
      + lambda_vicreg    * (var + cov)(z)

Prototypes are initialized (non-random) from the mean anchor embedding of each
class before the loop (Lightning `on_fit_start`). lambda_ce is high at the start
(anchors are the most reliable signal). No Sinkhorn / no proportion prior in this
first iteration.

Checkpointing (per run, never overwritten across jobs):
    <out_dir>/<run_tag>/checkpoints/
      - last.ckpt                                  (updated via save_last)
      - best-epoch<NNN>-acc<ACC>.ckpt              (one file per improving epoch)
`run_tag` is a timestamp (%Y%m%d_%H%M%S), optionally suffixed with the W&B run
name. The full config lives INSIDE every checkpoint under `hyper_parameters`
(via `save_hyperparameters`). Because the model is stored as `self.model`, its
weights appear in `state_dict` with the `model.` prefix (e.g.
`model.head.prototypes`, `model.encoder...`).

Run inside the Apptainer image gatr_v9.sif (see GATrAutoencoder/CLAUDE.md).
"""

from __future__ import annotations

import argparse
import glob
import os
from datetime import datetime

import numpy as np
import torch

# H200/A100 tensor cores: run fp32 matmuls in TF32. With bf16-mixed this only
# affects the ops autocast keeps in fp32; "high" is the safe precision point.
torch.set_float32_matmul_precision("high")

import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch_geometric.loader import DataLoader

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import lightning as L  # noqa: E402
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint  # noqa: E402


class ResumableEarlyStopping(EarlyStopping):
    """EarlyStopping that keeps `patience` from the current config on resume.

    Lightning 2.3.x's EarlyStopping.load_state_dict restores `patience` from the
    checkpoint, so bumping `early_stop_patience` in the YAML would be ignored when
    resuming (and a run stopped by patience would immediately stop again). We keep
    the freshly-constructed patience and only restore wait_count/best_score.
    """

    def load_state_dict(self, state_dict: dict) -> None:
        keep_patience = self.patience
        super().load_state_dict(state_dict)
        self.patience = keep_patience
from lightning.pytorch.loggers import WandbLogger  # noqa: E402
from lightning.pytorch.strategies import DDPStrategy  # noqa: E402

from .augment import hit_dropout  # noqa: E402
from .data.dataset import make_clustering_splits  # noqa: E402
from .losses.swap_loss import swap_loss  # noqa: E402
from .losses.vicreg import vicreg_loss  # noqa: E402
from .losses.prior_loss import prior_kl_loss  # noqa: E402
from .losses.domain_align import DomainAlignLoss  # noqa: E402
from .models.clustering_model import ClusteringModel  # noqa: E402
from .plots import plot_latent_pca  # noqa: E402

try:
    import wandb
except ImportError:
    wandb = None


def load_config(path: str) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


class ClusteringLitModule(L.LightningModule):
    """LightningModule wrapping ClusteringModel with the swap+CE+VICReg objective.

    The full config is saved as hyper_parameters (goes inside the checkpoint).
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters(cfg)
        # Keep a plain-dict view of the config for convenient access.
        self.cfg = cfg

        self.model = ClusteringModel(cfg["model"], cfg["features"])

        # ----- static objective config -----
        self.p_drop = cfg["augment"]["hit_dropout"]
        lcfg = cfg["loss"]
        self.swap_w = lcfg["swap"]["weight"]
        self.sharpen = lcfg["swap"].get("sharpen_temp", 0.25)
        self.lam_ce = lcfg["lambda_ce"]
        self.lam_vic = lcfg["lambda_vicreg"]
        self.vcfg = lcfg["vicreg"]

        # ----- cluster-proportion prior (anti-collapse), opt-in ---------------
        # Ties the batch-mean soft assignment p_bar to a target proportion via
        # KL(prior || p_bar). prior=null or lambda_prior<=0 disables it entirely.
        # Optional EMA smooths p_bar across batches (helps when a target cluster
        # is rare vs the batch size); the EMA buffer is non-persistent (resets on
        # resume, warms up fast) so enabling the prior never breaks old ckpts.
        K = cfg["model"]["head"]["num_clusters"]
        prior = lcfg.get("cluster_prior", None)
        self.lam_prior = float(lcfg.get("lambda_prior", 0.0) or 0.0)
        self.prior_enabled = prior is not None and self.lam_prior > 0.0
        self.prior_ema = bool(lcfg.get("prior_ema", False))
        self.prior_ema_momentum = float(lcfg.get("prior_ema_momentum", 0.9))
        # The prior describes the mix of the UNLABELED population. Anchors are
        # hand-picked / simulation (typically balanced across classes), so counting
        # them in p_bar biases it toward uniform and fights the prior. Exclude them.
        self.prior_exclude_anchors = bool(lcfg.get("prior_exclude_anchors", True))
        if self.prior_enabled:
            p = torch.as_tensor(prior, dtype=torch.float32)
            if p.numel() != K:
                raise ValueError(
                    f"loss.cluster_prior has {p.numel()} entries but "
                    f"model.head.num_clusters={K}"
                )
            self.register_buffer("cluster_prior", p / p.sum(), persistent=False)
            self.register_buffer("pbar_ema", torch.full((K,), 1.0 / K), persistent=False)
        # Optional linear warmup [e0,e1] for the prior weight (None -> always on).
        pw = lcfg.get("prior_warmup_epochs", None)
        self.prior_warmup = (int(pw[0]), int(pw[1])) if pw is not None else None

        # ----- class-conditional domain alignment (sim -> TB anchors), opt-in --
        # Pulls the per-class mean of the sim-anchor embeddings toward a stop-grad
        # EMA of the TB-anchor mean of the same class (see losses/domain_align.py).
        # Absent block or enabled: false -> nothing is constructed (zero behavior
        # change; old configs and checkpoints are unaffected).
        dacfg = lcfg.get("domain_align") or {}
        self.da_enabled = bool(dacfg.get("enabled", False))
        if self.da_enabled:
            self.lam_da = float(dacfg.get("lambda", 1.0))
            w = dacfg.get("warmup_epochs", [5, 15])
            self.da_warmup = (int(w[0]), int(w[1]))
            self.domain_align = DomainAlignLoss(
                num_classes=K,
                dim=int(cfg["model"]["head"]["proj_dim"]),
                ema_momentum=float(dacfg.get("ema_momentum", 0.9)),
            )

        tcfg = cfg["train"]
        self.lr = float(tcfg["lr"])
        self.weight_decay = float(tcfg.get("weight_decay", 1e-4))
        self.max_epochs = tcfg["epochs"]
        self.warmup_epochs = int(tcfg.get("warmup_epochs", 0))
        self.warmup_start_factor = float(tcfg.get("warmup_start_factor", 0.01))
        self.plot_max_events = tcfg.get("plot_max_events", 3000)

        # References filled in by the driver (train()).
        self._train_loader = None   # for prototype init in on_fit_start
        self._run_dir = None        # where PCA pngs are written

        # Validation-time accumulators.
        self._val_z: list = []
        self._val_cluster: list = []
        self._val_anchor: list = []
        self._val_seen = 0

    # ------------------------------------------------------------------
    def set_train_loader(self, loader):
        self._train_loader = loader

    def set_run_dir(self, run_dir: str):
        self._run_dir = run_dir

    # ------------------------------------------------------------------
    def forward(self, batch):
        return self.model(batch)

    # ----- loss helpers -----
    @staticmethod
    def _anchor_ce(logits, anchor_label):
        mask = anchor_label >= 0
        if not mask.any():
            return logits.new_tensor(0.0)
        return F.cross_entropy(logits[mask], anchor_label[mask])

    # ------------------------------------------------------------------
    def on_fit_start(self):
        """Init prototypes (non-random) from anchor embeddings over the train set.

        The model is already on the correct device here.
        """
        # Each rank iterates the full (non-distributed) loader here, so every rank
        # computes identical prototypes -> the replicas stay in sync under DDP.
        rank0 = self.trainer.is_global_zero
        loader = self._train_loader
        if loader is None:
            loader = self.trainer.train_dataloader
        if loader is None:
            if rank0:
                print("[init] WARNING: no train loader available; prototypes stay random.")
            return

        was_training = self.model.training
        self.model.eval()
        zs, labels = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                anchor = batch.anchor_label
                mask = anchor >= 0
                if mask.any():
                    out = self.model(batch)
                    zs.append(out["z"][mask].detach())
                    labels.append(anchor[mask])
        if zs:
            z = torch.cat(zs, 0)
            lab = torch.cat(labels, 0)
            self.model.head.init_prototypes_from_anchors(z, lab)
            if rank0:
                print(f"[init] Prototypes initialized from {z.shape[0]} anchor events.")
        elif rank0:
            print("[init] WARNING: no anchor events found; prototypes stay random.")
        if was_training:
            self.model.train()

    def _da_ramp(self) -> float:
        """Linear warmup for the domain-alignment weight: 0 before epoch w0,
        1.0 from epoch w1 on. The first epochs the CE must place the prototypes
        without the alignment pulling on a still-amorphous latent."""
        e0, e1 = self.da_warmup
        if self.current_epoch < e0:
            return 0.0
        if self.current_epoch >= e1:
            return 1.0
        return (self.current_epoch - e0) / max(1, e1 - e0)

    def _prior_ramp(self) -> float:
        """Linear warmup for the cluster-proportion prior: 0 before epoch w0,
        1.0 from epoch w1 on. Lets clusters FORM (swap/ce/vicreg) before the
        proportion prior is imposed, so a minority class (electron) is not
        crushed toward the dominant mode before it is even separable. No key ->
        always 1.0 (unchanged for configs that don't set prior_warmup_epochs)."""
        if self.prior_warmup is None:
            return 1.0
        e0, e1 = self.prior_warmup
        if self.current_epoch < e0:
            return 0.0
        if self.current_epoch >= e1:
            return 1.0
        return (self.current_epoch - e0) / max(1, e1 - e0)

    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        view_a = hit_dropout(batch, self.p_drop)
        view_b = hit_dropout(batch, self.p_drop)

        out_a = self.model(view_a)
        out_b = self.model(view_b)

        l_swap = swap_loss(out_a["logits"], out_b["logits"], sharpen_temp=self.sharpen)
        l_ce = self._anchor_ce(out_a["logits"], batch.anchor_label)
        l_vic, _, _ = vicreg_loss(
            out_a["z_raw"], self.vcfg["var_weight"], self.vcfg["cov_weight"],
            self.vcfg.get("var_gamma", 1.0),
        )
        l_prior = out_a["logits"].new_tensor(0.0)
        pbar_obs = None
        if self.prior_enabled:
            # p_bar: batch-mean soft assignment over the K clusters, (K,), computed
            # over the UNLABELED events only (anchors are excluded by default).
            logits_p = out_a["logits"]
            if self.prior_exclude_anchors:
                logits_p = logits_p[batch.anchor_label < 0]
            if logits_p.shape[0] > 0:
                pbar = F.softmax(logits_p, dim=-1).mean(0)
                if self.prior_ema:
                    # Smoothed estimate: history is detached (no grad), gradient still
                    # flows through the current batch's contribution.
                    m = self.prior_ema_momentum
                    pbar = m * self.pbar_ema + (1.0 - m) * pbar
                    self.pbar_ema = pbar.detach()
                l_prior = prior_kl_loss(pbar, self.cluster_prior)
                pbar_obs = pbar.detach()

        l_da = out_a["logits"].new_tensor(0.0)
        da_dists = None
        if self.da_enabled:
            # batch.domain: 0 = primary/TB, >0 = sim anchor files (filled with 0
            # by the dataset when the reader is single-file -> loss no-ops).
            l_da, da_dists = self.domain_align(
                out_a["z"], batch.anchor_label, batch.domain
            )

        loss = (
            self.swap_w * l_swap
            + self.lam_ce * l_ce
            + self.lam_vic * l_vic
            + self.lam_prior * self._prior_ramp() * l_prior
        )
        if self.da_enabled:
            loss = loss + self.lam_da * self._da_ramp() * l_da

        lr_now = self.optimizers().param_groups[0]["lr"]
        self.log("loss/total", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=1, sync_dist=True)
        self.log("loss/swap", l_swap, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log("loss/ce", l_ce, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log("loss/vic", l_vic, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log("lr", lr_now, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        if self.prior_enabled:
            self.log("loss/prior", l_prior, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            # Observed cluster usage (unlabeled events) -> watch it converge toward
            # cluster_prior. None only if a batch had no unlabeled events.
            if pbar_obs is not None:
                for k in range(pbar_obs.numel()):
                    self.log(f"prior/pbar_{k}", pbar_obs[k], on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        if self.da_enabled:
            self.log("loss/da", l_da, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            # EMA distance sim<->TB per class: THE success metric of the alignment
            # (should fall as the sim island merges into the TB cloud).
            for c, d in (da_dists or {}).items():
                self.log(f"da/dist_{c}", d, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        return loss

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        w = max(0, min(self.warmup_epochs, self.max_epochs - 1))
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs - w)
        if w > 0:
            # Linear warmup (lr: start_factor*lr -> lr over w epochs), then cosine.
            warmup = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=self.warmup_start_factor, total_iters=w
            )
            sched = torch.optim.lr_scheduler.SequentialLR(
                opt, schedulers=[warmup, cosine], milestones=[w]
            )
        else:
            sched = cosine
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
        }

    # ------------------------------------------------------------------
    def on_validation_epoch_start(self):
        self._val_z, self._val_cluster, self._val_anchor = [], [], []
        self._val_energy = []
        self._val_seen = 0

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # ----- val loss (same objective as training) -> early-stop monitor -----
        view_a = hit_dropout(batch, self.p_drop)
        view_b = hit_dropout(batch, self.p_drop)
        out_a = self.model(view_a)
        out_b = self.model(view_b)
        l_swap = swap_loss(out_a["logits"], out_b["logits"], sharpen_temp=self.sharpen)
        l_ce = self._anchor_ce(out_a["logits"], batch.anchor_label)
        l_vic, _, _ = vicreg_loss(
            out_a["z_raw"], self.vcfg["var_weight"], self.vcfg["cov_weight"],
            self.vcfg.get("var_gamma", 1.0),
        )
        val_loss = self.swap_w * l_swap + self.lam_ce * l_ce + self.lam_vic * l_vic
        if self.prior_enabled:
            # Instantaneous p_bar (no EMA update at val time -> training state stays
            # untouched); keeps val/loss consistent with the training objective.
            # Anchors excluded here too, matching training_step.
            logits_p = out_a["logits"]
            if self.prior_exclude_anchors:
                logits_p = logits_p[batch.anchor_label < 0]
            if logits_p.shape[0] > 0:
                pbar = F.softmax(logits_p, dim=-1).mean(0)
                val_loss = val_loss + self.lam_prior * prior_kl_loss(pbar, self.cluster_prior)
        # sync_dist=True -> averaged over batches AND ranks, so EarlyStopping sees
        # a single consistent value across the DDP group.
        self.log("val/loss", val_loss, on_epoch=True, prog_bar=True,
                 sync_dist=True, batch_size=1)

        # ----- collect embeddings for the metric/plot (bounded per rank) -------
        # Under DDP each rank only sees its shard; the cap keeps the collected set
        # bounded. Shards are gathered in on_validation_epoch_end.
        if self._val_seen >= self.plot_max_events:
            # Past the plot cap, still collect anchored events: the held-out
            # anchor accuracy must see ALL val anchors (~hundreds), not just the
            # ones that happen to land in the first plot_max_events events —
            # otherwise ckpt_monitor selects on a ~14-anchor sample.
            am = batch.anchor_label >= 0
            if am.any():
                self._val_z.append(out_a["z"][am].detach().cpu().numpy())
                self._val_cluster.append(out_a["logits"][am].argmax(1).detach().cpu().numpy())
                self._val_anchor.append(batch.anchor_label[am].detach().cpu().numpy())
                self._val_energy.append(self._batch_energy(batch)[am.cpu().numpy()])
            return
        self._val_z.append(out_a["z"].detach().cpu().numpy())
        self._val_cluster.append(out_a["logits"].argmax(1).detach().cpu().numpy())
        self._val_anchor.append(batch.anchor_label.detach().cpu().numpy())
        self._val_energy.append(self._batch_energy(batch))
        self._val_seen += out_a["z"].shape[0]

    @staticmethod
    def _batch_energy(batch) -> np.ndarray:
        """Beam energy in GeV per event, rounded to int (0 if the field is absent).

        ``energy_gev`` is the UNSCALED copy the dataset keeps precisely for this:
        ``batch.energy`` has already been through log/log_z and would group
        events by a scaled value that changes with the scaling config.
        """
        e = getattr(batch, "energy_gev", None)
        if e is None:
            return np.zeros(int(batch.anchor_label.shape[0]), dtype=np.int64)
        return np.rint(e.detach().cpu().numpy().reshape(-1)).astype(np.int64)

    @staticmethod
    def _all_gather_np(arr: np.ndarray) -> np.ndarray:
        """Concatenate a per-rank numpy array across all DDP ranks.

        Uses all_gather_object so ranks may hold different-length shards. Every
        rank receives the full concatenation (identical on all ranks), so the
        metric below is computed the same everywhere. No-op outside DDP.
        """
        if not (dist.is_available() and dist.is_initialized()):
            return arr
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, arr)
        return np.concatenate(gathered, axis=0)

    def on_validation_epoch_end(self):
        d = self.model.head.proj_dim
        z = np.concatenate(self._val_z) if self._val_z else np.zeros((0, d), np.float32)
        cluster = np.concatenate(self._val_cluster) if self._val_cluster else np.zeros((0,), np.int64)
        anchor = np.concatenate(self._val_anchor) if self._val_anchor else np.zeros((0,), np.int64)
        energy = np.concatenate(self._val_energy) if self._val_energy else np.zeros((0,), np.int64)

        # Gather shards across ranks so every rank sees the full val set and the
        # monitored metric is global (not a per-rank fraction).
        z = self._all_gather_np(z)
        cluster = self._all_gather_np(cluster)
        anchor = self._all_gather_np(anchor)
        energy = self._all_gather_np(energy)
        if z.shape[0] == 0:
            return

        # held-out anchor accuracy (val-split anchors never entered CE / proto init)
        amask = anchor >= 0
        if amask.any():
            acc = float((cluster[amask] == anchor[amask]).mean())
            if self.trainer.is_global_zero:
                print(
                    f"[epoch {self.current_epoch:03d}] "
                    f"val held-out anchor acc={acc:.3f} ({int(amask.sum())} anchors)"
                )
            # Value is identical on every rank (full gathered set); log without
            # sync so ModelCheckpoint sees the true global metric on all ranks.
            self.log("val/heldout_anchor_acc", acc, prog_bar=True, batch_size=1, sync_dist=False)

            # ----- the same accuracy, broken down by (class, energy) ----------
            # In a mixed run the aggregate hides exactly what the run is asking:
            # a model that nails 70 GeV and fails 30 GeV scores the same as one
            # that is mediocre at both, because the two energies contribute
            # different numbers of anchors. `..._worst_group` is that blind spot
            # made into a single number: the accuracy of the WORST (class,
            # energy) cell, i.e. what the model is actually guaranteed to do.
            a_lab, a_clu, a_en = anchor[amask], cluster[amask], energy[amask]
            group_accs, lines = [], []
            for cls in np.unique(a_lab):
                for en in np.unique(a_en[a_lab == cls]):
                    m = (a_lab == cls) & (a_en == en)
                    g_acc = float((a_clu[m] == a_lab[m]).mean())
                    group_accs.append(g_acc)
                    self.log(f"val/acc_c{int(cls)}_E{int(en)}", g_acc,
                             batch_size=1, sync_dist=False)
                    lines.append(f"    class {int(cls)}  {int(en):>3d} GeV : "
                                 f"acc={g_acc:.3f}  (n={int(m.sum())})")
            if group_accs:
                self.log("val/heldout_anchor_acc_worst_group", float(min(group_accs)),
                         prog_bar=True, batch_size=1, sync_dist=False)
                if self.trainer.is_global_zero:
                    print("\n".join(lines))

        # ----- latent PCA plot (rank 0 only: avoids racing on the same path) --
        if not self.trainer.is_global_zero:
            return

        # Bound the plotted set (gathering may yield up to world_size * cap points).
        if z.shape[0] > self.plot_max_events:
            idx = np.random.default_rng(0).choice(z.shape[0], self.plot_max_events, replace=False)
            z, cluster, anchor = z[idx], cluster[idx], anchor[idx]

        proto = self.model.head.prototypes.detach().cpu().numpy()
        base = self._run_dir if self._run_dir is not None else "."
        png = os.path.join(base, "pca", f"epoch_{self.current_epoch:03d}.png")
        fig = plot_latent_pca(z, cluster, anchor, proto, self.current_epoch, png)

        if wandb is not None and isinstance(self.logger, WandbLogger):
            self.logger.experiment.log({"latent/pca": wandb.Image(fig)})
        plt.close(fig)


# ----------------------------------------------------------------------
def _resolve_accelerator(device: str):
    """Map a device string to (accelerator, devices) for L.Trainer.

        "cpu"          -> ("cpu", 1)
        "cuda:0"       -> ("gpu", [0])      # one specific physical GPU
        "cuda:0,1,2"   -> ("gpu", [0,1,2])  # these physical GPUs (multi-GPU)
        "cuda" / "gpu" -> ("gpu", -1)       # all visible GPUs (multi-GPU)
        "gpu:4"        -> ("gpu", 4)        # first 4 visible GPUs (multi-GPU)
    """
    dev = (device or "cpu").lower().strip()
    if dev.startswith(("cuda", "gpu")) and torch.cuda.is_available():
        if ":" not in dev:
            return "gpu", -1  # all visible GPUs
        spec = dev.split(":", 1)[1]
        if "," in spec:
            return "gpu", [int(i) for i in spec.split(",") if i != ""]
        n = int(spec)
        # "gpu:N" = a count of N GPUs; "cuda:N" = physical GPU index N.
        return ("gpu", n) if dev.startswith("gpu") else ("gpu", [n])
    return "cpu", 1


def _is_multi_gpu(accelerator, devices) -> bool:
    if accelerator != "gpu":
        return False
    if devices == -1:
        return True
    if isinstance(devices, int):
        return devices > 1
    if isinstance(devices, (list, tuple)):
        return len(devices) > 1
    return False


def _make_run_tag(tcfg: dict) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = (tcfg.get("wandb", {}) or {}).get("name")
    return f"{stamp}_{name}" if name else stamp


def _wandb_id_from_ckpt(ckpt_path: str):
    """Best-effort recovery of the W&B run id from a checkpoint's run dir.

    Layout: <run_dir>/checkpoints/<file>.ckpt  and  <run_dir>/wandb/run-<ts>-<id>.
    Returns the <id> of the most recent wandb run dir, or None if not found.
    """
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(ckpt_path)))
    wandb_dir = os.path.join(run_dir, "wandb")
    if not os.path.isdir(wandb_dir):
        return None
    runs = sorted(glob.glob(os.path.join(wandb_dir, "run-*-*")), key=os.path.getmtime)
    if not runs:
        return None
    return os.path.basename(runs[-1]).rsplit("-", 1)[-1]


def train(cfg: dict, resume_from: str = None):
    tcfg = cfg["train"]
    L.seed_everything(tcfg.get("seed", 42), workers=True)

    out_dir = tcfg["out_dir"]
    run_tag = _make_run_tag(tcfg)
    run_dir = os.path.join(out_dir, run_tag)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"[run_dir] artifacts -> {run_dir}")
    print(f"[ckpt_dir] checkpoints -> {ckpt_dir}")

    # ----- data -----
    train_ds, val_ds, _ = make_clustering_splits(cfg["data"], cfg["features"], cfg["scaling"])
    nw = tcfg.get("num_workers", 4)
    train_loader = DataLoader(
        train_ds, batch_size=tcfg["batch_size"], shuffle=True,
        num_workers=nw, drop_last=True,
        pin_memory=True, persistent_workers=nw > 0,
        prefetch_factor=4 if nw > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tcfg["batch_size"], shuffle=False,
        num_workers=nw, pin_memory=True, persistent_workers=nw > 0,
    )

    # ----- module -----
    module = ClusteringLitModule(cfg)
    module.set_train_loader(train_loader)
    module.set_run_dir(run_dir)

    # ----- logger (W&B) -----
    wcfg = tcfg.get("wandb", {}) or {}
    mode = wcfg.get("mode", "online")
    enabled = wcfg.get("enabled", False)
    # Resume the SAME W&B run: reuse its id (explicit wandb.id, else auto-detected
    # from the checkpoint's run dir) with resume="must" so metrics continue on the
    # same run instead of starting a new one.
    wandb_id = wcfg.get("id")
    if enabled and resume_from and not wandb_id:
        wandb_id = _wandb_id_from_ckpt(resume_from)
        if wandb_id:
            print(f"[wandb] resuming run id={wandb_id} (auto-detected)")
    logger = False
    if enabled and mode != "disabled" and wandb is not None:
        logger = WandbLogger(
            project=wcfg.get("project", "gatr-clustering"),
            entity=wcfg.get("entity"),
            name=wcfg.get("name"),
            save_dir=run_dir,
            offline=(mode == "offline"),
            id=wandb_id,
            resume=("must" if wandb_id else None),
        )
        logger.experiment.log(
            {"model/params": sum(p.numel() for p in module.parameters())}
        )
    elif enabled and wandb is None:
        print("[wandb] not installed; running without a logger")

    # ----- callbacks -----
    monitor = tcfg.get("ckpt_monitor", "val/heldout_anchor_acc")
    save_top_k = tcfg.get("ckpt_save_top_k", 2)
    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(dirpath=ckpt_dir, save_last=True),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor=monitor,
            mode="max",
            save_top_k=save_top_k,
            filename="best-epoch{epoch:03d}-acc{" + monitor + ":.4f}",
            auto_insert_metric_name=False,
        ),
    ]

    # ----- early stopping on val loss (patience counted in validation checks) --
    es_patience = tcfg.get("early_stop_patience", 0)
    if es_patience and es_patience > 0:
        callbacks.append(
            ResumableEarlyStopping(
                monitor="val/loss",
                mode="min",
                patience=int(es_patience),
                min_delta=float(tcfg.get("early_stop_min_delta", 0.0)),
                check_finite=True,
                verbose=True,
            )
        )

    # ----- trainer -----
    accelerator, devices = _resolve_accelerator(tcfg.get("device", "cuda:0"))
    multi_gpu = _is_multi_gpu(accelerator, devices)
    # DDP for >1 GPU. find_unused_parameters=True guards against the attention
    # aggregation leaving some params out of the graph on a given step (which
    # would otherwise crash DDP's gradient sync). Lightning auto-wraps the
    # dataloaders in a DistributedSampler, so each GPU sees a disjoint 1/N shard.
    strategy = DDPStrategy(find_unused_parameters=True) if multi_gpu else "auto"
    print(
        f"[trainer] accelerator={accelerator}, devices={devices}, "
        f"strategy={'ddp(find_unused_parameters=True)' if multi_gpu else 'auto'}"
    )
    trainer = L.Trainer(
        max_epochs=tcfg["epochs"],
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        # bf16 autocast on tensor cores (same dynamic range as fp32, no loss
        # scaling needed). Set train.precision: 32-true to go back to full fp32.
        precision=tcfg.get("precision", "bf16-mixed" if accelerator == "gpu" else "32-true"),
        use_distributed_sampler=True,
        gradient_clip_val=1.0,
        check_val_every_n_epoch=tcfg.get("plot_every", 5),
        logger=logger,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
    )

    if resume_from:
        print(f"[resume] restoring full training state from {resume_from}")
    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_from,
    )

    if wandb is not None and wandb.run is not None:
        wandb.finish()
    print(f"[done] checkpoints in {ckpt_dir} (last.ckpt + best-epoch*.ckpt)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--data_path", default=None, help="override data.path")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--wandb", dest="wandb", action="store_true", default=None,
                    help="enable W&B logging (overrides config)")
    ap.add_argument("--no_wandb", dest="wandb", action="store_false",
                    help="disable W&B logging (overrides config)")
    ap.add_argument("--wandb_project", default=None)
    ap.add_argument("--wandb_mode", default=None, help="online | offline | disabled")
    ap.add_argument("--wandb_id", default=None,
                    help="resume logging into this existing W&B run id (resume='must'); "
                         "with --resume it is auto-detected from the checkpoint's run dir")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--resume", default=None,
                    help="path to a .ckpt to resume from (restores epoch, optimizer, "
                         "LR scheduler, global_step, RNG and callback state)")
    args = ap.parse_args()

    cfg = load_config(args.cfg)
    if args.data_path:
        cfg["data"]["path"] = args.data_path
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.out_dir:
        cfg["train"]["out_dir"] = args.out_dir
    if args.device:
        cfg["train"]["device"] = args.device

    cfg["train"].setdefault("wandb", {})
    if args.wandb is not None:
        cfg["train"]["wandb"]["enabled"] = args.wandb
    if args.wandb_project:
        cfg["train"]["wandb"]["project"] = args.wandb_project
    if args.wandb_mode:
        cfg["train"]["wandb"]["mode"] = args.wandb_mode
    if args.wandb_id:
        cfg["train"]["wandb"]["id"] = args.wandb_id
    if args.run_name:
        cfg["train"]["wandb"]["name"] = args.run_name
    train(cfg, resume_from=args.resume)


if __name__ == "__main__":
    main()
