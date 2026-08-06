"""
Class-conditional domain alignment (sim anchors -> TB anchors), opt-in.

The sim<->TB domain gap makes the encoder embed simulation showers in their own
"island", so the anchor CE learned on sim does not transfer to the test-beam
cloud. Marginal CORAL/MMD is the wrong tool here (sim is class-balanced while TB
is ~73% muon, so aligning marginals superimposes the wrong classes); instead we
align *per class*, using the classes that have anchors in BOTH domains:

    L_da = sum_{c seen in both} || mean_batch(z_sim_c) - sg(mu_tb_c) ||^2

Design choices (deliberate, see the POC discussion):

  * Stop-gradient on the TB side: only the sim embeddings are pulled toward
    their TB homologues. The TB cloud -- the thing we actually cluster -- is
    never distorted by this term.
  * The TB target ``mu_tb`` is an EMA over batches (updated under no_grad):
    with ~1% anchor events a per-batch TB mean is pure noise.
  * The *loss* uses the CURRENT batch's sim class-mean (keeps gradient through
    the sim-anchor embeddings). The ``mu_sim`` EMA buffer is NOT part of the
    loss -- it only feeds the logged per-class distances, which would otherwise
    jump around with the batch composition.
  * Buffers are non-persistent: enabling the flag never breaks old checkpoints,
    and new checkpoints stay loadable by evaluate_clustering (no extra keys).
  * Under DDP with broadcast_buffers=True the EMAs follow rank 0's batches;
    that is fine for a smoothed logging/target signal.

Classes with anchors in only one domain (e.g. electron: sim-only) contribute
nothing until the other domain has been seen at least once.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn


class DomainAlignLoss(nn.Module):
    def __init__(self, num_classes: int, dim: int, ema_momentum: float = 0.9):
        super().__init__()
        self.num_classes = num_classes
        self.m = float(ema_momentum)
        # EMA class means per domain + "seen at least once" flags (non-persistent:
        # they reset on resume and warm up again within a few batches).
        self.register_buffer("mu_sim", torch.zeros(num_classes, dim), persistent=False)
        self.register_buffer("mu_tb", torch.zeros(num_classes, dim), persistent=False)
        self.register_buffer("seen_sim", torch.zeros(num_classes, dtype=torch.bool), persistent=False)
        self.register_buffer("seen_tb", torch.zeros(num_classes, dtype=torch.bool), persistent=False)

    @torch.no_grad()
    def _update_ema(self, mu: torch.Tensor, seen: torch.Tensor, c: int, batch_mean: torch.Tensor):
        if seen[c]:
            mu[c] = self.m * mu[c] + (1.0 - self.m) * batch_mean
        else:  # first observation initializes the EMA directly
            mu[c] = batch_mean
            seen[c] = True

    def forward(
        self, z: torch.Tensor, anchor_label: torch.Tensor, domain: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """z: (B, d) embeddings; anchor_label: (B,) with -1 = unlabeled;
        domain: (B,) with 0 = primary/TB, >0 = sim anchor files.

        Returns (loss, {class: EMA distance}) -- the dict only holds classes
        currently seen in both domains and is detached (logging only).
        """
        loss = z.new_tensor(0.0)
        dists: Dict[int, torch.Tensor] = {}

        is_anchor = anchor_label >= 0
        if not is_anchor.any():
            return loss, dists
        sim_mask = is_anchor & (domain > 0)
        tb_mask = is_anchor & (domain == 0)

        for c in range(self.num_classes):
            tb_c = tb_mask & (anchor_label == c)
            if tb_c.any():
                self._update_ema(self.mu_tb, self.seen_tb, c, z[tb_c].detach().mean(0))
            sim_c = sim_mask & (anchor_label == c)
            if sim_c.any():
                batch_mu_sim = z[sim_c].mean(0)  # keeps grad for the loss
                self._update_ema(self.mu_sim, self.seen_sim, c, batch_mu_sim.detach())
                if self.seen_tb[c]:
                    # sg() on the TB target: pull sim toward TB, never the reverse.
                    loss = loss + (batch_mu_sim - self.mu_tb[c].detach()).pow(2).sum()
            if self.seen_sim[c] and self.seen_tb[c]:
                dists[c] = (self.mu_sim[c] - self.mu_tb[c]).norm().detach()

        return loss, dists
