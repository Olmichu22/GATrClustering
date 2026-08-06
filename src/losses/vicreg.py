"""
VICReg variance + covariance regularization on the event projection.

Applied to the projector output BEFORE L2 normalization (``z_raw``). VICReg was
designed for unnormalized embeddings: on the L2-normalized ``z`` the per-dim std
is capped at ~1/sqrt(d), so ``var_gamma=1`` would be unreachable and the hinge
would fight the normalization. On ``z_raw`` a target std of ``gamma=1`` is well
posed.

    variance  : hinge that keeps the per-dimension std above ``gamma`` (prevents
                collapse of z to a point).
    covariance: pushes off-diagonal covariance entries toward zero (decorrelates
                the dimensions of z).

The invariance term of full VICReg is omitted on purpose: there are no explicit
positive pairs here; that role is covered by the swap loss across dropout views.
"""

import torch
import torch.nn.functional as F


def vicreg_loss(
    z: torch.Tensor,
    var_weight: float = 1.0,
    cov_weight: float = 0.04,
    gamma: float = 1.0,
    eps: float = 1e-4,
):
    """z: (B, d). Returns (total, var_term, cov_term)."""
    B, d = z.shape
    if B < 2:
        z0 = z.new_zeros(())
        return z0, z0, z0
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    var_term = torch.mean(F.relu(gamma - std))

    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.t() @ zc) / max(B - 1, 1)
    off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    cov_term = off_diag / d

    total = var_weight * var_term + cov_weight * cov_term
    return total, var_term, cov_term
