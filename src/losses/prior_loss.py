"""
Cluster-proportion prior (anti-collapse).

The swap loss (SwAV without Sinkhorn) has a trivial minimum where every event is
routed to a single cluster; VICReg only regularizes the embedding z and is blind
to the assignment marginal, so nothing stops a cluster from dying. This term ties
the batch-averaged soft assignment ``p_bar`` to a target proportion ``prior`` via

    KL(prior || p_bar) = sum_k prior_k * (log prior_k - log p_bar_k)

which -> +inf as any used cluster's mass p_bar_k -> 0, i.e. it explicitly forbids
starving a cluster. With ``prior = 1/K`` this reduces to the usual equipartition
prior; with a physics prior (e.g. mu:0.73, pi:0.23, e:0.04) it biases toward that
baseline instead. It is a *soft* prior: lambda controls how hard it pulls.
"""

import torch


def prior_kl_loss(pbar: torch.Tensor, prior: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """KL(prior || pbar).

    pbar, prior: (K,) vectors on the simplex. ``pbar`` carries gradient; ``prior``
    is a constant target. Returns a non-negative scalar (0 iff pbar == prior).
    """
    pbar = pbar.clamp_min(eps)
    prior = prior.clamp_min(eps)
    return (prior * (prior.log() - pbar.log())).sum()
