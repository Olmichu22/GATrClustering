"""
Swapped-assignment loss (SwAV-style, WITHOUT Sinkhorn).

Two augmented views of the same events produce cluster logits. The (sharpened,
detached) soft assignment of one view supervises the prediction of the other,
and vice versa. No Sinkhorn / no proportion prior in this first iteration:
the anchor CE term is what prevents degenerate assignments.
"""

import torch
import torch.nn.functional as F


def swap_loss(logits_a: torch.Tensor, logits_b: torch.Tensor, sharpen_temp: float = 0.25) -> torch.Tensor:
    """logits_*: (B, K). Returns a scalar swapped cross-entropy."""
    with torch.no_grad():
        q_a = F.softmax(logits_a / sharpen_temp, dim=-1)
        q_b = F.softmax(logits_b / sharpen_temp, dim=-1)
    log_p_a = F.log_softmax(logits_a, dim=-1)
    log_p_b = F.log_softmax(logits_b, dim=-1)
    # assignment of one view predicts the other view
    loss = -(q_b * log_p_a).sum(dim=-1).mean() - (q_a * log_p_b).sum(dim=-1).mean()
    return 0.5 * loss
