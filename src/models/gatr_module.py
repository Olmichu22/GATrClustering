"""
GATr encoder wrapper, forked from
``GATrAutoencoder/src/models/gatr_module.py::GATrBasicModule``.

Generalizations for the clustering POC:
    - ``build_geom_embedding`` accepts a geometric-scalar tensor of shape (N, S)
      (0..S features) instead of a single hardcoded ``k``: each column is
      embedded with ``embed_scalar`` and the results are summed. S=0 (a zeros
      (N,1) placeholder) simply contributes a zero scalar.
    - AR-decoder-specific machinery is dropped; this module only encodes hits.

Per-event isolation is enforced by a block-diagonal attention mask (xformers),
so hits only attend within their own event.
"""

import torch
import torch.nn as nn
from gatr.interface import embed_point, embed_scalar, extract_point, extract_scalar
from gatr import GATr, SelfAttentionConfig, MLPConfig
from xformers.ops.fmha import BlockDiagonalMask
from torch_scatter import scatter_mean


class GATrEncoder(nn.Module):
    def __init__(
        self,
        hidden_mv_channels: int = 16,
        hidden_s_channels: int = 64,
        num_blocks: int = 4,
        in_s_channels: int = 1,
        in_mv_channels: int = 1,
        out_mv_channels: int = 1,
        out_s_channels: int = 64,
        dropout: float = 0.1,
        post_dropout: float = 0.0,
        mv_embedding_mode: str = "centroid",
        attention: SelfAttentionConfig = SelfAttentionConfig(),
        mlp: MLPConfig = MLPConfig(),
    ):
        super().__init__()
        expected_mv = 2 if mv_embedding_mode == "centroid" else 1
        assert in_mv_channels == expected_mv, (
            f"in_mv_channels={in_mv_channels} does not match "
            f"mv_embedding_mode='{mv_embedding_mode}' (expected {expected_mv})"
        )
        self.mv_embedding_mode = mv_embedding_mode
        self.out_mv_channels = out_mv_channels
        self.out_s_channels = out_s_channels

        self.gatr = GATr(
            in_mv_channels=in_mv_channels,
            out_mv_channels=out_mv_channels,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=max(in_s_channels, 1),
            out_s_channels=out_s_channels,
            hidden_s_channels=hidden_s_channels,
            num_blocks=num_blocks,
            attention=attention,
            mlp=mlp,
            dropout_prob=dropout,
        )
        self._in_s_channels = in_s_channels
        self.scalar_dropout = nn.Dropout(post_dropout)

    # ---- geometric embedding (generalized geometric scalar) ----
    def build_geom_embedding(self, mv_v_part, mv_s_part, batch):
        # mv_v_part: (N,3) point ; mv_s_part: (N,S) geometric scalar features
        mv_vec = embed_point(mv_v_part)  # (N,16)
        # Sum embed_scalar over each geometric-scalar column
        s = mv_s_part
        mv_scalar = embed_scalar(s[:, :1])
        for c in range(1, s.shape[1]):
            mv_scalar = mv_scalar + embed_scalar(s[:, c : c + 1])

        if self.mv_embedding_mode == "single":
            return (mv_vec + mv_scalar).unsqueeze(1)  # (N,1,16)
        # centroid: channel 0 = absolute pos + scalar, channel 1 = pos relative to centroid
        ch0 = (mv_vec + mv_scalar).unsqueeze(1)
        centroid = scatter_mean(mv_v_part, batch, dim=0)
        pos_rel = mv_v_part - centroid[batch]
        ch1 = embed_point(pos_rel).unsqueeze(1)
        return torch.cat([ch0, ch1], dim=1)  # (N,2,16)

    def build_attention_mask(self, batch):
        return BlockDiagonalMask.from_seqlens(torch.bincount(batch.long()).tolist())

    def forward(self, mv_v_part, mv_s_part, scalars, batch):
        embedded_geom = self.build_geom_embedding(mv_v_part, mv_s_part, batch)
        # GATr requires >=1 scalar channel; feed zeros if none configured
        if scalars.shape[1] == 0:
            scalars = torch.zeros((scalars.shape[0], 1), device=scalars.device)
        mask = self.build_attention_mask(batch)

        mv_out, scalar_out = self.gatr(embedded_geom, scalars=scalars, attention_mask=mask)
        scalar_out = self.scalar_dropout(scalar_out)

        mv0 = mv_out[:, 0, :]
        point = extract_point(mv0)
        scalar = extract_scalar(mv0.unsqueeze(1)).view(-1, 1)
        return mv_out, scalar_out, point, scalar

    def node_tokens(self, mv_out, scalar_out):
        """Per-hit token features fed to the aggregation stage:
        [flattened multivector | scalar output], dim = out_mv*16 + out_s."""
        return torch.cat([mv_out.reshape(mv_out.size(0), -1), scalar_out], dim=-1)

    @property
    def token_dim(self) -> int:
        return self.out_mv_channels * 16 + self.out_s_channels
