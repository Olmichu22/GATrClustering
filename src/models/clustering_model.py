"""
End-to-end clustering model:

    hits --GATrEncoder--> per-hit tokens --aggregation--> event embedding
         --PrototypeHead--> z (L2-normalized) + cluster logits

Built entirely from config: the encoder/aggregation/head sections plus the
feature-routing config (which decides in_s_channels / in_mv_channels).
"""

import torch
import torch.nn as nn

from ..data.feature_routing import (
    build_inputs,
    compute_in_mv_channels,
    compute_in_s_channels,
)
from .aggregation import build_aggregation
from .gatr_module import GATrEncoder
from .prototype_head import PrototypeHead


class ClusteringModel(nn.Module):
    def __init__(self, model_cfg: dict, features_cfg: dict):
        super().__init__()
        self.features_cfg = features_cfg
        enc = model_cfg["encoder"]
        mode = enc.get("mv_embedding_mode", "centroid")

        self.encoder = GATrEncoder(
            hidden_mv_channels=enc["hidden_mv_channels"],
            hidden_s_channels=enc["hidden_s_channels"],
            num_blocks=enc["num_blocks"],
            in_s_channels=compute_in_s_channels(features_cfg),
            in_mv_channels=compute_in_mv_channels(mode),
            out_mv_channels=enc["out_mv_channels"],
            out_s_channels=enc["out_s_channels"],
            dropout=enc.get("dropout", 0.1),
            post_dropout=enc.get("post_dropout", 0.0),
            mv_embedding_mode=mode,
        )
        self.aggregation = build_aggregation(model_cfg["aggregation"], self.encoder.token_dim)

        head = model_cfg["head"]
        self.head = PrototypeHead(
            in_dim=self.aggregation.out_dim,
            proj_dim=head["proj_dim"],
            num_clusters=head["num_clusters"],
            temperature=head.get("temperature", 0.1),
        )

    def encode_event(self, batch) -> torch.Tensor:
        """hits -> per-event embedding (before the prototype head)."""
        inp = build_inputs(batch, self.features_cfg)
        mv_out, scalar_out, _, _ = self.encoder(
            inp["mv_v_part"], inp["mv_s_part"], inp["scalars"], inp["batch_idx"]
        )
        tokens = self.encoder.node_tokens(mv_out, scalar_out)  # (N, token_dim)
        return self.aggregation(tokens, inp["batch_idx"])  # (B, agg_out_dim)

    def forward(self, batch):
        event_emb = self.encode_event(batch)
        z, logits = self.head(event_emb)
        return {"event_embedding": event_emb, "z": z, "logits": logits}
