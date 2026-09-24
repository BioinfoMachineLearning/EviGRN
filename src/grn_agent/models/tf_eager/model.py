"""TF-centered windowed EAGER model."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .window_batch import NUM_TOKEN_KINDS, TARGET_POSITION_VOCAB, TOKEN_LAYOUT_EVIDENCE, VALUE_DIM, TfEagerWindowBatch


@dataclass
class TfEagerConfig:
    d_model: int = 128
    n_heads: int = 4
    n_encoder_layers: int = 2
    dropout: float = 0.1
    tf_vocab: int = 8192
    gene_vocab: int = 8192
    context_vocab: int = 1024
    target_pos_vocab: int = TARGET_POSITION_VOCAB
    token_layout: str = TOKEN_LAYOUT_EVIDENCE
    use_tf_identity: bool = True
    use_gene_identity: bool = True
    use_context_identity: bool = True
    drop_token_kinds: list[str] = field(default_factory=list)
    decoder_mode: str = "staged"


class TfEagerWindowModel(nn.Module):
    """Scores one TF-centered candidate window with one logit per target gene."""

    def __init__(self, cfg: TfEagerConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or TfEagerConfig()
        c = self.cfg
        decoder_mode = str(c.decoder_mode or "staged").strip().lower()
        if decoder_mode not in {"staged", "single_stage"}:
            raise ValueError(f"unknown tf-eager decoder_mode: {c.decoder_mode!r}")
        d = c.d_model
        self.token_kind_emb = nn.Embedding(NUM_TOKEN_KINDS, d)
        self.value_mlp = nn.Sequential(nn.Linear(VALUE_DIM, d), nn.GELU(), nn.Linear(d, d))
        self.conf_proj = nn.Linear(1, d)
        self.target_pos_emb = nn.Embedding(c.target_pos_vocab, d)
        self.ctx_token_emb = nn.Embedding(max(1, c.context_vocab), d) if c.use_context_identity else None
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=c.n_heads,
            dim_feedforward=4 * d,
            dropout=c.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=c.n_encoder_layers)
        self.tf_emb = nn.Embedding(max(1, c.tf_vocab), d) if c.use_tf_identity else None
        self.gene_emb = nn.Embedding(max(1, c.gene_vocab), d) if c.use_gene_identity else None
        self.ctx_query_emb = nn.Embedding(max(1, c.context_vocab), d) if c.use_context_identity else None
        self.stage1 = nn.MultiheadAttention(d, c.n_heads, dropout=c.dropout, batch_first=True)
        self.stage2 = nn.MultiheadAttention(d, c.n_heads, dropout=c.dropout, batch_first=True)
        self.stage3 = nn.MultiheadAttention(d, c.n_heads, dropout=c.dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.ln3 = nn.LayerNorm(d)
        self.ff1 = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.ff2 = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.ff3 = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.dropout = nn.Dropout(c.dropout)
        self.head = nn.Linear(d, 1)

    def _initial_memory(self, batch: TfEagerWindowBatch) -> torch.Tensor:
        b, n = batch.token_kind.shape
        d = self.cfg.d_model
        h = (
            self.token_kind_emb(batch.token_kind)
            + self.value_mlp(batch.x_value)
            + self.conf_proj(batch.conf.unsqueeze(-1))
            + self.target_pos_emb(batch.token_target_pos.clamp(0, self.cfg.target_pos_vocab - 1))
        )
        if self.ctx_token_emb is not None:
            h = h + self.ctx_token_emb(batch.context_idx.clamp(0, max(1, self.cfg.context_vocab) - 1)).unsqueeze(1).expand(b, n, d)
        return h * batch.token_mask.unsqueeze(-1)

    def _attend(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        allow_mask: torch.Tensor,
        valid_mask: torch.Tensor,
        layer: nn.MultiheadAttention,
        *,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        allow = (allow_mask > 0.5) & (valid_mask > 0.5)
        fallback = valid_mask > 0.5
        no_allow = allow.sum(dim=1, keepdim=True) <= 0
        allow = torch.where(no_allow, fallback, allow)
        key_padding_mask = ~allow
        out, weights = layer(
            z,
            h,
            h,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=True,
        )
        return out, (weights if need_weights else None)

    def forward(
        self,
        batch: TfEagerWindowBatch,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        """Score candidate genes in a TF-centered window.

        When ``return_attention`` is True, also return average cross-attention
        maps for each decoder stage (analysis / interpretability only; no effect
        on training or default inference logits).
        """
        h0 = self._initial_memory(batch)
        h = self.encoder(h0, src_key_padding_mask=(batch.token_mask <= 0.5))
        h = h * batch.token_mask.unsqueeze(-1)

        b, w = batch.gene_idx.shape
        d = self.cfg.d_model
        z = self.target_pos_emb(batch.gene_pos.clamp(0, self.cfg.target_pos_vocab - 1))
        if self.tf_emb is not None:
            z = z + self.tf_emb(batch.tf_idx.clamp(0, max(1, self.cfg.tf_vocab) - 1)).unsqueeze(1).expand(b, w, d)
        if self.gene_emb is not None:
            z = z + self.gene_emb(batch.gene_idx.clamp(0, max(1, self.cfg.gene_vocab) - 1))
        if self.ctx_query_emb is not None:
            z = z + self.ctx_query_emb(batch.context_idx.clamp(0, max(1, self.cfg.context_vocab) - 1)).unsqueeze(1).expand(b, w, d)
        z = z * batch.gene_mask.unsqueeze(-1)

        attn: dict[str, torch.Tensor | None] = {
            "stage1": None,
            "stage2": None,
            "stage3": None,
        }

        if str(self.cfg.decoder_mode or "staged").strip().lower() == "single_stage":
            s, w3 = self._attend(z, h, batch.token_mask, batch.token_mask, self.stage3, need_weights=return_attention)
            attn["stage3"] = w3
            z3 = self.ln3(z + self.dropout(s))
            z = z3 + self.dropout(self.ff3(z3))
            logits = self.head(z).squeeze(-1)
            return (logits, attn) if return_attention else logits

        mech_available = ((batch.modality[:, 1] + batch.modality[:, 2] + batch.modality[:, 3]) > 0.5).float().view(b, 1, 1)
        if bool(mech_available.any().item()):
            s1, w1 = self._attend(
                z, h, batch.mech_mask, batch.token_mask, self.stage1, need_weights=return_attention
            )
            attn["stage1"] = w1
            s1 = s1 * mech_available
            z1 = self.ln1(z + self.dropout(s1))
            z = z1 + self.dropout(self.ff1(z1))

        s2, w2 = self._attend(z, h, batch.func_mask, batch.token_mask, self.stage2, need_weights=return_attention)
        attn["stage2"] = w2
        z2 = self.ln2(z + self.dropout(s2))
        z = z2 + self.dropout(self.ff2(z2))

        s3, w3 = self._attend(z, h, batch.token_mask, batch.token_mask, self.stage3, need_weights=return_attention)
        attn["stage3"] = w3
        z3 = self.ln3(z + self.dropout(s3))
        z = z3 + self.dropout(self.ff3(z3))
        logits = self.head(z).squeeze(-1)
        return (logits, attn) if return_attention else logits
