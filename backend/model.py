"""
model.py — RangeFlow-aware GPT architecture.

The key innovation here is RangeAwareAttention, which operates in one of
three modes:

  • "standard" — plain causal self-attention, no constraints applied.
  • "capture"  — a forward pass over the *prompt* tokens.  The min/max
                  bounding box of the K and V projections across the
                  sequence dimension is recorded as an "anchor".
  • "guard"    — each new token's K/V projections are intersected with
                  the captured anchor box (expanded by ±epsilon).  This
                  steers the attention geometry back toward the prompt's
                  semantic neighbourhood, keeping generation on-topic.

range_epsilon (set per-request) controls how tightly the generation is
constrained:
  ≈ 0.05  →  strict, stays very close to the prompt's topic.
  ≈ 0.20  →  looser, more creative but still grounded.
"""

import logging

import torch
import torch.nn as nn

from config import ModelConfig

logger = logging.getLogger("noir_whisper.model")


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class RangeAwareAttention(nn.Module):
    """
    Multi-head self-attention with optional RangeFlow constraints.

    Buffers anchor_k_min / anchor_k_max / anchor_v_min / anchor_v_max are
    registered so they move with the module when .to(device) is called, but
    they start as None and are populated during a 'capture' pass.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_k      = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj  = nn.Linear(d_model, d_model)
        self.dropout   = nn.Dropout(dropout)

        # RangeFlow runtime state (not saved in checkpoint)
        self.mode:    str   = "standard"
        self.epsilon: float = 0.1          # overridden per-request

        # Anchor buffers — None until a 'capture' pass runs
        self.register_buffer("anchor_k_min", None)
        self.register_buffer("anchor_k_max", None)
        self.register_buffer("anchor_v_min", None)
        self.register_buffer("anchor_v_max", None)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Project all heads at once, then split Q / K / V
        qkv = (
            self.qkv_proj(x)
            .reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Shape: [batch, n_heads, seq_len, d_k]

        # ---- RANGEFLOW LOGIC ------------------------------------------
        if self.mode == "capture":
            # Record the bounding box of K and V across the sequence dim.
            # dim=2 is the seq_len axis after the permute above.
            self.anchor_k_min = k.min(dim=2, keepdim=True)[0].detach()
            self.anchor_k_max = k.max(dim=2, keepdim=True)[0].detach()
            self.anchor_v_min = v.min(dim=2, keepdim=True)[0].detach()
            self.anchor_v_max = v.max(dim=2, keepdim=True)[0].detach()

        elif self.mode == "guard" and self.anchor_k_min is not None:
            # Expand each new token's K/V interval by ±epsilon, then
            # intersect with the captured anchor box.
            k_int_lo, k_int_hi = k - self.epsilon, k + self.epsilon
            v_int_lo, v_int_hi = v - self.epsilon, v + self.epsilon

            valid_k_lo = torch.max(k_int_lo, self.anchor_k_min)
            valid_k_hi = torch.min(k_int_hi, self.anchor_k_max)
            valid_v_lo = torch.max(v_int_lo, self.anchor_v_min)
            valid_v_hi = torch.min(v_int_hi, self.anchor_v_max)

            # Guard against degenerate (empty) intervals
            valid_k_lo = torch.min(valid_k_lo, valid_k_hi)
            valid_v_lo = torch.min(valid_v_lo, valid_v_hi)

            # Clamp K and V into the valid range
            k = torch.max(valid_k_lo, torch.min(k, valid_k_hi))
            v = torch.max(valid_v_lo, torch.min(v, valid_v_hi))
        # ---------------------------------------------------------------

        # Standard scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = self.dropout(torch.softmax(scores, dim=-1))
        out = (
            torch.matmul(attn_weights, v)
            .transpose(1, 2)
            .contiguous()
            .reshape(batch_size, seq_len, self.d_model)
        )
        return self.out_proj(out)

    # ------------------------------------------------------------------
    def clear_anchor(self) -> None:
        """Reset anchor buffers so a fresh capture can be run."""
        self.anchor_k_min = None
        self.anchor_k_max = None
        self.anchor_v_min = None
        self.anchor_v_max = None


# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        # NOTE: these attribute names MUST stay as linear1 / linear2 to match
        # the checkpoint's state_dict keys (blocks.N.ff.linear1.weight, etc.)
        self.linear1   = nn.Linear(d_model, d_ff)
        self.linear2   = nn.Linear(d_ff, d_model)
        self.dropout   = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.attn    = RangeAwareAttention(d_model, n_heads, dropout)
        self.ff      = FeedForward(d_model, d_ff, dropout)
        self.ln1     = nn.LayerNorm(d_model)
        self.ln2     = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x


# ---------------------------------------------------------------------------

class GPTModel(nn.Module):
    """
    GPT-style language model with RangeFlow-aware attention layers.
    """

    def __init__(self, config: type[ModelConfig]):
        super().__init__()
        self.token_embed = nn.Embedding(config.VOCAB_SIZE, config.D_MODEL)
        self.pos_embed   = nn.Embedding(config.MAX_SEQ_LEN, config.D_MODEL)
        self.blocks      = nn.ModuleList([
            TransformerBlock(config.D_MODEL, config.N_HEADS, config.D_FF, config.DROPOUT)
            for _ in range(config.N_LAYERS)
        ])
        self.ln_final = nn.LayerNorm(config.D_MODEL)
        self.head     = nn.Linear(config.D_MODEL, config.VOCAB_SIZE, bias=False)

        # Weight tying: token embedding and output projection share weights
        self.token_embed.weight = self.head.weight

        logger.debug(
            "GPTModel initialised — vocab=%d, d_model=%d, layers=%d, heads=%d",
            config.VOCAB_SIZE, config.D_MODEL, config.N_LAYERS, config.N_HEADS,
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        h = self.token_embed(x) + self.pos_embed(positions)
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device)
        ).unsqueeze(0).unsqueeze(0)
        for block in self.blocks:
            h = block(h, mask)
        return self.head(self.ln_final(h))

    # ------------------------------------------------------------------
    def set_range_mode(self, mode: str) -> None:
        """Propagate mode to every attention layer."""
        for block in self.blocks:
            block.attn.mode = mode

    def set_epsilon(self, epsilon: float) -> None:
        """Set the RangeFlow epsilon on every attention layer."""
        for block in self.blocks:
            block.attn.epsilon = epsilon

    def clear_anchors(self) -> None:
        """Clear all captured anchors — call before each new prompt."""
        for block in self.blocks:
            block.attn.clear_anchor()

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())