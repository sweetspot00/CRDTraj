"""
Transformer backbone for CRDTraj.

TransformerBlock — Eqs. (11-13):
  (11) Self-attention  on agent tokens (agent-agent coordination)
  (12) Cross-attention from agent tokens to [map ∥ ctx] tokens
  (13) FFN + LayerNorm

TransformerBackbone — L stacked TransformerBlocks
"""

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """
    Single CRDTraj transformer block.

    Inputs:
      H_agent: (B, N, d)
      H_ctx:   (B, P+L, d)  concatenation of map and context tokens

    Operations (per Eqs. 11-13):
      H'  = LN(SelfAttn(H_agent)  + H_agent)
      H'' = LN(CrossAttn(H', H_ctx) + H')
      H   = LN(FFN(H'')            + H'')

    Args:
        d:         model dimension
        n_heads:   number of attention heads
        ffn_mult:  FFN hidden dim multiplier (default 4)
        dropout:   dropout rate
    """

    def __init__(self, d: int, n_heads: int = 8, ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        # --- self-attention (agent ↔ agent) ---
        self.self_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d)

        # --- cross-attention (agent → map+ctx) ---
        self.cross_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d)

        # --- feed-forward ---
        hidden = d * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d)

    def forward(
        self,
        H_agent: torch.Tensor,
        H_ctx: torch.Tensor,
        agent_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            H_agent:    (B, N, d)   agent tokens
            H_ctx:      (B, P+L, d) map + context tokens (keys/values for cross-attn)
            agent_mask: (B, N) bool mask — True for padding agents (ignored in self-attn)
        Returns:
            (B, N, d) updated agent tokens
        """
        # Eq. (11) — self-attention among agents
        sa_out, _ = self.self_attn(
            H_agent, H_agent, H_agent,
            key_padding_mask=agent_mask,
        )
        H = self.norm1(H_agent + sa_out)

        # Eq. (12) — cross-attention: agents query map+context
        ca_out, _ = self.cross_attn(H, H_ctx, H_ctx)
        H = self.norm2(H + ca_out)

        # Eq. (13) — FFN
        H = self.norm3(H + self.ffn(H))
        return H


class TransformerBackbone(nn.Module):
    """
    Stack of L TransformerBlocks.

    Returns the final agent token representations {z_i}_{i=1}^N.

    Args:
        d:       model dimension
        L:       number of layers
        n_heads: attention heads per block
        ffn_mult, dropout: passed to each block
    """

    def __init__(self, d: int, L: int = 6, n_heads: int = 8, ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(d, n_heads, ffn_mult, dropout) for _ in range(L)]
        )

    def forward(
        self,
        H_agent: torch.Tensor,
        H_ctx: torch.Tensor,
        agent_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            H_agent:    (B, N, d)   agent tokens (with timestep emb added)
            H_ctx:      (B, P+L, d) map + context tokens
            agent_mask: (B, N) bool padding mask
        Returns:
            z: (B, N, d)  contextualized agent representations
        """
        for block in self.blocks:
            H_agent = block(H_agent, H_ctx, agent_mask)
        return H_agent
