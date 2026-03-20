"""
Encoders for CRDTraj:
  AgentTokenizer  — Eq. (6)  MLP([T×2 + 6] → d) per agent
  MapEncoder      — Eq. (7)  ResNet-18 CNN → P patch tokens → linear to d
  ContextEncoder  — Eq. (8)  frozen SentenceBERT → trainable MLP to d (L tokens)
  TimestepEmbedding — Eq. (9) sinusoidal → MLP → d
"""

import math
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mlp(dims: list[int], act: type = nn.GELU) -> nn.Sequential:
    """Build a simple MLP with the given layer dimensions."""
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Standard sinusoidal positional encoding for scalar timesteps.

    Args:
        timesteps: (B,) integer tensor
        dim: embedding dimension
    Returns:
        (B, dim) float tensor
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# ---------------------------------------------------------------------------
# AgentTokenizer — Eq. (6)
# ---------------------------------------------------------------------------

class AgentTokenizer(nn.Module):
    """
    Maps each agent's noisy trajectory + initial state to a d-dimensional token.

    Input per agent:
      τ_t^i  ∈ R^{T×2}  — flattened to R^{T*2}
      s_0^i  ∈ R^6       — (x0, y0, vx0, vy0, gx, gy)
    Concatenation: R^{T*2 + 6} → MLP → R^d
    """

    def __init__(self, T: int, d: int, hidden_dim: int = 256):
        super().__init__()
        in_dim = T * 2 + 6
        self.mlp = _make_mlp([in_dim, hidden_dim, d])

    def forward(self, tau_t: torch.Tensor, s0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tau_t: (B, N, T, 2) noisy trajectories
            s0:    (B, N, 6)    initial agent states
        Returns:
            (B, N, d) agent tokens
        """
        B, N, T, _ = tau_t.shape
        tau_flat = tau_t.reshape(B, N, T * 2)   # (B, N, T*2)
        x = torch.cat([tau_flat, s0], dim=-1)   # (B, N, T*2+6)
        return self.mlp(x)                       # (B, N, d)


# ---------------------------------------------------------------------------
# MapEncoder — Eq. (7)
# ---------------------------------------------------------------------------

class MapEncoder(nn.Module):
    """
    Encodes a top-down map image into P spatial patch tokens of dimension d.

    Architecture:
      ResNet-18 (optionally frozen) → last feature map (B, 512, H', W')
      Flatten patches → (B, P, 512)
      Linear projection → (B, P, d)

    Args:
        d:       output token dimension
        frozen:  freeze ResNet-18 weights (default False for fine-tuning)
        P_h, P_w: spatial grid size after feature extractor (default 7×7 = 49 patches)
    """

    def __init__(self, d: int, frozen: bool = False):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Remove avgpool + fc; keep up to layer4
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        if frozen:
            for p in self.features.parameters():
                p.requires_grad = False

        self.proj = nn.Linear(512, d)

    def forward(self, M: torch.Tensor) -> torch.Tensor:
        """
        Args:
            M: (B, 3, H, W) map image tensor (ImageNet-normalised)
        Returns:
            (B, P, d) map patch tokens where P = H'×W' (typically 7×7=49)
        """
        feats = self.features(M)          # (B, 512, H', W')
        B, C, Hf, Wf = feats.shape
        patches = feats.permute(0, 2, 3, 1).reshape(B, Hf * Wf, C)  # (B, P, 512)
        return self.proj(patches)          # (B, P, d)


# ---------------------------------------------------------------------------
# ContextEncoder — Eq. (8)
# ---------------------------------------------------------------------------

class ContextEncoder(nn.Module):
    """
    Encodes a natural-language behavioral context string into L tokens of dimension d.

    Architecture:
      Frozen SentenceBERT → fixed-length sentence embedding e ∈ R^{sent_dim}
      Trainable linear → L tokens each in R^d

    The sentence embedding is broadcast/projected to L tokens via a trainable MLP.
    The caller is expected to pass pre-computed SBERT embeddings (to avoid loading
    the heavy model inside this module on every device in DDP).  If a SentenceBERT
    model is provided at construction time it is stored and used during forward when
    raw strings are supplied.

    Args:
        d:        output token dimension
        L:        number of output context tokens (default 16)
        sent_dim: SentenceBERT output dimension (default 768 for 'all-MiniLM-L6-v2')
        sbert_model_name: model name passed to sentence_transformers.SentenceTransformer
    """

    def __init__(
        self,
        d: int,
        L: int = 16,
        sent_dim: int = 768,
        sbert_model_name: str = "all-MiniLM-L6-v2",
    ):
        super().__init__()
        self.L = L
        self.sent_dim = sent_dim
        # Trainable projection: R^{sent_dim} → R^{L*d}
        self.proj = _make_mlp([sent_dim, d, L * d])
        self.d = d

        # Optionally load frozen SentenceBERT
        self._sbert = None
        try:
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer(sbert_model_name)
            for p in self._sbert.parameters():
                p.requires_grad = False
        except ImportError:
            pass  # caller must supply pre-computed embeddings

    def encode_strings(self, texts: list[str], device: torch.device) -> torch.Tensor:
        """Encode raw strings to (B, sent_dim) using frozen SBERT."""
        assert self._sbert is not None, "sentence_transformers not installed; pass pre-computed embeddings."
        with torch.no_grad():
            embs = self._sbert.encode(texts, convert_to_tensor=True, device=device)
        return embs  # (B, sent_dim)

    def forward(self, C: torch.Tensor) -> torch.Tensor:
        """
        Args:
            C: (B, sent_dim) pre-computed SBERT embeddings, or (B, L, sent_dim)
               Already frozen embeddings — no gradients expected on C.
        Returns:
            (B, L, d) context tokens
        """
        if C.ndim == 3:
            # Already L tokens; project each
            return self.proj(C.reshape(-1, self.sent_dim)).reshape(C.shape[0], self.L, self.d)
        # C shape: (B, sent_dim)
        out = self.proj(C)          # (B, L*d)
        B = C.shape[0]
        return out.reshape(B, self.L, self.d)


# ---------------------------------------------------------------------------
# TimestepEmbedding — Eq. (9)
# ---------------------------------------------------------------------------

class TimestepEmbedding(nn.Module):
    """
    Encodes diffusion timestep t as a d-dimensional vector.

    h_t = MLP_t(SinEmb(t)) ∈ R^d
    Added as a global signal to all tokens (Eq. 10).
    """

    def __init__(self, d: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or d * 4
        self.mlp = _make_mlp([d, hidden_dim, d])
        self.d = d

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) integer timesteps
        Returns:
            (B, d) timestep embeddings
        """
        emb = sinusoidal_embedding(t, self.d)  # (B, d)
        return self.mlp(emb)                   # (B, d)
