import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LoRA(nn.Module):
    """
    Lightweight wrapper that injects a low-rank adapter into an existing nn.Linear layer.

    Given a base weight W (out_features x in_features), we learn a low-rank update
    ΔW = (lora_B @ lora_A) scaled by alpha/rank, where:
      - lora_A has shape (rank, in_features)
      - lora_B has shape (out_features, rank)

    Forward computes: base(x) + dropout(x) @ lora_A.T @ lora_B.T * scaling.
    Initially, lora_B is zero to preserve the original model's behavior.
    """

    def __init__(
        self,
        layer: nn.Linear,
        rank: int = 30,
        alpha: int = 16,
        dropout: float = 0.0,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(layer, nn.Linear):
            raise TypeError("LoRA only supports wrapping nn.Linear layers.")
        if rank <= 0:
            raise ValueError("LoRA rank must be > 0")

        self.layer = layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / float(rank)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else (lambda x: x)

        # Create LoRA parameters on the same device/dtype as the base layer
        in_features = layer.in_features
        out_features = layer.out_features
        w = layer.weight
        self.lora_A = nn.Parameter(w.new_zeros((rank, in_features)))
        self.lora_B = nn.Parameter(w.new_zeros((out_features, rank)))

        # Init: A ~ Kaiming, B = 0 so initial delta is zero
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Optionally freeze the base layer's weights
        if freeze_base:
            for p in self.layer.parameters():
                p.requires_grad = False

        self.merged = False

    def delta_weight(self) -> torch.Tensor:
        """Compute ΔW = (lora_B @ lora_A) * scaling with shape (out_features, in_features)."""
        return (self.lora_B @ self.lora_A) * self.scaling

    def to_linear_merged(self) -> nn.Linear:
        """Return a fresh nn.Linear with merged weights (W + ΔW) and copied bias."""
        base = self.layer
        merged = nn.Linear(base.in_features, base.out_features, bias=base.bias is not None)
        # Match dtype/device
        merged = merged.to(base.weight.device, dtype=base.weight.dtype)
        with torch.no_grad():
            merged.weight.copy_(base.weight + self.delta_weight())
            if base.bias is not None:
                merged.bias.copy_(base.bias)
        return merged

    def merge_inplace(self) -> None:
        """In-place merge: update base layer's weight to include ΔW and flag merged."""
        if not self.merged:
            with torch.no_grad():
                self.layer.weight.add_(self.delta_weight())
            self.merged = True

    def unmerge_inplace(self) -> None:
        """In-place unmerge: subtract ΔW from base layer's weight if merged."""
        if self.merged:
            with torch.no_grad():
                self.layer.weight.sub_(self.delta_weight())
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If weights are merged into base, we just defer to the base layer
        if self.merged:
            return self.layer(x)
        # base output
        result = self.layer(x)
        # low-rank adapter output: F.linear applies weight with shape (out, in)
        after_A = F.linear(self.dropout(x), self.lora_A)  # (batch, rank)
        delta = F.linear(after_A, self.lora_B)            # (batch, out_features)
        return result + delta * self.scaling

