from .lora import LoRA
import torch
import torch.nn as nn
from typing import Dict, Iterable, List, Optional


def _iter_target_linear_modules(model: nn.Module, targets: Iterable[str]):
    """Yield (name, module) for Linear submodules whose qualified name contains any target substrings."""
    targets = list(targets)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not isinstance(module, LoRA):
            if any(t in name for t in targets):
                yield name, module


def _get_parent_and_attr(model: nn.Module, qualified_name: str):
    parent = model
    parts = qualified_name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def apply_lora_to_sam(
    model: nn.Module,
    rank: int,
    alpha: int,
    target_layers: List[str] = ("attn.qkv", "attn.proj"),
    dropout: float = 0.0,
    verbose: Optional[bool] = None,
):
    """Wrap selected Linear layers with LoRA.

    - Avoids double-wrapping.
    - Keeps device/dtype consistent with the original layer.
    """
    # Collect first to avoid mutating while iterating named_modules
    to_wrap = list(_iter_target_linear_modules(model, target_layers))
    for name, linear in to_wrap:
        wrapper = LoRA(linear, rank=rank, alpha=alpha, dropout=dropout)
        # Ensure same device/dtype
        wrapper = wrapper.to(linear.weight.device, dtype=linear.weight.dtype)
        parent, attr = _get_parent_and_attr(model, name)
        setattr(parent, attr, wrapper)
        if verbose:
            print(f"Applied LoRA to {name}")
    return model


def merge_lora(model: nn.Module, verbose: Optional[bool] = None) -> nn.Module:
    """Replace all LoRA wrappers in the model with merged nn.Linear layers.

    This removes the LoRA overhead for inference by folding ΔW into the base weights.
    """
    # We need qualified names, so iterate named_modules at the top-level model
    # but don't mutate while iterating; collect names first.
    lora_modules: List[tuple[str, LoRA]] = []
    for name, module in model.named_modules():
        if isinstance(module, LoRA):
            lora_modules.append((name, module))

    for name, lora_mod in lora_modules:
        parent, attr = _get_parent_and_attr(model, name)
        merged_linear = lora_mod.to_linear_merged()
        setattr(parent, attr, merged_linear)
        if verbose:
            print(f"Merged LoRA into {name}")
    return model


def mark_only_lora_as_trainable(model: nn.Module) -> None:
    """Freeze all parameters except LoRA adapter matrices."""
    for p in model.parameters():
        p.requires_grad = False
    for _, module in model.named_modules():
        if isinstance(module, LoRA):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return a state dict containing only LoRA parameters with qualified names."""
    sd: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRA):
            sd[f"{name}.lora_A"] = module.lora_A.detach().cpu()
            sd[f"{name}.lora_B"] = module.lora_B.detach().cpu()
    return sd