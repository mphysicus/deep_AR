import torch
import torch.nn as nn

from typing import Dict, List, Optional

from .layers import LoRALayer
from adalora import SVDLinear


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                hasattr(m, 'bias') and \
                m.bias is not None:
                    m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError
    
def convert_linear_to_adalora(
        module: nn.Module,
        r: int = 4,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None
) -> None:
    """
    Convert nn.Linear layers in a module to SVDLinear layers for AdaLoRA.
    """
    linear_layers = []
    for name, mod in module.named_modules():
        if isinstance(mod, nn.Linear):
            # Check if this layer should be converted
            if target_modules is None or _should_convert_layer(name, target_modules):
                linear_layers.append((name, mod))

    print(f"Found {len(linear_layers)} Linear layers to convert:")
    for name, _ in linear_layers:
        print(f" - {name}")
    
    # Replace the layers
    for name, linear_layer in linear_layers:
        new_Layer = SVDLinear(in_features=linear_layer.in_features,
                              out_features=linear_layer.out_features,
                              r=r,
                              lora_alpha=lora_alpha,
                              lora_dropout=lora_dropout,
                              bias=linear_layer.bias is not None)
        
        # Copy the original weights
        new_Layer.weight.data = linear_layer.weight.data.clone()
        if linear_layer.bias is not None:
            new_Layer.bias.data = linear_layer.bias.data.clone()

        # Replace
        _replace_module_by_name(module,name, new_Layer)
        print(f"Converted {name}")

def _should_convert_layer(layer_name:str, target_modules: List[str]) -> bool:
    """
    Check if a layer should be converted based on target_modules patterns
    """
    for pattern in target_modules:
        if pattern in layer_name:
            return True
    return False

def _replace_module_by_name(parent_module: nn.Module, module_name:str, new_module:nn.Module) -> None:
    """
    Replace a module by its name path
    """
    if '.' in module_name:
        parts = module_name.split('.')
        current_module = parent_module
        for part in parts[:-1]:
            current_module = getattr(current_module, part)
        setattr(current_module, parts[-1], new_module)
    else:
        setattr(parent_module, module_name, new_module)

def get_lora_model_from_pretrained(
        model: nn.Module,
        lora_config: Optional[Dict] = None, 
        target_modules: Optional[List[str]] = None) -> nn.Module:
    """
    Convert a pretrained model to use AdaLora.
    """
    if lora_config is None:
        lora_config = {
            'r':8,
            'lora_alpha':16,
            'lora_dropout':0.1
        }
    
    convert_linear_to_adalora(model, r=lora_config['r'],
                              lora_alpha=lora_config['lora_alpha'],
                              lora_dropout=lora_config['lora_dropout'],
                              target_modules=target_modules)
    
    mark_only_lora_as_trainable(model)
    return model

def print_trainable_parameters(model: nn.Module) -> None:
    """
    Print the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_params = 0
    lora_params = 0

    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        if 'lora_' in name:
            lora_params += param.numel()
    
    print(f"Trainable parameters: {trainable_params:,} ||"
          f"All params: {all_params:,} ||"
          f"Trainable%: {100*trainable_params/all_params:.2f}% ||"
          f"LoRA params: {lora_params:,}")