import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional

from .sam_no_prompt import SamAR
from .input_generator import IVT2RGB
from .map_reconstructor import Mask2ARMaps

class DeepAR(nn.Module):
    def __init__(self, 
                 sam_model: SamAR,
                 input_generator: IVT2RGB,
                 map_reconstructor: Mask2ARMaps,
                 mask_threshold: float = 0.0):
        super().__init__()
        self.sam_model = sam_model
        self.input_generator = input_generator
        self.map_reconstructor = map_reconstructor
        self.mask_threshold = mask_threshold

    def forward(self,
                x: torch.Tensor,
                multimask_output: bool = False,
                return_intermediate: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete DeepAR pipeline.

        Args:
            x: Input tensor (B, C, H, W)
            multimask_output: Whether to output multiple masks from SAM.
            return_intermediate: Whether to return intermediate outputs.

        Returns:
            A dictionary containing:
                - 'masks': Predicted masks from SAM.
                - 'ar_maps': Reconstructed AR maps.
                - 'intermediate': (Optional) Intermediate outputs if requested.
        """
        if isinstance(x, torch.Tensor):
            image_tensor = x
        else:
            image_tensor = torch.stack([d["image"] for d in x])
        outputs = {}

        # Step 1: Generate input features
        x_features = self.input_generator(image_tensor)
        if return_intermediate:
            outputs['input_features'] = x_features.detach()

        # Step 2: Predict masks using SAM
        batch_size = x_features.shape[0]
        batched_input = []
        for i in range(batch_size):
            batched_input.append({
                "image": x_features[i],
                "original_size": (x_features.shape[2], x_features.shape[3])
            })
        sam_outputs = self.sam_model(batched_input, multimask_output=multimask_output)

        #Extract masks
        masks = torch.stack([output['masks'].squeeze(1) for output in sam_outputs])

        if return_intermediate:
            outputs['masks'] = masks.detach()

        del sam_outputs, batched_input

        # Step 3: AR Map Reconstruction
        reconstructed = self.map_reconstructor(masks)
        outputs['output'] = reconstructed

        return outputs
    
    def get_binary_masks(self, x:torch.Tensor):
        """
        Get thresholded binary masks for inference/visualization.
        """
        with torch.no_grad():
            outputs = self.forward(x)
            binary_masks = (outputs['output'] > self.mask_threshold).float()
        return binary_masks
