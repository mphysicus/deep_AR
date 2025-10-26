import torch
from functools import partial

from .modeling import ImageEncoderViT, MaskDecoder, TwoWayTransformer
from .modeling.sam_no_prompt import SamAR
from .modeling.deep_ar import DeepAR
from .modeling.input_generator import IVT2RGB
from .modeling.map_reconstructor import Mask2ARMaps


def build_deep_ar_vit_h(checkpoint=None):
    return _build_deep_ar(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


def build_deep_ar_vit_l(checkpoint=None):
    return _build_deep_ar(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )

deep_ar_model_registry = {
    "default": build_deep_ar_vit_h,
    "vit_h": build_deep_ar_vit_h,
    "vit_l": build_deep_ar_vit_l,
}

def _build_deep_ar(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam_ar = SamAR(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
    )

    input_generator = IVT2RGB()
    map_reconstructor = Mask2ARMaps()
    deep_ar = DeepAR(
        sam_model=sam_ar,
        input_generator=input_generator,
        map_reconstructor=map_reconstructor,)
    
    sam_ar.eval()
    if checkpoint is not None:
        print(f"Loading model from {checkpoint}")
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        deep_ar.load_state_dict(state_dict)
    return deep_ar