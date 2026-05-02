import torch

from functools import partial

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer
import torch.nn.functional as F


def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint=None, w=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
        w=w
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def load_sam_checkpoint_with_interpolation(sam_model, checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}...")
    with open(checkpoint_path, "rb") as f:
        state_dict = torch.load(f, map_location="cpu")

    # 获取模型当前的 state_dict 作为参考
    model_dict = sam_model.state_dict()

    # 创建一个新的字典用于更新
    new_state_dict = {}

    for k, v in state_dict.items():
        if k in model_dict:
            # 获取模型中该参数的目标形状
            target_shape = model_dict[k].shape

            # 如果形状匹配，直接使用
            if v.shape == target_shape:
                new_state_dict[k] = v

            # 如果是相对位置编码 (rel_pos) 且形状不匹配，进行插值
            elif "rel_pos" in k:
                # print(f"Resizing {k}: {v.shape} -> {target_shape}")

                # v 的形状通常是 [2*Win-1, Head_Dim] (例如 [27, 64])
                # 变为 [1, Head_Dim, 2*Win-1] 以适应 interpolate
                v_reshaped = v.unsqueeze(0).permute(0, 2, 1)

                # 目标长度 (例如 127 或 23)
                target_len = target_shape[0]

                # 执行线性插值
                v_interp = F.interpolate(v_reshaped, size=target_len, mode='linear', align_corners=False)

                # 变回原来的形状 [Target_Len, Head_Dim]
                v_final = v_interp.permute(0, 2, 1).squeeze(0)

                new_state_dict[k] = v_final

            # 如果是其他参数不匹配（通常不应该发生），忽略它让模型保持随机初始化，或者报错
            else:
                print(f"WARNING: Shape mismatch for {k}, skipping. Ckpt: {v.shape}, Model: {target_shape}")
                continue
        else:
            # 如果模型里没有这个 key (比如你删减了层)，就忽略
            pass

    # 加载处理后的权重，strict=False 允许你的新模块 (SpectralFusion) 保持随机初始化
    sam_model.load_state_dict(new_state_dict, strict=False)
    print("Checkpoint loaded successfully with interpolation.")


# ==========================================
# 修改你的 _build_sam 函数
# ==========================================
def _build_sam(
        encoder_embed_dim,
        encoder_depth,
        encoder_num_heads,
        encoder_global_attn_indexes,
        checkpoint=None,
        w=None
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    # 实例化 SAM
    sam = Sam(
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
            window_size=0,
            out_chans=prompt_embed_dim,
            w=w
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
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
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    sam.eval()

    # 使用新的加载函数
    if checkpoint is not None:
        load_sam_checkpoint_with_interpolation(sam, checkpoint)

    return sam
