import torch
from pathlib import Path

from .vision_transformer import vit_base  # 你贴的 DINOv3 版本 vit.py

CKPT = r"F:\读研\代码\exercise\first_idea\SAMdino\ckpt\dinov3\dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model", "backbone", "teacher", "student", "ema", "model_ema"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def strip_prefixes(sd):
    out = {}
    for k, v in sd.items():
        nk = k
        for p in ["module.", "backbone.", "model.", "encoder."]:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out


def detect_ffn_layer_v3(sd):
    keys = sd.keys()
    if any(".ffn." in k for k in keys):
        return "swiglu"
    if any(".mlp.fc1.weight" in k for k in keys):
        return "mlp"
    return "swiglu"


def infer_n_storage_tokens(sd):
    t = sd.get("storage_tokens", None)
    if isinstance(t, torch.Tensor) and t.ndim == 3:
        return t.shape[1]
    return 0


def has_layerscale(sd: dict) -> bool:
    return any(k.endswith("ls1.gamma") or k.endswith("ls2.gamma") for k in sd.keys())


def has_qkv_bias_mask(sd: dict) -> bool:
    return any(k.endswith("attn.qkv.bias_mask") for k in sd.keys())


def load_dinov3_vitb16(path: str):
    assert Path(path).exists(), f"Checkpoint not found: {path}"

    ckpt = torch.load(path, map_location="cpu")
    sd_raw = extract_state_dict(ckpt)
    sd = strip_prefixes(sd_raw)

    ffn_layer = detect_ffn_layer_v3(sd)
    n_storage_tokens = infer_n_storage_tokens(sd)

    # 🔎 自动探测
    use_layerscale = has_layerscale(sd)  # True -> 需要 ls1/ls2.gamma
    use_bias_mask = has_qkv_bias_mask(sd)  # True -> 需要 qkv.bias_mask
    layerscale_init = 1e-6 if use_layerscale else None

    print(f"[DINOv3] ffn_layer={ffn_layer}  n_storage_tokens={n_storage_tokens}  "
          f"layerscale_init={layerscale_init}  mask_k_bias={use_bias_mask}")

    # ✅ 构造时对齐 ckpt
    model = vit_base(
        patch_size=16,
        img_size=224,
        ffn_layer=ffn_layer,
        n_storage_tokens=n_storage_tokens,
        qkv_bias=True,  # 确保有 bias
        mask_k_bias=use_bias_mask,  # 需要时注册 bias_mask
        layerscale_init=layerscale_init,  # 需要时创建 ls1/ls2.gamma
    )

    model_sd = model.state_dict()
    ckpt_filtered = {k: v for k, v in sd.items() if not k.startswith("head.")}

    loadable, skipped, mismatched = {}, [], []
    for k, v in ckpt_filtered.items():
        if k in model_sd:
            if model_sd[k].shape == v.shape:
                loadable[k] = v
            else:
                mismatched.append((k, v.shape, model_sd[k].shape))
        else:
            skipped.append((k, v.shape, None))

    missing, unexpected = model.load_state_dict(loadable, strict=False)

    print(f"[DINOv3] loaded params: {len(loadable)}")
    print(f"[DINOv3] ckpt keys not used (extra): {len(skipped)}")
    print(f"[DINOv3] ckpt keys mismatched shape: {len(mismatched)}")
    print(f"[DINOv3] model params without pretrained weights: {len(missing)}")
    print(f"[DINOv3] unexpected keys after load_state_dict: {len(unexpected)}")
    if mismatched: print("  mismatched(example):", mismatched[:5])
    if skipped:    print("  extra_in_ckpt(example):", skipped[:5])
    if missing:    print("  missing_in_model(example):", missing[:5])
    if unexpected: print("  unexpected(example):", unexpected[:5])

    model.to(DEVICE).eval()
    return model


if __name__ == "__main__":
    model = load_dinov3_vitb16(CKPT)

    x = torch.randn(1, 3, 224, 224, device=DEVICE)
    with torch.no_grad():
        y = model(x)  # 默认返回 CLS token
    print("Output shape:", tuple(y.shape))
