"""Source-guided MaskGen variants for isolated digits.

The reference controls the glyph geometry. Morphology or rotation supplies the
requested style. MaskGen refills selected TA-TiTok tokens, and the change it
makes to the decoded ink is applied locally to the styled reference.
"""
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import torch


METHOD = "reference_masked_tokens_v1"
METHOD_CONFIG = {
    "method": METHOD, "canvas_px": 256, "ink_box_px": 186, "ink_threshold": 40,
    "thin_radii": [1, 2, 3, 2], "thin_mix": [0.80, 0.85, 0.70, 1.0],
    "bleed_radii": [2, 3, 2, 4], "bleed_blur_sigmas": [0.65, 0.90, 0.45, 1.15],
    "rotation_ccw_degrees": [-8, 8, -14, 14], "masked_counts": [4, 6, 8, 5],
    "sampler_top_k": 8, "sampler_temperature": 0.8, "aesthetic_score": 6.5,
    "residual_clip_gray": 64, "halo_radius_px": 5, "halo_blur_sigma_px": 1.5,
}


def prepare_reference(source):
    """Center one dark glyph on a 256-pixel white field, preserving aspect ratio."""
    raw = ImageOps.exif_transpose(source).convert("RGBA")
    background = Image.new("RGBA", raw.size, "white")
    background.alpha_composite(raw)
    gray = background.convert("L")
    pixels = np.asarray(gray)
    border = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
    if np.median(border) < 128:
        gray = ImageOps.invert(gray)
    ink = ImageOps.invert(gray)
    box = ink.point(lambda value: 255 if value > METHOD_CONFIG["ink_threshold"] else 0).getbbox()
    if not box:
        raise ValueError("The reference contains no visible digit.")
    cropped = ink.crop(box)
    scaled = ImageOps.contain(cropped, (METHOD_CONFIG["ink_box_px"],) * 2, method=Image.Resampling.LANCZOS)
    side = METHOD_CONFIG["canvas_px"]
    field = Image.new("L", (side, side), 0)
    field.paste(scaled, ((side - scaled.width) // 2, (side - scaled.height) // 2))
    return ImageOps.invert(field).convert("RGB")


def style_reference(reference, variant):
    """Apply a known, recorded change in stroke thickness or glyph angle."""
    if not (1 <= variant <= 12):
        raise ValueError("Expected a variant number in 1..12.")
    ink = ImageOps.invert(reference.convert("L"))
    index = (variant - 1) % 4
    if variant <= 4:
        radius = METHOD_CONFIG["thin_radii"][index]
        amount = METHOD_CONFIG["thin_mix"][index]
        eroded = ink.filter(ImageFilter.MinFilter(2 * radius + 1))
        ink = Image.blend(ink, eroded, amount)
        control = dict(style="thin-brush", erosion_radius_px=radius, erosion_mix=amount)
    elif variant <= 8:
        radius = METHOD_CONFIG["bleed_radii"][index]
        blur = METHOD_CONFIG["bleed_blur_sigmas"][index]
        ink = ink.filter(ImageFilter.MaxFilter(2 * radius + 1)).filter(ImageFilter.GaussianBlur(blur))
        control = dict(style="marker-bleed", dilation_radius_px=radius, blur_sigma_px=blur)
    else:
        angle = METHOD_CONFIG["rotation_ccw_degrees"][index]
        ink = ink.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=0)
        control = dict(style="rotation", rotation_ccw_degrees=angle)
    return ImageOps.invert(ink).convert("RGB"), control


@torch.inference_mode()
def generate_variant(source, item, generator, tokenizer, clip, text_tokenizer, device, *, stages=None):
    """Return an RGB 256×256 image and JSON-compatible diagnostics.

    Set `stages={}` to also inspect the styled reference and both raw decoder
    outputs. A model, shape or numerical failure raises and produces no image.
    """
    from modeling.maskgen import open_clip_text_encoding

    variant, seed = int(item["variant"]), int(item["seed"])
    reference = prepare_reference(source)
    styled, control = style_reference(reference, variant)
    image = np.asarray(styled, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    _, quantized = tokenizer.encode(tensor)
    codes = quantized["min_encoding_indices"].reshape(1, -1).long()
    if codes.shape != (1, 128):
        raise RuntimeError(f"Unexpected TA-TiTok code shape {tuple(codes.shape)}")

    prompt = str(item["prompt"])
    context, pooled = open_clip_text_encoding(text_tokenizer, clip, [prompt])
    words = text_tokenizer([prompt]).to(device)
    guidance = clip.ln_final(clip.transformer(clip.token_embedding(words) + clip.positional_embedding,
                                             attn_mask=clip.attn_mask))
    # Seed on CPU for the same mask positions and multinomial draws on CPU/CUDA.
    rng = torch.Generator(device="cpu").manual_seed(seed)
    count = METHOD_CONFIG["masked_counts"][(variant - 1) % 4]
    positions = torch.randperm(128, generator=rng)[:count].sort().values.to(device)
    masked = codes.clone()
    masked[:, positions] = generator.mask_token_id
    score = torch.full((1,), METHOD_CONFIG["aesthetic_score"], device=device)
    logits, _ = generator(masked, context, pooled, aesthetic_score=score)
    logits = logits[0, positions].float()
    if not torch.isfinite(logits).all().item():
        raise RuntimeError("Non-finite MaskGen logits.")
    # Preserve every unmasked code. Among the eight highest-scoring available
    # alternatives, sample one new code for each selected position.
    logits.scatter_(1, codes[0, positions].unsqueeze(1), float("-inf"))
    values, indices = logits.topk(METHOD_CONFIG["sampler_top_k"], dim=-1)
    probabilities = (values / METHOD_CONFIG["sampler_temperature"]).softmax(-1).cpu()
    drawn = torch.multinomial(probabilities, 1, generator=rng).to(device)
    edited = codes.clone()
    edited[0, positions] = indices.gather(1, drawn).squeeze(1)

    # Reconstruct both code sequences under the SAME decoder text. Their
    # difference isolates the effect of resampling the selected MaskGen codes.
    pair = tokenizer.decode_tokens(torch.cat([codes, edited]), guidance.expand(2, -1, -1))
    if pair.shape != (2, 3, 256, 256) or not torch.isfinite(pair).all().item():
        raise RuntimeError("TA-TiTok returned invalid decoded pixels.")
    rgb = (pair.clamp(0, 1).mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy())
    base_recon, edited_recon = (Image.fromarray(value) for value in rgb)
    styled_ink = ImageOps.invert(styled.convert("L"))
    # At full intensity the local learned perturbation may change up to 64
    # grayscale levels. Beyond a 5 px dilated/softened stroke halo it vanishes.
    halo = styled_ink.filter(ImageFilter.MaxFilter(METHOD_CONFIG["halo_radius_px"] * 2 + 1))
    envelope = np.asarray(halo.filter(ImageFilter.GaussianBlur(METHOD_CONFIG["halo_blur_sigma_px"])),
                          dtype=np.float32) / 255.0
    anchor = np.asarray(styled_ink, dtype=np.float32)
    ink_difference = (np.asarray(base_recon.convert("L"), dtype=np.float32) -
                      np.asarray(edited_recon.convert("L"), dtype=np.float32))
    limit = METHOD_CONFIG["residual_clip_gray"]
    residual = envelope * np.clip(ink_difference, -limit, limit)
    final_ink = np.clip(np.rint(anchor + residual), 0, 255)
    result = Image.fromarray((255 - final_ink).astype(np.uint8)).convert("RGB")
    delta = abs(final_ink - anchor)
    support = envelope > 0.05
    # A numerically inert model edit is surfaced, never silently substituted.
    if not np.any(delta >= 1):
        raise RuntimeError("MaskGen token edit produced no visible pixel change.")
    info = dict(method=METHOD, control=control, seed=seed, mask_indices=positions.tolist(),
                original_codes=codes[0, positions].tolist(), replacement_codes=edited[0, positions].tolist(),
                changed_code_count=int((codes != edited).sum().item()),
                sampler_top_k=METHOD_CONFIG["sampler_top_k"], sampler_temperature=METHOD_CONFIG["sampler_temperature"],
                aesthetic_score=METHOD_CONFIG["aesthetic_score"], residual_clip_gray=limit,
                halo_radius_px=METHOD_CONFIG["halo_radius_px"], halo_blur_sigma_px=METHOD_CONFIG["halo_blur_sigma_px"],
                mean_absolute_change_on_stroke=float(delta[support].mean()),
                changed_pixels=int(np.count_nonzero(delta)), max_pixel_change=int(delta.max()))
    if stages is not None:
        stages.update(reference=reference, styled=styled, reconstruction=base_recon,
                      edited_decode=edited_recon)
    return result, info
