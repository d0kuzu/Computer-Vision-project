"""Load the pinned CPU digit-generation models from local weights or the Hub cache."""
import hashlib
import json
from pathlib import Path


WEIGHT_DIGESTS = {
    "maskgen": "7021e110ee7d28fe34da72abcce6aa8f882d23f50d978bc7a41b95a2fed568d3",
    "tatitok": "4b7b4a6661285d6fb56565061712914a1458d39779f0d5a17f7ed87648f3f196",
    "clip": "fbc415c3d0d7b79faed8f5ccfb740c32b7c4f5ffe7283b851f89c6231c01a8e0",
}
CONFIG_DIGESTS = {
    "maskgen": "eb7d9c7257ac0a7bbfa17098e55e5151fdfce030935d63fabaf1f125baf1a47a",
    "tatitok": "7611ddacb574d38f8da14eddc840cbf5635b19311c052117755faca8092e4c99",
}


def load_models(weights_dir, pins):
    import gc
    from huggingface_hub import hf_hub_download
    import open_clip
    import torch
    from modeling.maskgen import MaskGen_VQ
    from modeling.tatitok import TATiTok
    from safetensors.torch import load_file

    def file(component, name):
        repo, revision = pins[component]
        if weights_dir is None:
            path = Path(hf_hub_download(repo_id=repo, revision=revision, filename=name))
        else:
            path = weights_dir / component / revision / name
        expected = CONFIG_DIGESTS[component] if name == "config.json" else WEIGHT_DIGESTS[component]
        with path.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != expected:
                raise ValueError(f"Invalid model file digest: {path}")
        return path

    def model(component, cls):
        cfg = json.loads(file(component, "config.json").read_text(encoding="utf-8"))
        with torch.device("meta"):
            net = cls(cfg)
        weights = load_file(str(file(component, "model.safetensors")), device="cpu")
        if component == "maskgen":
            weights["lm_head.weight"] = weights["embeddings.weight"]
        net.load_state_dict(weights, assign=True, strict=True)
        if component == "maskgen":
            net.lm_head.weight = net.embeddings.weight
        if any(t.device.type != "cpu" for t in (*net.parameters(), *net.buffers())):
            raise ValueError(f"Incomplete model load for {component}")
        return net.requires_grad_(False).eval()

    print("Checking and loading OpenCLIP", flush=True)
    clip = open_clip.create_model("ViT-L-14-336", force_quick_gelu=True,
                                  pretrained=str(file("clip", "open_clip_model.safetensors")), device="cpu")
    del clip.visual
    clip.requires_grad_(False).eval()
    text = open_clip.get_tokenizer("ViT-L-14-336")
    print("Checking and loading MaskGen-L VQ", flush=True)
    generator = model("maskgen", MaskGen_VQ)
    print("Checking and loading TA-TiTok BL128 VQ", flush=True)
    tokenizer = model("tatitok", TATiTok)
    gc.collect()
    if not (generator.image_seq_len == tokenizer.num_latent_tokens == 128 and
            tokenizer.quantize_mode == "vq" and clip.transformer.batch_first):
        raise ValueError("Incompatible generator, tokenizer, or text encoder")
    return generator, tokenizer, clip, text, "cpu"
