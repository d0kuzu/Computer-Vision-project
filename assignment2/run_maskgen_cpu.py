"""Run the Colab digit pilot on a CPU using its pinned source and pretrained weights.

Run this script with the notebook's Python dependencies installed, using
`--source PATH` for a checkout of the pinned ByteDance repository. Models can
come from the Hugging Face cache or a SHA-256-verified `--weights-dir` tree
whose filenames are arranged as COMPONENT/HUB_REVISION/FILENAME.
"""
import argparse
import ast
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT = Path(__file__).resolve().parent
NOTEBOOK = PROJECT / "01_generate_digits_maskgen_colab.ipynb"
WEIGHT_DIGESTS = {
    "maskgen": "7021e110ee7d28fe34da72abcce6aa8f882d23f50d978bc7a41b95a2fed568d3",
    "tatitok": "4b7b4a6661285d6fb56565061712914a1458d39779f0d5a17f7ed87648f3f196",
    "clip": "fbc415c3d0d7b79faed8f5ccfb740c32b7c4f5ffe7283b851f89c6231c01a8e0",
}


@contextmanager
def run_lock(path):
    """Use an OS-owned lock, released if the process exits or is interrupted."""
    with path.open("a+b") as handle:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--data-root", type=Path, default=PROJECT.parent / "assignment1/data/mnist_subset")
    parser.add_argument("--output", type=Path, default=PROJECT / "outputs")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--limit", type=int, default=10,
                        help="10 previews by default. Use 120 to complete the reviewed recipe.")
    args = parser.parse_args()
    if args.threads < 1 or not 1 <= args.limit <= 120:
        parser.error("--threads must be positive and --limit must be between 1 and 120")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    import torch
    from PIL import Image

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    cells = {c["id"]: "".join(c["source"]) for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]}
    # Read the same literal seed and source pin as the notebook. The setup cell
    # itself installs Colab packages and is intentionally skipped locally.
    config = {}
    for node in ast.parse(cells["env"]).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {"BASE_SEED", "CODE_REV", "CODE_URL"}:
                config[name] = ast.literal_eval(node.value)
    checkout = args.source.resolve()
    head = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if head != config["CODE_REV"]:
        raise ValueError(f"Expected source revision {config['CODE_REV']}, found {head}")
    sys.path.insert(0, str(checkout))
    state = {**config, "torch": torch, "Path": Path, "Image": Image, "DEVICE": "cpu",
             "DATA_ROOT": args.data_root.resolve(), "OUTPUT_ROOT": args.output.resolve()}
    def execute(code, name):
        tree = ast.parse(code) if isinstance(code, str) else code
        # The CPU CLI writes the same previews to disk without requiring Jupyter.
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.ImportFrom) and node.module == "IPython.display"
        )]
        exec(compile(tree, f"{NOTEBOOK.name}#{name}", "exec"), state)

    for name in ("references", "glyph-method", "plan"):
        execute(cells[name], name)
    state["display"] = lambda image: print("Preview dimensions:", image.size)
    method_source = (PROJECT / "glyph_synthesis.py").read_text(encoding="utf-8")
    if (method_source != cells["glyph-method"] or
        hashlib.sha256(method_source.encode()).hexdigest() != state["METHOD_SOURCE_SHA256"]):
        raise ValueError("The source module hash differs from the notebook's pinned implementation.")
    run_dir, plan, pilot = state["RUN_DIR"], state["PLAN"], state["PILOT"]
    pilot_ids = set(state["PILOT_IDS"])
    selected = (pilot + [row for row in plan if row["sample_id"] not in pilot_ids])[:args.limit]
    validators = [node for node in ast.parse(cells["models"]).body
                  if isinstance(node, ast.FunctionDef) and node.name in {"complete_png", "complete_candidate"}]
    execute(ast.Module(body=validators, type_ignores=[]), "models")
    for name in ("inference", "preview-tools", "export-tools"):
        execute(cells[name], name)

    with run_lock(run_dir / ".cpu.lock"):
        missing = [item for item in selected if not state["complete_candidate"](item)]
        if missing:
            import open_clip
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            from modeling.maskgen import MaskGen_VQ
            from modeling.tatitok import TATiTok

            def weight_file(component, name):
                repository, revision = state["WEIGHTS"][component]
                if args.weights_dir:
                    path = args.weights_dir / component / revision / name
                else:
                    path = Path(hf_hub_download(repo_id=repository, revision=revision, filename=name))
                if name.endswith(".safetensors"):
                    with path.open("rb") as f:
                        digest = hashlib.file_digest(f, "sha256").hexdigest()
                    if digest != WEIGHT_DIGESTS[component]:
                        raise ValueError(f"Invalid {component} checkpoint digest {digest}")
                return path

            def load_component(name, cls):
                cfg = json.loads(weight_file(name, "config.json").read_text(encoding="utf-8"))
                # Assign the published tensors directly to an empty model. This
                # avoids a second multi-GB set of random parameters in RAM.
                with torch.device("meta"):
                    model = cls(cfg)
                weights = load_file(str(weight_file(name, "model.safetensors")), device="cpu")
                if name == "maskgen":
                    weights["lm_head.weight"] = weights["embeddings.weight"]
                model.load_state_dict(weights, assign=True, strict=True)
                if name == "maskgen":
                    model.lm_head.weight = model.embeddings.weight
                assert all(t.device.type == "cpu" for t in (*model.parameters(), *model.buffers()))
                return model.requires_grad_(False).eval()

            print("Loading OpenCLIP text tower…", flush=True)
            clip = open_clip.create_model("ViT-L-14-336", force_quick_gelu=True,
                                         pretrained=str(weight_file("clip", "open_clip_model.safetensors")),
                                         device="cpu")
            del clip.visual
            state["clip"] = clip.requires_grad_(False).eval()
            state["text_tokenizer"] = open_clip.get_tokenizer("ViT-L-14-336")
            print("Loading MaskGen-L VQ…", flush=True)
            state["generator"] = load_component("maskgen", MaskGen_VQ)
            print("Loading TA-TiTok BL128 VQ…", flush=True)
            tokenizer = load_component("tatitok", TATiTok)
            state["tokenizer"] = tokenizer
            gc.collect()
            assert state["generator"].image_seq_len == tokenizer.num_latent_tokens == 128
            assert tokenizer.quantize_mode == "vq" and clip.transformer.batch_first

            for item in missing:
                print("Generating", item["sample_id"], flush=True)
                start = time.monotonic()
                state["ensure_candidate"](item)
                elapsed = time.monotonic() - start
                print(f"Saved {item['sample_id']} in {elapsed:.1f} s", flush=True)
        if all(state["complete_candidate"](item) for item in pilot):
            execute(cells["pilot"], "pilot")
        execute(cells["pilot-package"], "pilot-package")
        print(f"{sum(state['complete_candidate'](x) for x in plan)}/120 saved in {run_dir}.")


if __name__ == "__main__":
    main()
