"""Run the Colab digit pilot on a CPU using its pinned source and pretrained weights.

Run this script with the notebook's Python dependencies installed, using
`--source PATH` for a checkout of the pinned ByteDance repository. Models can
come from the Hugging Face cache or a SHA-256-verified `--weights-dir` tree
whose filenames are arranged as COMPONENT/HUB_REVISION/FILENAME.
"""
import argparse
import ast
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from cpu_maskgen_models import load_models


PROJECT = Path(__file__).resolve().parent
NOTEBOOK = PROJECT / "01_generate_digits_maskgen_colab.ipynb"


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
            generator, tokenizer, clip, text_tokenizer, _ = load_models(args.weights_dir, state["WEIGHTS"])
            state.update(generator=generator, tokenizer=tokenizer, clip=clip, text_tokenizer=text_tokenizer)

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
