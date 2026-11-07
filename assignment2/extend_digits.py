"""Generate 88 new source-guided TA-TiTok/MaskGen-L variants for each digit.

Use the original pilot's CPU environment. The pinned Hugging Face weights
download on first use unless a local weight tree is supplied. Resume with the
same command and output path.
"""
import argparse
import csv
import hashlib
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

from cpu_maskgen_models import WEIGHT_DIGESTS, load_models
from run_maskgen_cpu import PROJECT, run_lock


PRIOR_DIR = PROJECT / "outputs/reference_maskgen_cadcae3f15d2"
DATA_DIR = PROJECT.parent / "assignment1/data/mnist_subset"
OUT_DIR = PROJECT / "outputs/reference_maskgen_extension_880"
RESERVE_DIR = PROJECT / "outputs/reference_maskgen_reserve"
DATA_HASH = "d50edc0025cc483c5ba13a594221fc9a07ba84552dd79a9315e91b405b1623af"
PRIOR_HASH = "eb5c45bab298232720b46e32de4aa2014a32d774fa55ee544cc89dc43131815f"
STYLES = ("thin-brush", "marker-bleed", "rotation")
REVIEW_FIELDS = ("sample_id", "requested_digit", "path", "sha256", "status",
                 "reviewed_digit", "notes", "train_consent")
BASE_SEED = 100_000


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def checked_rows(path, expected):
    raw = path.read_text(encoding="utf-8")
    if digest(raw.encode()) != expected:
        raise ValueError(f"Unexpected manifest contents in {path}")
    return list(csv.DictReader(io.StringIO(raw)))


def atomic_write(path, raw):
    tmp = tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False)
    temp = Path(tmp.name)
    try:
        with tmp:
            tmp.write(raw)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def store_json(path, obj):
    atomic_write(path, (json.dumps(obj, indent=2) + "\n").encode())


def store_csv(path, rows):
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, out.getvalue().encode())


def allocate_references(rows, excluded, requests):
    hashes = {r["source_sha256"] for r in excluded}
    paths = {r["source_relative_path"] for r in excluded}
    indices = {int(r["source_index"]) for r in excluded}
    # Exclude exact pixel matches with validation and test even if their
    # official MNIST index belongs to another partition.
    hashes.update(r["sha256"] for r in rows if r["split"] != "train")
    by_digit = {digit: [] for digit in range(10)}
    for row in rows:
        if row["split"] == "train" and row["source_partition"] == "train":
            by_digit[int(row["label"])].append(row)
    streams = {digit: iter(group) for digit, group in by_digit.items()}
    plan = []
    for digit, ordinal, control, lineage in requests:
        for row in streams[digit]:
            path, sha, index = row["relative_path"], row["sha256"], int(row["source_index"])
            if path not in paths and sha not in hashes and index not in indices:
                break
        else:
            raise ValueError(f"Insufficient verified training references for digit {digit}")
        paths.add(path)
        hashes.add(sha)
        indices.add(index)
        sample_id = f"d{digit}-v{ordinal:03d}"
        plan.append(dict(sample_id=sample_id, requested_digit=digit, variant=ordinal,
                         control_variant=control, style=STYLES[(control - 1) // 4],
                         prompt=f"A single handwritten digit {digit}, isolated black strokes on a plain white field.",
                         seed=BASE_SEED + digit * 1000 + ordinal, path=f"candidates/{sample_id}.png",
                         source="synthetic", source_relative_path=path, source_sha256=sha,
                         source_index=index, source_split="train", source_partition="train", **lineage))
    return plan


def plan_extension(data_root, prior_root):
    old = checked_rows(prior_root / "manifest.csv", PRIOR_HASH)
    rows = checked_rows(data_root / "manifest.csv", DATA_HASH)
    # Start each class at a different style to balance the entire set.
    requests = [(d, n, 1 + (n - 13 + 4 * d) % 12, {}) for d in range(10) for n in range(13, 101)]
    return allocate_references(rows, old, requests), old


def plan_reserve(request_file, data_root, pilot, extension):
    with request_file.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["replaces", "reason"]:
            raise ValueError("Expected replacement CSV header 'replaces,reason'")
        reviews = list(reader)
    rejected = [r["replaces"] for r in reviews]
    if not reviews or len(set(rejected)) != len(reviews):
        raise ValueError("Expected one or more distinct rejected sample IDs")
    by_id = {row["sample_id"]: row for row in extension}
    next_ordinal = [101] * 10
    requests = []
    for review in reviews:
        sample_id, reason = review["replaces"], review["reason"].strip()
        if sample_id not in by_id or not reason:
            raise ValueError(f"Unknown raw extension ID or missing review reason: {sample_id}")
        replaced = by_id[sample_id]
        digit = int(replaced["requested_digit"])
        lineage = dict(replaces=sample_id, replaces_sha256=replaced["sha256"], replacement_reason=reason)
        requests.append((digit, next_ordinal[digit], int(replaced["control_variant"]), lineage))
        next_ordinal[digit] += 1
    refs = checked_rows(data_root / "manifest.csv", DATA_HASH)
    return allocate_references(refs, pilot + extension, requests)


def diagnostic_path(root, item):
    return root / "diagnostics" / f"{item['sample_id']}.json"


def identity(item):
    fields = ("sample_id", "seed", "source_relative_path", "source_sha256", "source_index",
              "variant", "control_variant", "replaces", "replaces_sha256", "replacement_reason")
    return {key: item[key] for key in fields if key in item}


def completed(root, item):
    try:
        record = json.loads(diagnostic_path(root, item).read_text(encoding="utf-8"))
        sha = record["sha256"]
        if (record["method"] == "reference_masked_tokens_v1" and record["changed_pixels"] > 0 and
                record["changed_code_count"] == (4, 6, 8, 5)[(item["control_variant"] - 1) % 4] and
                record["control"]["style"] == item["style"] and
                all(record[k] == v for k, v in identity(item).items()) and
                digest((root / item["path"]).read_bytes()) == sha):
            return sha
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        pass
    return None


def generate_one(item, root, data_root, models):
    from PIL import Image
    from glyph_synthesis import generate_variant
    with Image.open(data_root / item["source_relative_path"]) as source:
        if digest(source.convert("L").tobytes()) != item["source_sha256"]:
            raise ValueError(f"Source pixel digest mismatch: {item['source_relative_path']}")
        variant, record = generate_variant(source, {**item, "variant": item["control_variant"]}, *models)
    if variant.mode != "RGB" or variant.size != (256, 256):
        raise ValueError("Invalid synthetic image geometry")
    out = io.BytesIO()
    variant.save(out, format="PNG")
    raw = out.getvalue()
    record.update(identity(item))
    record["sha256"] = digest(raw)
    atomic_write(root / item["path"], raw)
    store_json(diagnostic_path(root, item), record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Pinned ByteDance checkout")
    parser.add_argument("--weights-dir", type=Path, help="Local weight tree; default downloads pinned Hugging Face files")
    parser.add_argument("--data-root", type=Path, default=DATA_DIR)
    parser.add_argument("--prior", type=Path, default=PRIOR_DIR)
    parser.add_argument("--output", type=Path, help="Separate output directory for this run")
    parser.add_argument("--replacements", type=Path, help="Generate additive QA reserves listed in a replaces,reason CSV")
    parser.add_argument("--raw-extension", type=Path, default=OUT_DIR, help="Completed raw 880 run for reserve generation")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=880, help="Round-robin prefix length for a partial run")
    args = parser.parse_args()
    if not 1 <= args.limit <= 880 or args.threads < 1:
        parser.error("Expected 1..880 items and a positive thread count")
    root = (args.output or (RESERVE_DIR if args.replacements else OUT_DIR)).resolve()
    if root == args.prior.resolve() or (args.replacements and root == args.raw_extension.resolve()):
        parser.error("Use a separate output directory to preserve earlier runs")
    root.mkdir(parents=True, exist_ok=True)
    for name in ("candidates", "diagnostics"):
        (root / name).mkdir(exist_ok=True)
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    import torch
    from glyph_synthesis import METHOD, METHOD_CONFIG
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    prior = json.loads((args.prior / "run.json").read_text(encoding="utf-8"))
    module_hash = digest((PROJECT / "glyph_synthesis.py").read_text(encoding="utf-8").encode())
    if (METHOD != prior["method"] or METHOD_CONFIG != prior["edit"] or module_hash != prior["implementation_sha256"]):
        raise ValueError("Glyph synthesis method differs from the original pilot")
    checkout = args.source.resolve()
    revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if revision != prior["source_revision"]:
        raise ValueError(f"Expected pinned source {prior['source_revision']}, found {revision}")
    sys.path.insert(0, str(checkout))
    plan, original = plan_extension(args.data_root, args.prior)
    runtime = dict(device="cpu", gpu=None, cuda=None, python=platform.python_version(),
                   packages={k: version(k) for k in prior["runtime"]["packages"]}, threads=args.threads)
    meta = {k: prior[k] for k in ("method", "implementation_sha256", "edit", "source_url", "source_revision",
                                  "weights", "a1_revision", "precision", "batch_size")}
    meta.update(reference_subset="A1 train", extension="880 additional examples, 88 per class",
                base_seed=BASE_SEED, runtime=runtime, source_manifest_sha256_lf=DATA_HASH,
                prior_manifest_sha256_lf=PRIOR_HASH, checkpoint_sha256=WEIGHT_DIGESTS,
                label_provenance="Inherited MNIST train labels; new samples pending visual review.",
                samples=plan)
    if args.replacements:
        # Verify the fixed raw plan and its actual PNG/diagnostic pairs before
        # linking a reserve to an exclusion. Leave the raw run unchanged.
        raw_dir = args.raw_extension
        if json.loads((raw_dir / "run.json").read_text(encoding="utf-8"))["samples"] != plan:
            raise ValueError("Raw extension source plan differs from the canonical 880")
        raw_rows = [{**item, "sha256": completed(raw_dir, item)} for item in plan]
        raw_manifest = (raw_dir / "manifest.csv").read_text(encoding="utf-8")
        recorded = list(csv.DictReader(io.StringIO(raw_manifest)))
        if (len(recorded) != len(plan) or any(not r["sha256"] for r in raw_rows) or
                recorded != [{k: str(v) for k, v in r.items()} for r in raw_rows]):
            raise ValueError("Raw extension must have all 880 verified PNGs and an accurate manifest")
        plan = plan_reserve(args.replacements, args.data_root, original, raw_rows)
        original += raw_rows
        meta.update(extension="Additive QA reserves for visually excluded raw candidates",
                    raw_manifest_sha256_lf=digest(raw_manifest.encode()),
                    replacements_sha256_lf=digest(args.replacements.read_text(encoding="utf-8").encode()),
                    samples=plan)
    selected = sorted(plan, key=lambda r: (r["variant"], r["requested_digit"]))[:args.limit]
    with run_lock(root / ".cpu.lock"):
        meta_path = root / "run.json"
        if meta_path.exists():
            if json.loads(meta_path.read_text(encoding="utf-8")) != meta:
                raise ValueError("Run configuration differs from the existing output")
        else:
            store_json(meta_path, meta)
        atomic_write(root / "SOURCE.txt", (args.prior / "SOURCE.txt").read_bytes())
        missing = [item for item in selected if not completed(root, item)]
        if missing:
            models = load_models(args.weights_dir, meta["weights"])
            start = time.monotonic()
            for n, item in enumerate(missing, 1):
                tick = time.monotonic()
                generate_one(item, root, args.data_root, models)
                print(f"{n + len(selected) - len(missing)}/{len(selected)} {item['sample_id']} "
                      f"{time.monotonic() - tick:.2f}s; mean {(time.monotonic() - start)/n:.2f}s", flush=True)
        rows = [{**item, "sha256": completed(root, item)} for item in plan]
        if any(r["sha256"] is None for r in rows):
            print(f"{sum(r['sha256'] is not None for r in rows)}/{len(plan)} complete")
            return
        all_hashes = [r["sha256"] for r in rows + original]
        if len(set(all_hashes)) != len(all_hashes):
            raise ValueError("Duplicate output image detected in the old and extended sets")
        store_csv(root / "manifest.csv", rows)
        if not (root / "review.csv").exists():
            reviews = [{k: r.get(k, "") for k in REVIEW_FIELDS} for r in rows]
            for row in reviews:
                row.update(status="pending")
            store_csv(root / "review.csv", reviews)
        print(f"Verified all {len(rows)} new images in {root}", flush=True)


if __name__ == "__main__":
    main()
