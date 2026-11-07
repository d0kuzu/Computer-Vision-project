"""Fine-tune A1 with matched real replay and reviewed synthetic data on CPU.

Run from any directory:
  python train_compare.py --pilot outputs/<run>
  python train_compare.py                  # A1 + real-replay control only
"""
import argparse
import copy
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
A1 = ROOT.parent / "assignment1"
sys.path.insert(0, str(A1))
from predict import load_model, preprocess

REAL_BATCH = 56
EXTRAS = 8
DIGITS = list("0123456789")
LR = 0.0001
REFERENCE_METHOD = "reference_masked_tokens_v1"
REFERENCE_FIELDS = ("source_relative_path", "source_split", "source_partition", "source_index", "source_sha256")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_mnist():
    folder = A1 / "data/mnist_subset"
    splits = defaultdict(list)
    hashes, ids = set(), set()
    for row in read_csv(folder / "manifest.csv"):
        split, label = row["split"], int(row["label"])
        source_id = (row["source_partition"], row["source_index"])
        if (split not in {"train", "val", "test"} or str(label) not in DIGITS
                or row["class_name"] != str(label)
                or source_id[0] != ("test" if split == "test" else "train")
                or source_id in ids or row["sha256"] in hashes):
            raise ValueError(f"Invalid or overlapping A1 record: {row['relative_path']}")
        ids.add(source_id)
        hashes.add(row["sha256"])
        splits[split].append(row)
    for split, per_class in [("train", 140), ("val", 30), ("test", 30)]:
        if Counter(r["class_name"] for r in splits[split]) != dict.fromkeys(DIGITS, per_class):
            raise ValueError(f"Unexpected A1 class counts for {split}")
    return folder, splits


def load_split(folder, records):
    tensors = []
    for r in records:
        with Image.open(folder / r["relative_path"]) as im:
            gray = im.convert("L")
            if gray.size != (28, 28) or hashlib.sha256(gray.tobytes()).hexdigest() != r["sha256"]:
                raise ValueError(f"Changed MNIST image: {r['relative_path']}")
            tensors.append(preprocess(gray))
    return torch.cat(tensors), torch.tensor([int(r["label"]) for r in records])


def check_pilot_sources(run, records, real_train):
    method = run.get("method")
    plan = run.get("samples", [])
    if method != REFERENCE_METHOD:
        if method is not None or any(row.get(k) for row in [*records, *plan] for k in REFERENCE_FIELDS):
            raise ValueError(f"Unrecognized generator method or undeclared reference images: {method!r}")
        return {"method": "text_only"}

    planned = {r["sample_id"]: r for r in plan}
    if len(plan) != len(planned) or set(planned) != {r["sample_id"] for r in records}:
        raise ValueError("Reference candidates in run.json and manifest.csv differ")
    # The original A1 validation rows also have source_partition=train. Restrict the
    # lookup to the project-level train split, whose PNGs load_split verified.
    originals = {r["relative_path"]: r for r in real_train if r["split"] == "train"}
    for r in records:
        sample = r["sample_id"]
        declared = planned[sample]
        if any(str(r.get(k, "")) != str(declared.get(k, "")) for k in (*REFERENCE_FIELDS, "requested_digit", "path")):
            raise ValueError(f"Reference provenance disagrees between manifest.csv and run.json for {sample}")
        original = originals.get(r.get("source_relative_path"))
        if (original is None or r.get("source_split") != "train" or r.get("source_partition") != "train"
                or r.get("source_index") != original["source_index"] or r.get("source_sha256") != original["sha256"]
                or r["requested_digit"] != original["class_name"]):
            raise ValueError(f"Reference for {sample} does not match a verified A1 training image of its requested digit")
    return {"method": method, "reference_sources": len({r["source_relative_path"] for r in records})}


def load_pilot(pilot, review, real_train):
    manifest_path = pilot / "manifest.csv"
    run_path = pilot / "run.json"
    records = read_csv(manifest_path)
    manifest = {r["sample_id"]: r for r in records}
    review_rows = read_csv(review)
    if (not manifest or len(manifest) != len(records) or len(review_rows) != len(manifest)
            or {r["sample_id"] for r in review_rows} != manifest.keys()):
        raise ValueError("Review and manifest must cover the same distinct sample IDs")
    source_info = check_pilot_sources(json.loads(run_path.read_text(encoding="utf-8")), records, real_train)
    tensors, labels = [], []
    review_counts = Counter()
    accepted_hashes = set()
    accepted_sources = set()
    withheld = 0
    for r in sorted(review_rows, key=lambda entry: entry["sample_id"]):
        m = manifest[r["sample_id"]]
        if any(r[k] != m[k] for k in ("requested_digit", "path", "sha256")):
            raise ValueError(f"Review does not match manifest for {r['sample_id']}")
        if m["source"] != "synthetic" or m["requested_digit"] not in DIGITS:
            raise ValueError(f"Invalid generator manifest entry: {r['sample_id']}")
        if r["status"] not in {"accept", "reject", "pending"}:
            raise ValueError(f"Unknown review status: {r['status']}")
        review_counts[r["status"]] += 1
        if r["status"] != "accept":
            continue
        if r["reviewed_digit"] not in DIGITS:
            raise ValueError(f"Missing human-readable digit for {r['sample_id']}")
        # An explicit UI consent field takes precedence. In older Colab reviews,
        # the reviewer made the train-only decision through the accept status.
        if r.get("train_consent", "train_only") != "train_only":
            withheld += 1
            continue
        path = (pilot / r["path"]).resolve()
        if not path.is_relative_to(pilot.resolve()) or path.suffix.lower() != ".png":
            raise ValueError(f"Expected PNG inside pilot: {r['path']}")
        if sha_file(path) != r["sha256"] or r["sha256"] in accepted_hashes:
            raise ValueError(f"Changed or duplicate accepted PNG: {r['sample_id']}")
        accepted_hashes.add(r["sha256"])
        with Image.open(path) as im:
            if im.format != "PNG":
                raise ValueError(f"Invalid accepted image: {path}")
            tensors.append(preprocess(im))
        labels.append(int(r["reviewed_digit"]))
        if source_info["method"] == REFERENCE_METHOD:
            accepted_sources.add(m["source_relative_path"])
    if not tensors:
        raise ValueError("Pilot has no accepted PNGs with a reviewed_digit enabled for training")
    if source_info["method"] == REFERENCE_METHOD:
        source_info["accepted_reference_sources"] = len(accepted_sources)
    return (torch.cat(tensors), torch.tensor(labels)), {
        **source_info, "run": pilot.resolve().name, "run_sha256": sha_file(run_path),
        "manifest_sha256": sha_file(manifest_path), "withheld_for_consent": withheld,
        "review_sha256": sha_file(review), "review_counts": dict(review_counts),
        "accepted_by_digit": {str(i): labels.count(i) for i in range(10)},
    }


@torch.inference_mode()
def evaluate(model, data):
    x, y = data
    model.eval()
    logits = torch.cat([model(chunk) for chunk in x.split(64)])
    predictions = logits.argmax(dim=1)
    confusion = torch.bincount(y * 10 + predictions, minlength=100).reshape(10, 10)
    true_positive = confusion.diag()
    f1_denominator = confusion.sum(0) + confusion.sum(1)
    scores = {
        "loss": float(F.cross_entropy(logits, y)), "n": len(y),
        "accuracy": int(true_positive.sum()) / len(y),
        "macro_f1": float((2 * true_positive.double() / f1_denominator.clamp_min(1)).mean()),
    }
    return scores, predictions.tolist(), confusion.tolist()


def schedule(train_labels, epochs, seed):
    """Keep real examples, extra labels and their real fallback identical in both runs."""
    rng = torch.Generator().manual_seed(seed)
    by_class = [torch.where(train_labels == i)[0] for i in range(10)]
    batches = len(train_labels) // REAL_BATCH
    assert batches * REAL_BATCH == len(train_labels) and batches * EXTRAS % 10 == 0
    plan = []
    for _ in range(epochs):
        order = torch.randperm(len(train_labels), generator=rng).view(batches, REAL_BATCH)
        labels = (torch.randperm(batches * EXTRAS, generator=rng) % 10).reshape(batches, EXTRAS)
        replay = torch.empty_like(labels)
        for i, pool in enumerate(by_class):
            mask = labels == i
            replay[mask] = pool[torch.randperm(len(pool), generator=rng)[:int(mask.sum())]]
        plan.append((order, labels, replay))
    return plan


def validation_rank(scores):
    return scores["macro_f1"], -scores["loss"]


def history_row(name, epoch, model, train, val):
    train_scores = evaluate(model, train)[0]
    val_scores = evaluate(model, val)[0]
    return {"model": name, "epoch": epoch, "train_loss": train_scores["loss"],
            "val_loss": val_scores["loss"], "train_accuracy": train_scores["accuracy"],
            "val_accuracy": val_scores["accuracy"], "val_macro_f1": val_scores["macro_f1"]}, val_scores


def train_variant(name, original, train, val, plan, seed, synthetic=None):
    model = copy.deepcopy(original)
    torch.manual_seed(seed)  # Reset the dropout sequence for each branch.
    synthetic_rng = torch.Generator().manual_seed(seed + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    x, y = train
    pools = [] if synthetic is None else [torch.where(synthetic[1] == i)[0] for i in range(10)]
    history = []
    selected = None
    selected_state = None
    for epoch, (order, extra_labels, replay) in enumerate(plan, start=1):
        draws = replay.clone()
        replace = torch.zeros_like(extra_labels, dtype=torch.bool)
        for digit, pool in enumerate(pools):
            if len(pool):
                mask = extra_labels == digit
                repeats = (int(mask.sum()) + len(pool) - 1) // len(pool)
                indices = torch.cat([pool[torch.randperm(len(pool), generator=synthetic_rng)] for _ in range(repeats)])
                draws[mask] = indices[:int(mask.sum())]
                replace[mask] = True
        model.train()
        for real, cls, extra, ids, use_synthetic in zip(order, extra_labels, replay, draws, replace):
            extra_x = x[extra]
            if synthetic is not None:
                extra_x[use_synthetic] = synthetic[0][ids[use_synthetic]]
            inputs = torch.cat([x[real], extra_x])
            targets = torch.cat([y[real], cls])
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(inputs), targets).backward()
            optimizer.step()
        row, val_scores = history_row(name, epoch, model, train, val)
        history.append(row)
        print(f"{name} epoch {epoch}: real val F1={row['val_macro_f1']:.4f}, loss={row['val_loss']:.5f}", flush=True)
        if selected is None or validation_rank(val_scores) > validation_rank(selected["validation"]):
            selected = {
                "epoch": epoch, "validation": val_scores, "optimizer_steps_per_epoch": len(order),
                "training_originals_by_source": {
                    "real": len(y), "synthetic": 0 if synthetic is None else len(synthetic[1]),
                },
                "source_draws_per_epoch": {"real_original": len(y), "real_replay": int((~replace).sum()),
                                           "synthetic": int(replace.sum())},
            }
            selected_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(selected_state)
    return model.eval(), selected, history


def plot_history(path, history, names, synthetic_label="Reviewed synthetic treatment"):
    image = Image.new("RGB", (1050, 500), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=16)
    colors = {"control": "#246889", "synthetic": "#b34124"}
    draw.text((35, 18), "A2 fine-tuning on the fixed A1 split", "#172532", font=ImageFont.load_default(size=24))
    for panel, (field, title) in enumerate([("train_loss", "Real training loss"), ("val_loss", "Real validation loss")]):
        x0, x1, y0, y1 = 65 + panel * 525, 482 + panel * 525, 94, 391
        draw.text((x0, 60), title + " (cross-entropy)", "#172532", font=font)
        highest = max(r[field] for r in history) * 1.12 or 1
        epochs = max(r["epoch"] for r in history)
        for tick in range(5):
            y = y1 - (y1-y0)*tick/4
            draw.line((x0, y, x1, y), fill="#d7e0e2", width=1)
            draw.text((x0-7, y), f"{highest*tick/4:.3f}", "#4f6167", font=font, anchor="rm")
        for epoch in range(epochs + 1):
            px = x0 + (x1 - x0) * epoch / epochs
            draw.text((px, y1+8), str(epoch), "#4f6167", font=font, anchor="mt")
        draw.text(((x0+x1)/2, y1+29), "Epoch", "#4f6167", font=font, anchor="mt")
        for name in names:
            series = [r for r in history if r["model"] in {"a1", name}]
            points = [(x0+(x1-x0)*r["epoch"]/epochs, y1-(y1-y0)*r[field]/highest) for r in series]
            draw.line(points, fill=colors[name], width=3)
            for px, py in points:
                draw.ellipse((px-3, py-3, px+3, py+3), fill=colors[name])
    for i, name in enumerate(names):
        x = 70 + i*320
        draw.line((x, 464, x+28, 464), fill=colors[name], width=3)
        draw.text((x+37, 464), {"control":"Real replay control", "synthetic":synthetic_label}[name], "#273742", font=font, anchor="lm")
    image.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot", type=Path, help="Folder containing manifest.csv, review.csv and candidates/")
    parser.add_argument("--review", type=Path, help="Alternative reviewer CSV for the same pilot")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1 or (args.review and not args.pilot):
        parser.error("epochs/threads must be positive, and --review requires --pilot")
    base, classes = load_model()
    if classes != DIGITS:
        raise ValueError("Expected the A1 ten-digit checkpoint")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    folder, splits = read_mnist()
    train, val = (load_split(folder, splits[name]) for name in ("train", "val"))
    pilot_data, provenance = None, None
    if args.pilot:
        pilot_data, provenance = load_pilot(args.pilot, args.review or args.pilot / "review.csv", splits["train"])
    plan = schedule(train[1], args.epochs, args.seed)
    first, base_val = history_row("a1", 0, base, train, val)
    history = [first]
    models = {"a1": {"epoch": 0, "validation": base_val,
                     "training_originals_by_source": {"real": len(train[1]), "synthetic": 0}}}
    checkpoints = {"a1": base}
    variants = {"control": None}
    if pilot_data is not None:
        variants["synthetic"] = pilot_data
    for name, data in variants.items():
        model, selected, rows = train_variant(name, base, train, val, plan, args.seed, data)
        models[name] = selected
        checkpoints[name] = model
        history.extend(rows)
    chosen = max(models, key=lambda name: validation_rank(models[name]["validation"]))
    # The checkpoint and epoch are selected before test pixels are loaded.
    test = load_split(folder, splits["test"])
    predictions = []
    for r in splits["test"]:
        predictions.append({"relative_path": r["relative_path"], "label": int(r["label"])})
    for name, model in checkpoints.items():
        scores, digits, matrix = evaluate(model, test)
        models[name]["test"] = {**scores, "confusion_matrix": matrix}
        for r, predicted in zip(predictions, digits):
            r[name] = predicted
    paired = {}
    for old, new in combinations(models, 2):
        pairs = Counter((r[old] == r["label"], r[new] == r["label"]) for r in predictions)
        paired[f"{old}_to_{new}"] = {"fixed": pairs[False, True], "introduced": pairs[True, False],
                                    "both_correct": pairs[True, True], "both_wrong": pairs[False, False]}
    summary = {
        "status": "complete" if pilot_data is not None else "control_only",
        "selected_model": chosen, "classes": classes, "seed": args.seed,
        "epochs": args.epochs, "batch_size": REAL_BATCH + EXTRAS, "optimizer": "Adam",
        "lr": LR, "weight_decay": 1e-4, "real_batch_slots": REAL_BATCH,
        "extra_batch_slots": EXTRAS, "torch_version": str(torch.__version__), "threads": args.threads,
        "selection": "Maximize real validation macro F1, breaking ties by lower validation cross-entropy",
        "real_source": "A1 balanced public MNIST subset", "real_sizes": {k:len(v) for k,v in splits.items()},
        "a1_checkpoint_sha256": sha_file(A1 / "results/cnn.pt"),
        "mnist_manifest_sha256": sha_file(folder / "manifest.csv"),
        "synthetic_accepted": 0 if pilot_data is None else len(pilot_data[1]),
        "pilot": provenance, "models": models, "paired_comparisons": paired,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out, prefix=".training-") as staging:
        stage = Path(staging)
        write_csv(stage / "training_history.csv", history)
        write_csv(stage / "paired_predictions.csv", predictions)
        plot_history(stage / "learning_curves.png", history, variants)
        torch.save({"state_dict": checkpoints[chosen].state_dict(), "classes": classes}, stage / "cnn_a2.pt")
        summary["checkpoint_sha256"] = sha_file(stage / "cnn_a2.pt")
        artifacts = ["training_history.csv", "paired_predictions.csv", "learning_curves.png", "cnn_a2.pt"]
        if pilot_data is not None:
            # Keep the requested fine-tuned model even if validation selects A1 for the demo.
            augmented = "cnn_a2_synthetic.pt"
            torch.save({"state_dict": checkpoints["synthetic"].state_dict(), "classes": classes}, stage / augmented)
            models["synthetic"]["checkpoint"] = augmented
            models["synthetic"]["checkpoint_sha256"] = sha_file(stage / augmented)
            artifacts.append(augmented)
        (stage / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        for name in [*artifacts, "metrics.json"]:
            os.replace(stage / name, args.out / name)
    print(f"Selected {chosen} on validation. Results: {args.out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
