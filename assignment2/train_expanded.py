"""Train and evaluate the 3,000-image experiment on CPU.

From the repository root: python assignment2/train_expanded.py
The historical 120-image experiment remains in assignment2/results/.
"""
import argparse
import copy
import hashlib
import json
import os
import shutil
from collections import Counter
from io import BytesIO
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from PIL import Image
from torch.nn import functional as F

from train_compare import (
    A1, DIGITS, LR, REFERENCE_FIELDS, ROOT, check_pilot_sources, evaluate, history_row, load_model,
    load_split, plot_history, preprocess, read_csv, read_mnist, sha_file,
    validation_rank, write_csv,
)

REAL_BATCH = 28
EXTRA_BATCH = 20
BASE_RUN = "reference_maskgen_cadcae3f15d2"
EXTENSION_RUN = "reference_maskgen_extension_880"


def load_expansion(folder, original_rows):
    outputs = folder.resolve().parent
    manifest = read_csv(folder / "manifest.csv")
    reviews = read_csv(folder / "review.csv")
    run = json.loads((folder / "run.json").read_text(encoding="utf-8-sig"))
    origins = run.get("source_runs", [])
    if (run.get("schema") != "a2_combined_synthetic_v1" or run.get("path_root") != "assignment2/outputs"
            or not {BASE_RUN, EXTENSION_RUN}.issubset(origins) or len(set(origins)) != len(origins)
            or any(Path(name).name != name or name in {"", ".", ".."} for name in origins)):
        raise ValueError("Expected the combined MaskGen index and its declared source runs")
    # Resolve each included row against its own immutable source manifest. This also
    # covers QA replacements, whose origin identifier can differ from the combined ID.
    originals = {}
    for origin in origins:
        rows = read_csv(outputs / origin / "manifest.csv")
        originals[origin] = {r["sample_id"]: r for r in rows}
        if len(originals[origin]) != len(rows):
            raise ValueError(f"Duplicate source-run sample ID: {origin}")
    prior_review = {r["sample_id"]: r for r in read_csv(outputs / BASE_RUN / "review.csv")}
    provenance = check_pilot_sources(run, manifest, original_rows)
    planned = {r["sample_id"]: r for r in run["samples"]}
    accepted = {r["sample_id"]: r for r in reviews}
    ids = {m["sample_id"] for m in manifest}
    if len(manifest) != 1000 or len(reviews) != 1000 or len(ids) != 1000 or ids != accepted.keys():
        raise ValueError("Expected 1,000 distinct candidates with an exact label/index join")
    replacements = run.get("quality_exclusions", [])
    excluded = {r["excluded"] for r in replacements}
    reserved = {r["selected"]: r["excluded"] for r in replacements}
    included_reserve = {m["sample_id"]: m["replaces"] for m in manifest if m.get("replaces")}
    if excluded & ids or reserved != included_reserve or len(excluded) != len(replacements):
        raise ValueError("Selected membership differs from the published visual-QA exclusions")
    original_indices = {r["relative_path"]: i for i, r in enumerate(original_rows)}
    images, labels, sources, candidate_ids = [], [], [], []
    seen_hashes, label_counts = set(), Counter()
    for m in sorted(manifest, key=lambda row: (int(row["requested_digit"]), int(row["variant"]))):
        key, origin, origin_id = m["sample_id"], m["origin_run"], m["origin_sample_id"]
        review = accepted[key]
        plan_fields = ("sha256", "origin_run", "origin_sample_id", "origin_path", "label_source", "training_label", "human_reviewed")
        raw = originals.get(origin, {}).get(origin_id)
        if (m["source"] != "synthetic" or raw is None or m["origin_path"] != raw["path"]
                or any(raw[k] != m[k] for k in ("sha256", "requested_digit", *REFERENCE_FIELDS))
                or m["path"] != f"{origin}/{m['origin_path']}"
                or any(m[k] != str(planned[key][k]) for k in plan_fields)
                or any(m[k] != review[k] for k in ("requested_digit", "path", "sha256", "label_source"))
                or review["train_consent"] != "train_only"):
            raise ValueError(f"Invalid candidate identity, source or training permission: {key}")
        source_index = original_indices[m["source_relative_path"]]
        label = review["training_label"]
        if label != original_rows[source_index]["label"]:
            raise ValueError(f"Training label differs from the matched A1 source: {key}")
        if origin == BASE_RUN:
            expected = ("user_batch_review", "accept", label, "true")
            decision = prior_review.get(origin_id, {})
            if (decision.get("status") != "accept" or decision.get("reviewed_digit") != label
                    or decision.get("train_consent") != "train_only" or decision.get("sha256") != m["sha256"]):
                raise ValueError(f"Original batch approval differs from the combined index: {key}")
        else:
            expected = ("inherited_a1_train", "source_labelled", "", "false")
        actual = (review["label_source"], review["status"], review["reviewed_digit"], review["human_reviewed"].lower())
        if actual != expected or any(m[k] != review[k] for k in ("training_label", "human_reviewed") if k in m):
            raise ValueError(f"Label provenance does not match {origin}: {key}")
        path = (outputs / m["path"]).resolve()
        if path.parent != outputs / origin / "candidates" or path.suffix.lower() != ".png":
            raise ValueError(f"Candidate outside its origin: {key}")
        encoded = path.read_bytes()
        if m["sha256"] in seen_hashes or hashlib.sha256(encoded).hexdigest() != m["sha256"]:
            raise ValueError(f"Changed or duplicate candidate PNG: {key}")
        with Image.open(BytesIO(encoded)) as im:
            if im.format != "PNG" or im.size != (256, 256):
                raise ValueError(f"Unexpected MaskGen image: {key}")
            images.append(preprocess(im))
        seen_hashes.add(m["sha256"])
        labels.append(int(label))
        sources.append(source_index)
        candidate_ids.append(key)
        label_counts[(review["label_source"], label)] += 1
    expected_counts = {(source, label): n for source, n in [("user_batch_review", 12), ("inherited_a1_train", 88)] for label in DIGITS}
    if label_counts != expected_counts or len(set(sources)) != 1000:
        raise ValueError("Expected 100 distinct A1 train references per digit (12 reviewed, 88 label-inherited)")
    provenance.update({
        "run": folder.name, "run_sha256": sha_file(folder / "run.json"),
        "manifest_sha256": sha_file(folder / "manifest.csv"), "review_sha256": sha_file(folder / "review.csv"),
        "dataset_index_sha256": sha_file(folder / "dataset_index.csv"), "quality_replacements": len(replacements),
        "label_sources": {"user_batch_review": 120, "inherited_a1_train": 880},
        "retained_per_source_run": dict(Counter(row["origin_run"] for row in manifest)),
        "training_labels_per_digit": dict(Counter(str(label) for label in labels)),
    })
    return (torch.cat(images), torch.tensor(labels)), torch.tensor(sources), candidate_ids, provenance


def epoch_plan(real_rows, synthetic_ids, reference_indices, epochs, seed):
    rng = torch.Generator().manual_seed(seed)
    plan, coverage = [], []
    for epoch in range(1, epochs + 1):
        real = torch.randperm(len(real_rows), generator=rng).reshape(-1, REAL_BATCH)
        extras = torch.randperm(len(synthetic_ids), generator=rng).reshape(-1, EXTRA_BATCH)
        sources = reference_indices[extras]
        if (len(real) != len(extras) or real.unique().numel() != 1400 or extras.unique().numel() != 1000
                or sources.unique().numel() != 1000):
            raise ValueError("Epoch does not cover the full indexed dataset")
        plan.append((real, extras))
        # Hash all ordered identifiers, so a recreated batch plan can be verified independently.
        batches = [([real_rows[i]["relative_path"] for i in r], [synthetic_ids[i] for i in e],
                    [real_rows[i]["relative_path"] for i in s]) for r, e, s in zip(real.tolist(), extras.tolist(), sources.tolist())]
        digest = hashlib.sha256(json.dumps(batches, separators=(",", ":")).encode()).hexdigest()
        coverage.append({"epoch": epoch, "steps": len(real), "distinct_real_base": real.unique().numel(),
                         "distinct_synthetic": extras.unique().numel(), "distinct_control_replay": sources.unique().numel(),
                         "paired_batch_plan_sha256": digest})
    return plan, coverage


def fit(name, base, train, validation, extra_data, plan, coverage, seed):
    model = copy.deepcopy(base)
    torch.manual_seed(seed)  # identical dropout masks and Adam configuration across arms
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    best, best_state, history = None, None, []
    x, y = train
    extra_images, extra_labels = extra_data
    for epoch, (order, extras) in enumerate(plan, 1):
        model.train()
        seen_real, seen_extra = set(), set()
        updates, loss_sum = 0, 0.0
        for real, extra in zip(order, extras):
            inputs = torch.cat([x[real], extra_images[extra]])
            targets = torch.cat([y[real], extra_labels[extra]])
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(inputs), targets)
            loss.backward()
            optimizer.step()
            updates += 1
            loss_sum += loss.item() * len(targets)
            seen_real.update(real.tolist())
            seen_extra.update(extra.tolist())
        if (len(seen_real), len(seen_extra), updates) != (len(y), len(extra_images), len(order)):
            raise RuntimeError(f"Incomplete training coverage in {name} epoch {epoch}")
        coverage[epoch - 1].update({f"{name}_base_used": len(seen_real), f"{name}_extras_used": len(seen_extra),
                                    f"{name}_completed_updates": updates})
        row, scores = history_row(name, epoch, model, train, validation)
        row["optimization_loss"] = loss_sum / (len(y) + len(extra_images))
        history.append(row)
        print(f"{name} epoch {epoch}: real val F1={scores['macro_f1']:.5f}, loss={scores['loss']:.5f}", flush=True)
        if best is None or validation_rank(scores) > validation_rank(best["validation"]):
            best = {"epoch": epoch, "validation": scores}
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model.eval(), best, history


def restored(base, checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["classes"] != DIGITS:
        raise ValueError(f"Unexpected checkpoint classes: {checkpoint}")
    model = copy.deepcopy(base)
    model.load_state_dict(saved["state_dict"])
    return model.eval()


def paired_changes(rows, older, newer):
    counts = Counter((r[older] == r["label"], r[newer] == r["label"]) for r in rows)
    return {"fixed": counts[False, True], "introduced": counts[True, False],
            "both_correct": counts[True, True], "both_wrong": counts[False, False]}


def train(dataset, destination, epochs, seed, threads):
    if destination.resolve() in {(ROOT / "results").resolve(), (A1 / "results").resolve()}:
        raise ValueError("Choose another output directory to preserve the submitted experiments")
    base, classes = load_model()
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    if classes != DIGITS:
        raise ValueError("Unexpected A1 classifier classes")
    folder, splits = read_mnist()
    train_data, val = (load_split(folder, splits[key]) for key in ("train", "val"))
    synthetic, references, ids, provenance = load_expansion(dataset, splits["train"])
    if not torch.equal(synthetic[1], train_data[1][references]):
        raise ValueError("Real replay targets differ from synthetic targets")
    plan, coverage = epoch_plan(splits["train"], ids, references, epochs, seed)
    a1_path = A1 / "results/cnn.pt"
    prior_path = ROOT / "results/cnn_a2.pt"
    prior_metrics = json.loads((ROOT / "results/metrics.json").read_text(encoding="utf-8"))
    if (sha_file(a1_path) != prior_metrics["a1_checkpoint_sha256"]
            or sha_file(folder / "manifest.csv") != prior_metrics["mnist_manifest_sha256"]):
        raise ValueError("A1 initial weights or original split differs from the submitted experiment")
    if prior_metrics["selected_model"] != "synthetic" or sha_file(prior_path) != prior_metrics["checkpoint_sha256"]:
        raise ValueError("Archived A2-120 deployment differs from its report")
    first, a1_val = history_row("a1", 0, base, train_data, val)
    first["optimization_loss"] = None
    history = [first]
    prior = restored(base, prior_path)
    models = {"a1": {"epoch": 0, "validation": a1_val, "training_originals_by_source": {"real": 1400, "synthetic": 0}},
              "prior_120": {"epoch": prior_metrics["models"]["synthetic"]["epoch"], "validation": evaluate(prior, val)[0],
                            "training_originals_by_source": {"real": 1400, "synthetic": 120}}}
    candidates = {"a1": base, "prior_120": prior}
    matched_replay = train_data[0][references], train_data[1][references]
    for name, extra in [("control", matched_replay), ("synthetic", synthetic)]:
        model, result, rows = fit(name, base, train_data, val, extra, plan, coverage, seed)
        result["training_originals_by_source"] = {"real": 1400, "synthetic": 1000 if name == "synthetic" else 0}
        result["source_draws_per_epoch"] = {"real_original": 1400, "real_replay": 1000 if name == "control" else 0,
                                            "synthetic": 1000 if name == "synthetic" else 0}
        models[name], candidates[name] = result, model
        history.extend(rows)
    selected = max(models, key=lambda name: validation_rank(models[name]["validation"]))
    # Lock both branches' epochs and the deployment before loading test pixels.
    test = load_split(folder, splits["test"])
    paired = [{key: row[key] for key in ("relative_path", "source_partition", "source_index", "sha256")}
              | {"label": int(row["label"])} for row in splits["test"]]
    destination.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination, prefix=".training-") as temporary:
        stage = Path(temporary)
        paths = {"a1": a1_path, "prior_120": prior_path, "control": stage / "cnn_control.pt",
                 "synthetic": stage / "cnn_a2_synthetic.pt"}
        for key in ("control", "synthetic"):
            torch.save({"state_dict": candidates[key].state_dict(), "classes": DIGITS}, paths[key])
        for name, path in paths.items():
            scores, digits, matrix = evaluate(restored(base, path), test)
            if digits != evaluate(candidates[name], test)[1]:
                raise RuntimeError(f"Reloaded {name} predictions differ from training result")
            models[name]["test"] = {**scores, "correct": sum(p == r["label"] for r, p in zip(paired, digits)),
                                    "confusion_matrix": matrix}
            if path.parent == stage:
                models[name]["checkpoint"] = path.name
                models[name]["checkpoint_relative_to"] = "results directory"
            else:
                models[name]["checkpoint"] = path.relative_to(ROOT.parent).as_posix()
                models[name]["checkpoint_relative_to"] = "repository root"
            models[name]["checkpoint_sha256"] = sha_file(path)
            for row, digit in zip(paired, digits):
                row[name] = digit
        archived = {r["relative_path"]: r for r in read_csv(ROOT / "results/paired_predictions.csv")}
        if (set(archived) != {r["relative_path"] for r in paired}
                or any(tuple(int(archived[r["relative_path"]][k]) for k in ("label", "a1", "synthetic"))
                       != (r["label"], r["a1"], r["prior_120"]) for r in paired)):
            raise RuntimeError("The A1/A2-120 reference predictions differ from the archived test table")
        deploy = stage / "cnn_a2.pt"
        shutil.copyfile(paths[selected], deploy)
        if evaluate(restored(base, deploy), test)[1] != [r[selected] for r in paired]:
            raise RuntimeError("Reloaded deployment differs from the validation-selected model")
        summary = {
            "status": "complete", "selected_model": selected, "classes": DIGITS, "seed": seed, "epochs": epochs,
            "selection": "Maximum real validation macro-F1, then lowest real validation cross-entropy. "
                         "Within an arm exact ties use the earliest epoch. Across models exact ties use A1, prior_120, control, synthetic order.",
            "optimizer": "Adam", "lr": LR, "weight_decay": 1e-4, "torch_version": str(torch.__version__), "threads": threads,
            "batch_size": REAL_BATCH + EXTRA_BATCH, "real_batch_slots": REAL_BATCH, "extra_batch_slots": EXTRA_BATCH,
            "optimizer_steps_per_epoch": len(plan[0][0]), "optimizer_steps_per_arm": epochs * len(plan[0][0]),
            "control": "The 1,000 extra slots replay the exact A1 train reference for each synthetic variant in the same order.",
            "real_sizes": {k: len(v) for k, v in splits.items()}, "synthetic_in_training": len(ids),
            "expanded_train_size": len(train_data[1]) + len(ids), "dataset_size": sum(map(len, splits.values())) + len(ids),
            "a1_checkpoint_sha256": models["a1"]["checkpoint_sha256"], "mnist_manifest_sha256": sha_file(folder / "manifest.csv"),
            "dataset": provenance, "models": models,
            "paired_comparisons": {f"{a}_to_{b}": paired_changes(paired, a, b) for a, b in combinations(models, 2)},
            "checkpoint_sha256": sha_file(deploy), "reloaded_predictions_match": True, "archived_test_predictions_match": True,
        }
        write_csv(stage / "training_history.csv", history)
        write_csv(stage / "epoch_coverage.csv", coverage)
        write_csv(stage / "paired_predictions.csv", paired)
        plot_history(stage / "learning_curves.png", history, ("control", "synthetic"), "1,000 synthetic variants")
        (stage / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        for artifact in sorted(stage.iterdir(), key=lambda p: p.name == "metrics.json"):
            os.replace(artifact, destination / artifact.name)
    print(f"Selected {selected} on validation. Results: {destination.resolve()}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=ROOT / "outputs/synthetic_1000")
    parser.add_argument("--out", type=Path, default=ROOT / "results_3000")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1:
        parser.error("--epochs and --threads must be positive")
    train(args.dataset, args.out, args.epochs, args.seed, args.threads)
