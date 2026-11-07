"""Audit 2,000 A1 originals and 1,000 reference-guided derivatives; publish CSV indexes and contact sheets.

Run from any directory with the project's Python environment. Encoded PNGs stay in their
original directories. Input runs must be complete, including the reserve if rows were excluded.
"""
import argparse
import csv
import hashlib
import json
import tempfile
from collections import Counter
from io import BytesIO
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from glyph_synthesis import METHOD, prepare_reference, style_reference


A2 = Path(__file__).resolve().parent
PROJECT = A2.parent
OUTPUTS = A2 / "outputs"
MNIST = PROJECT / "assignment1/data/mnist_subset"
sys.path.insert(0, str(PROJECT / "assignment1"))
from predict import preprocess

DIGITS = list("0123456789")
REFERENCE = ["source_relative_path", "source_sha256", "source_index", "source_split", "source_partition"]
IDENTITY = ["sample_id", "requested_digit", "path", "sha256"]
DECISION = ["label_source", "training_label", "human_reviewed"]
ORIGIN = ["origin_run", "origin_sample_id", "origin_path"]
MANIFEST = ["sample_id", "requested_digit", "variant", "control_variant", "style", "prompt",
            "seed", "path", "source", *REFERENCE, "sha256", *ORIGIN, *DECISION,
            "replaces", "replaces_sha256", "replacement_reason"]
INDEX = ["sample_id", "path", "split", "label", "source", "label_source", "origin_run",
         "source_relative_path", "source_pixel_sha256", "file_sha256", "pixel_sha256", "input32_sha256"]


def require(condition, explanation):
    if not condition:
        raise ValueError(explanation)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def safe_file(folder, relative):
    path = (folder / relative).resolve()
    require(path.is_relative_to(folder.resolve()) and path.is_file(), f"Missing or invalid path: {relative}")
    return path


def image_record(path, size, mode):
    data = path.read_bytes()
    with Image.open(BytesIO(data)) as im:
        require((im.format, im.size, im.mode) == ("PNG", size, mode), f"Unexpected image format: {path}")
        gray = im.convert("L")
        if mode == "RGB":
            require(im.tobytes() == gray.convert("RGB").tobytes(), f"Unexpected color content: {path}")
        # Exact network input, converted back from [0,1] to its 1-byte grayscale representation.
        input32 = (preprocess(im)[0, 0].numpy() * 255).round().astype("uint8")
    require(np.ptp(input32) > 0, f"Empty or flat classifier input: {path}")
    return gray, {"file_sha256": digest(data), "pixel_sha256": digest(gray.tobytes()),
                  "input32_sha256": digest(input32.tobytes())}


def read_real():
    rows = read_csv(MNIST / "manifest.csv")
    expected = {(split, d): n for d in DIGITS for split, n in [("train", 140), ("val", 30), ("test", 30)]}
    require(Counter((r["split"], r["label"]) for r in rows) == expected, "Changed A1 split or class balance")
    require(len({(r["source_partition"], r["source_index"]) for r in rows}) == 2000, "Duplicate A1 source ID")
    require(len({r["relative_path"] for r in rows}) == 2000, "Duplicate A1 path")
    references, index = {}, []
    for r in rows:
        path = safe_file(MNIST, r["relative_path"])
        gray, hashes = image_record(path, (28, 28), "L")
        require(hashes["pixel_sha256"] == r["sha256"] and r["label"] == r["class_name"],
                f"Changed A1 image or label: {path}")
        require(r["source_partition"] == ("test" if r["split"] == "test" else "train"),
                f"Unexpected A1 source partition: {path}")
        index.append(dict(**hashes, sample_id=path.stem, path=path.relative_to(PROJECT).as_posix(),
                          split=r["split"], label=r["label"], source="MNIST", label_source="published_mnist",
                          origin_run="", source_relative_path=r["relative_path"], source_pixel_sha256=r["sha256"]))
        if r["split"] == "train":
            references[r["relative_path"]] = (r, gray)
    return references, index


def by_id(rows, description):
    table = {r["sample_id"]: r for r in rows}
    require(len(table) == len(rows), f"Duplicate sample ID in {description}")
    return table


def font(size, bold=False):
    choices = ["segoeuib.ttf", "DejaVuSans-Bold.ttf"] if bold else ["segoeui.ttf", "DejaVuSans.ttf"]
    for filename in choices:
        try:
            return ImageFont.truetype(filename, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def write_gallery(folder, digit, rows, references):
    """Show all 88 new pairs in variant order at the original grayscale polarity."""
    rows = sorted(rows, key=lambda r: int(r["variant"]))
    require(len(rows) == 88, f"Expected 88 new thumbnails for {digit}")
    cw, ch = 128, 94
    sheet = Image.new("RGB", (8 * cw + 16, 84 + 11 * ch), "#f5f6f2")
    draw = ImageDraw.Draw(sheet)
    draw.text((14, 9), f"Цифра {digit}   ·   88 новых вариантов", font=font(29, True), fill="#17362f")
    draw.text((14, 51), "Слева — исходный рисунок MNIST train. Справа — синтетическая производная. Подпись — ID в manifest.csv.",
              font=font(16), fill="#38574b")
    label_font = font(13)
    for n, row in enumerate(rows):
        x, y = 8 + (n % 8) * cw, 81 + (n // 8) * ch
        draw.rectangle((x, y, x + cw - 4, y + ch - 4), fill="white")
        src = references[row["source_relative_path"]][1]
        sheet.paste(src.resize((40, 40), Image.Resampling.NEAREST), (x + 4, y + 16))
        with Image.open(OUTPUTS / row["path"]) as im:
            sheet.paste(im.resize((68, 68), Image.Resampling.LANCZOS), (x + 50, y + 3))
        draw.text((x + 6, y + 71), row["sample_id"], font=label_font, fill="#203e32")
    sheet.save(folder / f"digit-{digit}.png")


def audit_run(folder, cohort, references, exclusions=None):
    is_new = cohort != "base"
    manifest_path, review_path, run_path = (folder / filename for filename in ["manifest.csv", "review.csv", "run.json"])
    records = read_csv(manifest_path)
    reviews = by_id(read_csv(review_path), str(review_path))
    recipe = read_json(run_path)
    require(recipe.get("method") == METHOD, f"Wrong synthesis method in {run_path}")
    plan = by_id(recipe["samples"], str(run_path))
    require(by_id(records, str(manifest_path)).keys() == plan.keys() == reviews.keys(), f"Different IDs in {folder}")
    if cohort == "reserve":
        require(exclusions and Counter(r["replaces"] for r in records) == dict.fromkeys(exclusions, 1),
                "Reserve run must replace each documented exclusion exactly once")
        require(all(int(r["variant"]) >= 101 and r["replacement_reason"] == exclusions[r["replaces"]]
                    for r in records), "Reserve ID or exclusion reason mismatch")
    else:
        expected = {(d, v): 1 for d in DIGITS for v in (range(13, 101) if is_new else range(1, 13))}
        require(Counter((r["requested_digit"], int(r["variant"])) for r in records) == expected,
                f"Unexpected per-digit variant set for {folder}")
    combined, decisions, index, shape_checks = [], [], [], []
    for row in sorted(records, key=lambda r: (int(r["requested_digit"]), int(r["variant"]))):
        key, digit = row["sample_id"], row["requested_digit"]
        require(all(row[k] == str(plan[key][k]) for k in [*REFERENCE, "requested_digit", "variant", "seed", "path", "style", "prompt"]),
                f"Plan and manifest differ: {key}")
        review = reviews[key]
        require(all(row[k] == review[k] for k in IDENTITY) and row["source"] == "synthetic",
                f"Changed review identity: {key}")
        original = references.get(row["source_relative_path"])
        require(original is not None, f"Reference outside A1 train: {key}")
        original_row, original_image = original
        require(row["source_split"] == row["source_partition"] == "train" and
                original_row["label"] == digit and original_row["source_index"] == row["source_index"] and
                original_row["sha256"] == row["source_sha256"], f"Reference or source label mismatch: {key}")
        path = safe_file(folder, row["path"])
        require(path.parent == (folder / "candidates").resolve(), f"Candidate outside run candidates directory: {key}")
        gray, hashes = image_record(path, (256, 256), "RGB")
        require(hashes["file_sha256"] == row["sha256"], f"Changed candidate PNG: {key}")
        diag = read_json(safe_file(folder, f"diagnostics/{key}.json"))
        require(diag["sha256"] == row["sha256"] and diag["method"] == METHOD and
                diag["sample_id"] == key and int(row["seed"]) == diag["seed"] and diag["changed_code_count"] > 0,
                f"Missing generator evidence: {key}")
        control = int(row.get("control_variant") or row["variant"])
        if is_new:
            require(str(control) == str(plan[key]["control_variant"]), f"Style control differs from plan: {key}")
            fields = ("variant", "control_variant", "source_relative_path", "source_sha256", "source_index")
            require(all(str(diag[k]) == row[k] for k in fields), f"Diagnostic source identity differs: {key}")
        if cohort == "reserve":
            require(all(plan[key][k] == row[k] == diag[k] for k in ["replaces", "replaces_sha256", "replacement_reason"]),
                    f"Reserve selection differs from plan: {key}")
        styled, used_control = style_reference(prepare_reference(original_image), control)
        require(diag["control"] == used_control and used_control["style"] == row["style"], f"Wrong style: {key}")
        anchor, pixels = np.asarray(styled.convert("L"), dtype=np.int16), np.asarray(gray, dtype=np.int16)
        diff = np.abs(anchor - pixels)
        require(0 < diff.max() == diag["max_pixel_change"] <= diag["residual_clip_gray"] and
                np.count_nonzero(diff) == diag["changed_pixels"], f"Output does not match source-guided edit: {key}")
        ink, target = pixels < 128, anchor < 128
        union = (ink | target).sum()
        require(ink.any() and target.any(), f"Missing dark digit foreground: {key}")
        iou = float((ink & target).sum() / union)
        shape_checks.append((key, iou, int(diff.max()), int(diag["changed_code_count"])))
        if is_new:
            require(review["status"] == "pending" and not review["reviewed_digit"],
                    f"New run already has a review for {key}; reconcile it explicitly")
            status, training_label, reviewed_digit = "source_labelled", original_row["label"], ""
            label_source, human, note = "inherited_a1_train", "false", "Label inherited from verified A1 train source."
        else:
            require(review["status"] == "accept" and review["reviewed_digit"] in DIGITS and
                    review.get("train_consent") == "train_only", f"Base sample not approved for training: {key}")
            status, training_label, reviewed_digit = "accept", review["reviewed_digit"], review["reviewed_digit"]
            label_source, human, note = "user_batch_review", "true", review.get("notes", "")
        metadata = dict(label_source=label_source, training_label=training_label, human_reviewed=human)
        merged = {**row, **metadata, "control_variant": control, "path": path.relative_to(OUTPUTS).as_posix(),
                  "origin_run": folder.name, "origin_sample_id": key, "origin_path": row["path"],
                  "replaces": row.get("replaces", ""), "replaces_sha256": row.get("replaces_sha256", ""),
                  "replacement_reason": row.get("replacement_reason", "")}
        combined.append(merged)
        decisions.append({**{k: merged[k] for k in IDENTITY}, **metadata, "status": status, "notes": note,
                          "reviewed_digit": reviewed_digit, "train_consent": "train_only"})
        index.append(dict(**hashes, sample_id=key, path=path.relative_to(PROJECT).as_posix(), split="train",
                          label=training_label, source="synthetic", label_source=label_source, origin_run=folder.name,
                          source_relative_path=row["source_relative_path"], source_pixel_sha256=row["source_sha256"]))
    evidence = {"run": folder.name, "samples": len(records), "method": recipe["method"],
                "run_sha256": digest(run_path.read_bytes()), "manifest_sha256": digest(manifest_path.read_bytes()),
                "review_sha256": digest(review_path.read_bytes())}
    return combined, decisions, index, shape_checks, evidence


def assemble(base, extension, destination, reserve, exclusion_csv):
    base, extension, reserve, destination = (p.resolve() for p in [base, extension, reserve, destination])
    paths = [base, extension, reserve, destination]
    require(all(p.parent == OUTPUTS.resolve() for p in paths) and len(set(paths)) == 4,
            "Use distinct direct children of assignment2/outputs")
    quality_rows = read_csv(exclusion_csv)
    exclusions = {r["replaces"]: r["reason"] for r in quality_rows}
    require(len(exclusions) == len(quality_rows) and all(exclusions.values()), "Invalid or duplicate quality exclusions")
    references, index = read_real()
    records, decisions, checks, origins = [], [], [], []
    cohorts = [(base, "base"), (extension, "extension")]
    if exclusions:
        cohorts.append((reserve, "reserve"))
    for folder, cohort in cohorts:
        items, approval, entries, shape, provenance = audit_run(folder, cohort, references, exclusions)
        records.extend(items)
        decisions.extend(approval)
        index.extend(entries)
        checks.extend(shape)
        origins.append(provenance)
    lookup = by_id(records, "raw candidate pool")
    require(len({r["source_relative_path"] for r in records}) == len(records) == 1000 + len(exclusions),
            "A source is reused in the candidate pool")
    superseded = []
    for r in (r for r in records if r["replaces"]):
        old = lookup.get(r["replaces"])
        require(old is not None and old["origin_run"] == extension.name and old["sha256"] == r["replaces_sha256"] and
                (old["requested_digit"], old["control_variant"]) == (r["requested_digit"], r["control_variant"]),
                f"Replacement outside the same new class/style: {r['sample_id']}")
        superseded.append({"excluded": old["sample_id"], "excluded_png": old["path"],
                           "excluded_sha256": old["sha256"], "selected": r["sample_id"],
                           "selected_png": r["path"], "selected_sha256": r["sha256"], "reason": r["replacement_reason"]})
    records = [r for r in records if r["sample_id"] not in exclusions]
    decisions = [r for r in decisions if r["sample_id"] not in exclusions]
    index = [r for r in index if r["sample_id"] not in exclusions]
    checks = [v for v in checks if v[0] not in exclusions]
    require(len({r["source_relative_path"] for r in records}) == 1000, "Synthetic sources overlap within or between runs")
    expected = {(split, digit): count for split, count in [("train", 240), ("val", 30), ("test", 30)] for digit in DIGITS}
    require(Counter((r["split"], r["label"]) for r in index) == expected, "Unexpected combined split counts")
    for key in ("sample_id", "path", "file_sha256", "pixel_sha256", "input32_sha256"):
        require(len({r[key] for r in index}) == len(index) == 3000, f"Duplicate or missing {key} across the 3,000 images")
    records.sort(key=lambda r: (int(r["requested_digit"]), int(r["variant"])))
    decision_map = by_id(decisions, "combined labels")
    decisions = [decision_map[r["sample_id"]] for r in records]
    with tempfile.TemporaryDirectory(prefix=".assemble-", dir=OUTPUTS) as staging:
        stage = Path(staging) / "bundle"
        galleries = stage / "galleries"
        galleries.mkdir(parents=True)
        write_csv(stage / "manifest.csv", MANIFEST, records)
        write_csv(stage / "review.csv", [*IDENTITY, "status", "reviewed_digit", "notes", "train_consent", *DECISION], decisions)
        write_csv(stage / "dataset_index.csv", INDEX, index)
        attribution = read_json(MNIST / "PROVENANCE.json")
        run = dict(schema="a2_combined_synthetic_v1", method=METHOD, path_root="assignment2/outputs",
                   source_runs=[p.name for p, _ in cohorts], source_run_metadata=origins,
                   a1_manifest_sha256=digest((MNIST / "manifest.csv").read_bytes()),
                   source_attribution={k: attribution[k] for k in ("source", "attribution", "license", "license_url")},
                   quality_exclusions=superseded, quality_csv_sha256=digest(exclusion_csv.read_bytes()),
                   samples=[{k: r[k] for k in MANIFEST} for r in records])
        write_json(stage / "run.json", run)
        scores = np.array([v[1] for v in checks])
        qa = dict(originals=2000, synthetic=1000, previous_synthetic=120, additional_synthetic=880,
                  raw_synthetic_candidates=1000 + len(exclusions), reserve_generated=len(exclusions),
                  quality_exclusions=superseded,
                  counts_by_label_source=dict(Counter(r["label_source"] for r in index)),
                  total=len(index), train=2400, validation=300, test=300,
                  train_synthetic_fraction=1000/2400, all_synthetic_fraction=1000/3000,
                  by_digit={d: dict(originals=200, prior_synthetic=12, new_synthetic=88, train=240, val=30, test=30)
                            for d in DIGITS},
                  unique_encoded_pngs=3000, unique_decoded_grayscale_images=3000, unique_input32=3000,
                  unique_a1_train_references=1000, references_from_validation_or_test=0,
                  cross_split_exact_input32_duplicates=0,
                  visual_screening="All raw extension pairs and reserve outputs inspected side by side with their A1 train source. Severe ambiguity recorded in quality_reserve.csv. Ordinary handwriting variation retained. No classifier predictions used for selection.",
                  overlap_scope="Source IDs, encoded PNGs, decoded grayscale pixels and normalized inputs were checked for exact overlap. Repeated handwriting styles and near-duplicates can still occur.",
                  label_policy={"user_batch_review": "Earlier 120 accepted as a group for train only.",
                                "inherited_a1_train": "New 880 use their A1 train source labels and verified generated pixels. No individual human annotation."},
                  file_paths="dataset_index.csv relative to repository root; synthetic manifest.csv relative to assignment2/outputs.",
                  input32_hash="SHA-256 of grayscale uint8 A1-preprocessed 32x32 pixel bytes before division by 255.",
                  shape_check="Black-mask IoU at gray < 128 against each digit's styled source before the model residual.",
                  shape_iou_min=float(scores.min()), shape_iou_median=float(np.median(scores)),
                  ten_lowest_shape_iou=[{"sample_id": v[0], "iou": v[1]} for v in sorted(checks, key=lambda v:v[1])[:10]],
                  max_observed_gray_change=max(v[2] for v in checks), min_changed_tokens=min(v[3] for v in checks),
                  source_runs=origins)
        extras = [r for r in records if r["label_source"] == "inherited_a1_train"]
        # Labels use available system fonts. Keep the original verified thumbnails on
        # another machine even if its local choice of font would render differently.
        previous_hashes = read_json(destination / "qa.json").get("gallery_png_sha256", {}) if destination.exists() else {}
        qa["gallery_png_sha256"] = {}
        for digit in DIGITS:
            filename = f"digit-{digit}.png"
            if destination.exists():
                checksum = digest(safe_file(destination / "galleries", filename).read_bytes())
                require(checksum == previous_hashes.get(filename), f"Existing contact sheet changed: {filename}")
            else:
                write_gallery(galleries, digit, [r for r in extras if r["requested_digit"] == digit], references)
                checksum = digest((galleries / filename).read_bytes())
            qa["gallery_png_sha256"][filename] = checksum
        write_json(stage / "qa.json", qa)
        links = "".join(f'<li><a href="digit-{d}.png">Цифра {d} · 88 изображений</a></li>\n' for d in DIGITS)
        (galleries / "index.html").write_text(
            '<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>880 новых изображений</title><style>body{font:18px/1.5 system-ui;max-width:680px;margin:2rem auto;padding:0 1rem}'
            'li{margin:.45rem 0}a{color:#175838}</style><h1>880 новых изображений</h1>'
            '<p>Каждый лист показывает исходник MNIST из обучающей выборки A1 слева и сгенерированный вариант справа. '
            'Метки новых изображений унаследованы от исходников.</p><ul>' + links + '</ul></html>', encoding="utf-8", newline="\n")
        if destination.exists():
            for item in stage.rglob("*"):
                if item.is_file():
                    existing = destination / item.relative_to(stage)
                    require(existing.is_file(), f"Existing bundle missing {existing}")
                    same = (item.read_text(encoding="utf-8") == existing.read_text(encoding="utf-8")
                            if item.suffix == ".html" else item.read_bytes() == existing.read_bytes())
                    require(same, f"Existing bundle differs at {existing}. Choose another output.")
        else:
            stage.rename(destination)
    print(f"Verified {len(index)} distinct originals and derivatives; 2,400 train / 300 val / 300 test.")
    print(f"Dataset: {destination / 'dataset_index.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=OUTPUTS / "reference_maskgen_cadcae3f15d2")
    parser.add_argument("--extension", type=Path, default=OUTPUTS / "reference_maskgen_extension_880")
    parser.add_argument("--reserve", type=Path, default=OUTPUTS / "reference_maskgen_reserve")
    parser.add_argument("--exclusions", type=Path, default=A2 / "quality_reserve.csv")
    parser.add_argument("--output", type=Path, default=OUTPUTS / "synthetic_1000")
    args = parser.parse_args()
    assemble(args.base, args.extension, args.output, args.reserve, args.exclusions)
