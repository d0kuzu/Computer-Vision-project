"""Export the published 3,000-image index as a portable split/class dataset."""
import argparse
import csv
import hashlib
import io
import os
import tempfile
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parent
DATASET = PROJECT / "outputs" / "synthetic_1000"


def add_member(bundle, name, contents):
    # Fixed ZIP timestamps and permissions produce the same export on repeated runs.
    info = ZipInfo(name)
    info.compress_type = ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    bundle.writestr(info, contents)


def export(output):
    with (DATASET / "dataset_index.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = [*reader.fieldnames, "repository_path"]
        rows = list(reader)
    expected = {(split, str(digit)): count for split, count in
                [("train", 240), ("val", 30), ("test", 30)] for digit in range(10)}
    if Counter((row["split"], row["label"]) for row in rows) != expected:
        raise ValueError("Expected 240 training, 30 validation and 30 test images per digit")

    manifest = io.StringIO(newline="")
    writer = csv.DictWriter(manifest, fieldnames=fields)
    writer.writeheader()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as staging:
        temporary = Path(staging.name)
    try:
        paths, hashes = set(), set()
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as bundle:
            for row in rows:
                file = (ROOT / row["path"]).resolve()
                name = f"{row['split']}/{row['label']}/{file.name}"
                if not file.is_relative_to(ROOT) or file.suffix.lower() != ".png" or name in paths:
                    raise ValueError(f"Invalid or repeated image path: {file}")
                data = file.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                if digest != row["file_sha256"] or digest in hashes:
                    raise ValueError(f"Changed or duplicate PNG: {file}")
                paths.add(name)
                hashes.add(digest)
                add_member(bundle, name, data)
                writer.writerow({**row, "path": name, "repository_path": row["path"]})
            add_member(bundle, "manifest.csv", manifest.getvalue().encode("utf-8"))
            attribution = PROJECT / "outputs/reference_maskgen_cadcae3f15d2/SOURCE.txt"
            source_note = (
                "3,000 images of digits 0-9: 2,000 public MNIST originals and 1,000 synthetic "
                "derivatives of training originals. All derivatives belong to train. "
                "See manifest.csv for split, class, source and label provenance.\n\n"
                + attribution.read_text(encoding="utf-8")
            )
            add_member(bundle, "SOURCE.txt", source_note.encode("utf-8"))
        with ZipFile(temporary) as checked:
            if checked.testzip() is not None or len(checked.namelist()) != 3002:
                raise ValueError("Dataset archive did not pass verification")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved {len(rows)} images, manifest and attribution to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATASET / "digits_3000.zip")
    export(parser.parse_args().output)
