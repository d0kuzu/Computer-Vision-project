"""Rebuild figures/gradcam.png and ../results/demo_latency.json using the deployed CNN."""
import base64
import csv
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REPORT = Path(__file__).resolve().parent
A2 = REPORT.parent
sys.path.insert(0, str(A2))
from demo import A1, Recognizer, preprocess

DATA = A1 / "data/mnist_subset"
RESULTS = A2 / "results"
FIGURE = REPORT / "figures/gradcam.png"


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def cpu_name():
    if sys.platform == "win32":
        import winreg
        key = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as processor:
            return winreg.QueryValueEx(processor, "ProcessorNameString")[0].strip()
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or platform.machine()


def font(size, bold=False):
    filenames = ("segoeuib.ttf", "DejaVuSans-Bold.ttf") if bold else ("segoeui.ttf", "DejaVuSans.ttf")
    for filename in filenames:
        try:
            return ImageFont.truetype(filename, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def render_figure(cases):
    page = Image.new("RGB", (1040, 354), "white")
    draw = ImageDraw.Draw(page)
    ink, muted = "#17372e", "#4e6259"
    draw.text((25, 13), "Grad-CAM on two held-out MNIST digits", font=font(31, True), fill=ink)
    draw.line((520, 68, 520, 307), fill="#c8d3c8", width=2)
    for i, (row, output) in enumerate(cases):
        x = 25 + 519 * i
        truth, prediction = row["label"], output["digit"]
        correct = truth == prediction
        title = f"{'A   Correct' if correct else 'B   Error'}   ·   True {truth}   /   Predicted {prediction}"
        draw.text((x, 63), title, font=font(23, True), fill=ink if correct else "#923e2c")
        draw.text((x, 98), f"MNIST test #{int(row['source_index']):05d}", font=font(17), fill=muted)
        for j, (field, caption) in enumerate((("input", "CNN input"), ("overlay", f"Grad-CAM for class {prediction}"))):
            left, top = x + 28 + 221 * j, 128
            data = base64.b64decode(output[field].partition(",")[2])
            with Image.open(BytesIO(data)) as original:
                assert original.size == (32, 32)
                image = original.convert("RGB").resize((160, 160), Image.Resampling.NEAREST)
            draw.rectangle((left - 1, top - 1, left + 160, top + 160), fill="#cbd6cc")
            page.paste(image, (left, top))
            draw.text((left + 80, 292), caption, font=font(17), fill=muted, anchor="mt")
    draw.text((25, 325), "Warm colors mark positive contributions to the predicted class. Each map is scaled to its own maximum.",
              font=font(16), fill=muted)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    page.save(FIGURE)


def main():
    checkpoint = RESULTS / "cnn_a2.pt"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    metrics = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    if digest != metrics["checkpoint_sha256"]:
        raise ValueError("Deployment checkpoint does not match metrics.json.")
    test = [r for r in read_rows(DATA / "manifest.csv") if r["split"] == "test"]
    per_digit = [[r for r in test if r["label"] == str(d)] for d in range(10)]
    if [len(rows) for rows in per_digit] != [30] * 10:
        raise ValueError("Expected the original 300-image held-out split.")
    # Interleave digits. Two unused examples per digit warm the kernels before measurement.
    warmup = [rows[i] for i in (28, 29) for rows in per_digit]
    measured = [rows[i] for i in range(10) for rows in per_digit]
    recognizer = Recognizer()

    def predict(row):
        with Image.open(DATA / row["relative_path"]) as im:
            if hashlib.sha256(im.convert("L").tobytes()).hexdigest() != row["sha256"]:
                raise ValueError(f"Changed benchmark input: {row['relative_path']}")
            tensor = preprocess(im)
        return recognizer.predict(tensor)

    for row in warmup:
        predict(row)
    samples = [{"relative_path": row["relative_path"], "elapsed_ms": predict(row)["elapsed_ms"]} for row in measured]
    timings = [r["elapsed_ms"] for r in samples]
    summary = {"n": len(samples), "mean_ms": float(np.mean(timings)),
               "median_ms": float(np.median(timings)), "p95_ms": float(np.percentile(timings, 95, method="linear"))}

    selected = metrics["selected_model"]
    paired = {r["relative_path"]: r for r in read_rows(RESULTS / "paired_predictions.csv")}
    cases = []
    for path in ("7/mnist_test_00086.png", "8/mnist_test_04140.png"):
        row = next(r for r in test if r["relative_path"] == path)
        output = predict(row)
        if output["digit"] != paired[path][selected]:
            raise ValueError(f"Demo and saved classifier disagree on {path}.")
        cases.append((row, output))
    if cases[0][0]["label"] != cases[0][1]["digit"] or cases[1][0]["label"] == cases[1][1]["digit"]:
        raise ValueError("These two examples no longer illustrate the saved correct/error pair.")
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != digest:
        raise ValueError("The deployment checkpoint changed during measurement.")
    render_figure(cases)
    payload = {
        "measured_utc": datetime.now(timezone.utc).isoformat(), "checkpoint": "assignment2/results/cnn_a2.pt",
        "checkpoint_sha256": digest, "selected_model": selected, "benchmark": "demo.Recognizer.predict (in process)",
        "scope": "perf_counter around encoder and classifier forward passes with gradients enabled for Grad-CAM",
        "excluded": ["model loading", "image I/O", "preprocessing", "softmax", "Grad-CAM backward pass and overlay",
                     "network, browser rendering and interaction"], "browser": None,
        "environment": {"device": "CPU", "processor": cpu_name(), "os": platform.platform(),
                        "logical_processors": os.cpu_count(), "torch_version": str(torch.__version__),
                        "torch_intraop_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
                        "python": platform.python_version(), "batch_size": 1},
        "input": "Fixed A1 test manifest order: ten examples per digit, interleaved 0–9.",
        "warmup_excluded": len(warmup), "warmup_paths": [r["relative_path"] for r in warmup],
        "summary": summary, "samples": samples,
        "figure_layer": "ReLU activation of the final convolution immediately before the last MaxPool",
        "figure_examples": [{"relative_path": r["relative_path"], "source_index": int(r["source_index"]),
                             "input_pixel_sha256": r["sha256"], "true_class": r["label"], "predicted_class": o["digit"],
                             "softmax_score": o["scores"][0]["score"], "positive_gradcam": o["active"]}
                            for r, o in cases]
    }
    (RESULTS / "demo_latency.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "environment": payload["environment"], "figure_examples": payload["figure_examples"]}, indent=2))


if __name__ == "__main__":
    main()
