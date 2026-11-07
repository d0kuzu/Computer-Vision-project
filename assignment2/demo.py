"""Local A2 digit lab: python assignment2/demo.py, then open 127.0.0.1:8766."""
import argparse
import base64
import csv
import hashlib
import json
import os
import platform
import sys
import tempfile
import threading
import time
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO, StringIO
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
A1 = HERE.parent / "assignment1"
sys.path.insert(0, str(A1))
from predict import load_model, preprocess

DIGITS = set("0123456789")
MAX_UPLOAD = 6 * 1024 * 1024
IDENTITY = ("sample_id", "requested_digit", "path", "sha256")
REFERENCE_METHOD = "reference_masked_tokens_v1"
SOURCE_FIELDS = ("source_relative_path", "source_sha256", "source_index", "source_partition", "source_split")
MNIST = A1 / "data" / "mnist_subset"


class Conflict(ValueError):
    pass


def image_url(image):
    output = BytesIO()
    image.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def user_input(payload):
    source = payload.get("source")
    if source not in {"draw", "upload"}:
        raise ValueError("Укажите источник изображения.")
    encoded = payload.get("image", "")
    if not isinstance(encoded, str) or ";base64," not in encoded:
        raise ValueError("Ожидается изображение PNG, JPEG или WebP.")
    data = base64.b64decode(encoded.split(";base64,", 1)[1], validate=True)
    if not 0 < len(data) <= MAX_UPLOAD:
        raise ValueError("Файл должен быть меньше 6 МБ.")
    with Image.open(BytesIO(data)) as raw:
        if raw.format not in {"JPEG", "PNG", "WEBP"} or max(raw.size) > 4096:
            raise ValueError("Поддерживаются PNG, JPEG и WebP размером до 4096 пикселей по стороне.")
        if source == "draw" and raw.size != (280, 280):
            raise ValueError("Некорректный размер холста.")
        rgba = ImageOps.exif_transpose(raw).convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        background.alpha_composite(rgba)
        image = background.convert("L")
    pixels = np.asarray(image)
    border = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
    if np.median(border) < 128:
        image = ImageOps.invert(image)
    low, high = image.getextrema()
    if high - low < 12:
        raise ValueError("Изображение пустое или штрих слишком бледный.")
    if source == "draw":
        bbox = ImageOps.invert(image).point(lambda p: 255 if p > 35 else 0).getbbox()
        if not bbox:
            raise ValueError("Нарисуйте цифру на холсте.")
        crop = image.crop(bbox)
        side = max(crop.size)
        margin = max(3, int(side * 0.18))
        image = Image.new("L", (side + 2 * margin,) * 2, 255)
        image.paste(crop, ((image.width - crop.width) // 2, (image.height - crop.height) // 2))
    return preprocess(image)


def metrics():
    path = HERE / "results" / "metrics.json"
    report = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    rows = []
    for name, model in report.get("models", {}).items():
        test = model.get("test") or {}
        if test.get("accuracy") is not None and test.get("macro_f1") is not None:
            rows.append({"name": name, "accuracy": test["accuracy"], "f1": test["macro_f1"]})
    if not rows:
        original = A1 / "results" / "model_comparison.csv"
        if original.is_file():
            with original.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    if row["model"] == "CNN":
                        rows.append({"name": "a1", "accuracy": float(row["accuracy"]),
                                     "f1": float(row["f1_macro"])})
    return {"rows": rows, "selected_model": report.get("selected_model"),
            "synthetic_accepted": report.get("synthetic_accepted"), "status": report.get("status")}


class Recognizer:
    def __init__(self):
        self.lock = threading.Lock()
        self.signature = None
        self.device = f"CPU · {platform.machine()}"
        self.refresh()

    def refresh(self):
        a2 = HERE / "results" / "cnn_a2.pt"
        path = a2 if a2.is_file() else A1 / "results" / "cnn.pt"
        stat = path.stat()
        signature = (path, stat.st_mtime_ns, stat.st_size)
        if signature != self.signature:
            model, classes = load_model()
            if path == a2:
                checkpoint = torch.load(a2, map_location="cpu", weights_only=True)
                model.load_state_dict(checkpoint["state_dict"])
                classes = checkpoint["classes"]
            model.eval()
            self.classes = [str(value) for value in classes]
            last_conv = max(i for i, layer in enumerate(model) if isinstance(layer, nn.Conv2d))
            # A1 uses Conv → ReLU → MaxPool. Explain the rectified maps entering the last pool.
            cut = last_conv + 2 if isinstance(model[last_conv + 1], nn.ReLU) else last_conv + 1
            self.encoder = model[:cut]
            self.classifier = model[cut:]
            self.name = "CNN A1 · MNIST"
            if path == a2:
                selected = metrics()["selected_model"]
                self.name = {"a1": "CNN A1 · выбрана в A2", "control": "CNN A2 · без синтетики",
                             "synthetic": "CNN A2 · с синтетикой"}.get(selected, "CNN · Assignment 2")
            self.total_ms = 0.0
            self.count = 0
            self.signature = signature

    def info(self):
        return {"model": self.name, "device": self.device, "count": self.count,
                "mean_ms": self.total_ms / self.count if self.count else None}

    def predict(self, tensor):
        with self.lock, torch.enable_grad():
            self.refresh()
            # Keep the final convolution's spatial activations for the class gradient.
            started = time.perf_counter()
            features = self.encoder(tensor)
            logits = self.classifier(features)
            elapsed = (time.perf_counter() - started) * 1000
            probabilities = torch.softmax(logits.detach()[0], dim=0)
            order = probabilities.argsort(descending=True).tolist()
            gradient = torch.autograd.grad(logits[0, order[0]], features)[0]
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * features.detach()).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=(32, 32), mode="bilinear", align_corners=False)[0, 0]
            peak = cam.max().item()
            heat = cam.numpy() / peak if peak > 0 else cam.numpy()
            gray = np.uint8(np.clip(tensor[0, 0].numpy() * 255, 0, 255))
            ink = (255 - gray).astype(float)[:, :, None]
            color = np.stack([255 * np.ones_like(heat), 190 * (1 - heat), 22 * (1 - heat)], axis=-1)
            alpha = (0.76 * heat)[:, :, None]
            overlay = Image.fromarray(np.uint8(ink * (1 - alpha) + color * alpha))
            self.total_ms += elapsed
            self.count += 1
            return {**self.info(), "digit": self.classes[order[0]], "elapsed_ms": elapsed,
                    "scores": [{"digit": self.classes[i], "score": probabilities[i].item()} for i in order[:3]],
                    "input": image_url(Image.fromarray(gray)), "overlay": image_url(overlay), "active": peak > 0}


def parse_csv(data):
    reader = csv.DictReader(StringIO(data.decode("utf-8-sig"), newline=""))
    return reader.fieldnames or [], list(reader)


def revision(run, row):
    return hashlib.sha256(json.dumps([run, row], sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@lru_cache(maxsize=1)
def a1_training_rows():
    _, rows = parse_csv((MNIST / "manifest.csv").read_bytes())
    return {r["relative_path"]: r for r in rows if r["split"] == r["source_partition"] == "train"}


def reference_png(row):
    source = a1_training_rows().get(row.get("source_relative_path"))
    if not source or any((
        source["label"] != row["requested_digit"],
        source["source_index"] != row.get("source_index"),
        source["sha256"] != row.get("source_sha256"),
        row.get("source_split") != "train", row.get("source_partition") != "train"
    )):
        raise ValueError("Исходник не совпадает с обучающей выборкой A1.")
    path = (MNIST / source["relative_path"]).resolve()
    if not path.is_relative_to(MNIST.resolve()) or path.suffix.lower() != ".png":
        raise ValueError("Некорректный путь исходной цифры A1.")
    data = path.read_bytes()
    with Image.open(BytesIO(data)) as image:
        if hashlib.sha256(image.convert("L").tobytes()).hexdigest() != source["sha256"]:
            raise ValueError("Пиксели исходной цифры A1 изменились.")
    return data


class Pilots:
    def __init__(self, outputs, specific=None):
        self.outputs = outputs
        self.specific = specific
        self.lock = threading.Lock()
        self.runs = {}
        self.skipped = []

    def available(self):
        if self.specific:
            folders = [(self.specific.name, self.specific)]
        else:
            folders = [(path.parent.relative_to(self.outputs).as_posix(), path.parent)
                       for path in self.outputs.rglob("manifest.csv")]
        ready, skipped = [], []
        for run, folder in folders:
            try:
                method = self.check_export(folder)
                stamp = (folder / "manifest.csv").stat().st_mtime_ns
                ready.append((method == REFERENCE_METHOD, stamp, run, folder, method))
            except (OSError, csv.Error, KeyError, TypeError, ValueError) as error:
                skipped.append({"run": run, "reason": str(error)})
        self.runs = {run: (folder, method) for _, _, run, folder, method in sorted(ready, reverse=True)}
        self.skipped = skipped
        return self.runs

    def check_export(self, folder):
        recipe = json.loads((folder / "run.json").read_text(encoding="utf-8-sig"))
        if not isinstance(recipe, dict):
            raise ValueError("Некорректный run.json.")
        method = recipe.get("method") or "text_only"
        if not isinstance(method, str):
            raise ValueError("Некорректный метод генерации в run.json.")
        records, _, _, _ = self.tables(folder)
        pairs = {(r["requested_digit"], int(r["variant"])) for r in records.values()}
        full = pairs == {(str(d), v) for d in range(10) for v in range(1, 13)} and len(records) == 120
        if not full and not self.specific:
            raise ValueError("Для основного пилота нужны 12 вариантов каждой из 10 цифр.")
        planned = recipe.get("samples")
        if planned is not None:
            plans = {r["sample_id"]: r for r in planned}
            if len(plans) != len(planned) or plans.keys() != records.keys():
                raise ValueError("Выгружены не все образцы из run.json.")
        elif not full or method == REFERENCE_METHOD:
            raise ValueError("В run.json отсутствует план образцов.")
        for key, row in records.items():
            if planned is not None:
                fields = ("requested_digit", "variant", "path")
                if method == REFERENCE_METHOD:
                    fields += SOURCE_FIELDS
                if any(str(plans[key][k]) != row[k] for k in fields):
                    raise ValueError(f"Источник или вариант {key} отличается от run.json.")
            data = self.candidate_bytes(folder, row)
            with Image.open(BytesIO(data)) as image:
                if image.format != "PNG" or image.size != (256, 256):
                    raise ValueError(f"Ожидается PNG 256×256: {key}.")
                image.verify()
            if method == REFERENCE_METHOD:
                reference_png(row)
        return method

    def read(self, run):
        if run not in self.runs and run not in self.available():
            raise ValueError("Набор недоступен. Распакуйте Colab ZIP в assignment2/outputs/.")
        folder, _ = self.runs[run]
        return folder, *self.tables(folder)

    @staticmethod
    def tables(folder):
        columns, manifest = parse_csv((folder / "manifest.csv").read_bytes())
        if not set((*IDENTITY, "variant", "style", "source")).issubset(columns):
            raise ValueError("В manifest.csv отсутствуют столбцы варианта или происхождения.")
        raw_review = (folder / "review.csv").read_bytes()
        fields, reviews = parse_csv(raw_review)
        if not set((*IDENTITY, "status", "reviewed_digit", "notes")).issubset(fields):
            raise ValueError("В review.csv отсутствуют столбцы ручной проверки.")
        records = {row["sample_id"]: row for row in manifest}
        by_id = {row["sample_id"]: row for row in reviews}
        if (not records or len(records) != len(manifest) or len(by_id) != len(reviews)
                or set(by_id) != set(records)):
            raise ValueError("Состав manifest.csv и review.csv различается.")
        for key, row in records.items():
            if (row.get("source") != "synthetic" or row["requested_digit"] not in DIGITS
                    or any(row[field] != by_id[key][field] for field in IDENTITY)):
                raise ValueError("Проверьте происхождение, ID и SHA-256 в manifest.csv и review.csv.")
            review = by_id[key]
            if (review["status"] not in {"pending", "reject", "accept"}
                    or review["reviewed_digit"] not in DIGITS | {""}
                    or (review["status"] == "accept" and review["reviewed_digit"] == "")):
                raise ValueError(f"Некорректная ручная оценка {key}.")
        return records, by_id, fields, raw_review

    @staticmethod
    def candidate_bytes(folder, row):
        path = (folder / row["path"]).resolve()
        if not path.is_relative_to(folder.resolve()) or path.suffix.lower() != ".png":
            raise ValueError("Некорректный путь кандидата в manifest.csv.")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("PNG изменился после экспорта. Проверьте SHA-256.")
        return data

    def gallery(self, run, digit):
        if digit not in DIGITS:
            raise ValueError("Выберите цифру от 0 до 9.")
        folder, records, reviews, _, _ = self.read(run)
        method = self.runs[run][1]
        items = []
        for row in sorted(records.values(), key=lambda r: int(r["variant"])):
            if row["requested_digit"] != digit:
                continue
            key = row["sample_id"]
            assessment = reviews[key]
            query = urlencode({"pilot": run, "sample": key})
            items.append({"sample": key, "style": row["style"], "variant": row["variant"],
                          "requested_digit": digit, "url": f"/api/image?{query}",
                          "reference": {"digit": digit, "source_index": row["source_index"],
                                        "path": row["source_relative_path"], "url": f"/api/reference?{query}"}
                                       if method == REFERENCE_METHOD else None,
                          **{k: assessment[k] for k in ("status", "reviewed_digit", "notes")},
                          "revision": revision(run, assessment)})
        try:
            cli_path = folder.relative_to(HERE.parent).as_posix()
        except ValueError:
            cli_path = str(folder)
        return {"items": items, "method": method, "accepted": sum(r["status"] == "accept" for r in reviews.values()),
                "total": len(reviews), "command": f'python assignment2/train_compare.py --pilot "{cli_path}"'}

    def image(self, run, sample, reference=False):
        folder, records, _, _, _ = self.read(run)
        if sample not in records:
            raise ValueError("Такого кандидата нет в наборе.")
        if reference:
            if self.runs[run][1] != REFERENCE_METHOD:
                raise ValueError("У этого кандидата отсутствует исходник A1.")
            return reference_png(records[sample])
        return self.candidate_bytes(folder, records[sample])

    def review(self, payload):
        with self.lock:
            folder, records, reviews, fields, before = self.read(payload.get("pilot"))
            key = payload.get("sample")
            if key not in records:
                raise ValueError("Такого кандидата нет в наборе.")
            row = reviews[key]
            self.candidate_bytes(folder, records[key])
            if payload.get("revision") != revision(payload["pilot"], row):
                raise Conflict("Оценка уже изменилась. Обновите галерею.")
            status, label = payload.get("status"), payload.get("reviewed_digit", "")
            if status not in {"accept", "reject", "pending"} or label not in (DIGITS | {""}):
                raise ValueError("Проверьте статус и метку цифры.")
            if status == "accept" and (label == "" or payload.get("train_only") is not True):
                raise ValueError("Подтвердите видимую цифру и разрешите использование в train.")
            if row["status"] != "pending" and payload.get("replace") is not True:
                raise Conflict("Подтвердите изменение ранее сохранённой оценки.")
            notes = payload.get("notes", "")
            if not isinstance(notes, str) or len(notes) > 1000:
                raise ValueError("Комментарий длиннее 1000 символов.")
            row.update(status=status, reviewed_digit=label, notes=notes,
                       train_consent="train_only" if status == "accept" else "")
            if "train_consent" not in fields:
                fields.append("train_consent")
            dest = folder / "review.csv"
            with tempfile.NamedTemporaryFile(dir=folder, delete=False) as temporary:
                staging = Path(temporary.name)
            try:
                with staging.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(reviews.values())
                    handle.flush()
                    os.fsync(handle.fileno())
                if dest.read_bytes() != before:
                    raise Conflict("CSV изменился во время проверки. Обновите галерею.")
                os.replace(staging, dest)
            finally:
                staging.unlink(missing_ok=True)
            return {"status": status, "sample": key}


class Handler(BaseHTTPRequestHandler):
    def respond(self, status, data, kind="application/json; charset=utf-8"):
        if not isinstance(data, bytes):
            data = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def route(self):
        url = urlsplit(self.path)
        params = {key: vals[0] for key, vals in parse_qs(url.query).items()}
        pilots, recognizer = self.server.pilots, self.server.recognizer
        if self.command == "GET":
            if url.path == "/":
                return self.respond(200, (HERE / "demo.html").read_bytes(), "text/html; charset=utf-8")
            if url.path == "/api/state":
                with recognizer.lock:
                    recognizer.refresh()
                    info = recognizer.info()
                available = pilots.available()
                return self.respond(200, {**info, "pilots": list(available), "metrics": metrics(),
                                          "pilot_methods": {k: v[1] for k, v in available.items()},
                                          "skipped_pilots": pilots.skipped})
            if url.path == "/api/gallery":
                return self.respond(200, pilots.gallery(params["pilot"], params["digit"]))
            if url.path in {"/api/image", "/api/reference"}:
                return self.respond(200, pilots.image(params["pilot"], params["sample"],
                                                      reference=url.path == "/api/reference"), "image/png")
        elif self.command == "POST" and url.path in {"/api/predict", "/api/candidate-prediction", "/api/review"}:
            if self.headers.get_content_type() != "application/json":
                return self.respond(415, {"error": "Ожидается запрос JSON."})
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 4 * MAX_UPLOAD // 3 + 4096:
                return self.respond(413, {"error": "Некорректный размер запроса."})
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError("Некорректный запрос.")
            if url.path == "/api/review":
                return self.respond(200, pilots.review(payload))
            if url.path == "/api/candidate-prediction":
                data = pilots.image(payload["pilot"], payload["sample"])
                with Image.open(BytesIO(data)) as raw:
                    tensor = preprocess(raw)
            else:
                tensor = user_input(payload)
            return self.respond(200, recognizer.predict(tensor))
        self.respond(404, {"error": HTTPStatus.NOT_FOUND.phrase})

    def do_GET(self):
        try:
            self.route()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Conflict as exc:
            self.respond(409, {"error": str(exc)})
        except (KeyError, TypeError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            self.respond(400, {"error": str(exc)})
        except (OSError, RuntimeError) as exc:
            self.respond(500, {"error": f"Не удалось обработать данные: {exc}"})

    do_POST = do_GET


def make_server(port, outputs=None, pilot=None):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.recognizer = Recognizer()
    server.pilots = Pilots(outputs or HERE / "outputs", pilot)
    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--pilot", type=Path, help="One extracted Colab run with manifest.csv and review.csv")
    args = parser.parse_args()
    with make_server(args.port, pilot=args.pilot.resolve() if args.pilot else None) as server:
        print(f"Open http://127.0.0.1:{server.server_port} · Ctrl+C to stop", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
