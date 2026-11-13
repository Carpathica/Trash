"""Real-time Axis licence-plate collector with YOLO and speed metrics."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RECTIFIER = SCRIPT_DIR.parent / "number_plate_recognition" / "rectify_plates.py"


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as config_file:
        return json.load(config_file)


def resolve_path(value: str, config_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")


def box_iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


@dataclass
class Detection:
    box: tuple[float, float, float, float]
    confidence: float


def center_shift(first: Detection, second: Detection) -> float:
    first_x = (first.box[0] + first.box[2]) / 2
    first_y = (first.box[1] + first.box[3]) / 2
    second_x = (second.box[0] + second.box[2]) / 2
    second_y = (second.box[1] + second.box[3]) / 2
    return ((first_x - second_x) ** 2 + (first_y - second_y) ** 2) ** 0.5


@dataclass
class ConfirmationTracker:
    frames_required: int
    min_iou: float
    max_center_shift_px: float
    consecutive_frames: int = 0
    previous: Detection | None = None

    def update(self, detection: Detection | None) -> Detection | None:
        if detection is None:
            self.reset()
            return None
        stable = (
            self.previous is not None
            and box_iou(self.previous.box, detection.box) >= self.min_iou
            and center_shift(self.previous, detection) <= self.max_center_shift_px
        )
        if stable:
            self.consecutive_frames += 1
        else:
            self.consecutive_frames = 1
        self.previous = detection
        return detection if self.consecutive_frames >= self.frames_required else None

    def reset(self) -> None:
        self.consecutive_frames = 0
        self.previous = None


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percent
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def timing_summary(values: list[float]) -> dict[str, float | None]:
    return {
        "avg": round(sum(values) / len(values), 3) if values else None,
        "p50": round(percentile(values, 0.50), 3) if values else None,
        "p95": round(percentile(values, 0.95), 3) if values else None,
        "max": round(max(values), 3) if values else None,
    }


@dataclass
class Metrics:
    started_at: float = field(default_factory=time.monotonic)
    interval_started_at: float = field(default_factory=time.monotonic)
    total: Counter = field(default_factory=Counter)
    interval: Counter = field(default_factory=Counter)
    total_timings: dict[str, list[float]] = field(default_factory=lambda: {"read_ms": [], "inference_ms": [], "cycle_ms": []})
    interval_timings: dict[str, list[float]] = field(default_factory=lambda: {"read_ms": [], "inference_ms": [], "cycle_ms": []})

    def count(self, name: str, amount: int = 1) -> None:
        self.total[name] += amount
        self.interval[name] += amount

    def observe(self, name: str, milliseconds: float) -> None:
        self.total_timings[name].append(milliseconds)
        self.interval_timings[name].append(milliseconds)

    def snapshot(self, *, final: bool = False) -> dict[str, Any]:
        now = time.monotonic()
        elapsed = now - (self.started_at if final else self.interval_started_at)
        counters = self.total if final else self.interval
        timings = self.total_timings if final else self.interval_timings
        frames_read = counters["frames_read"]
        inference_frames = counters["inference_frames"]
        detected_frames = counters["detected_frames"]
        payload = {
            "kind": "final" if final else "interval",
            "elapsed_seconds": round(elapsed, 3),
            "frames_read": frames_read,
            "inference_frames": inference_frames,
            "detected_frames": detected_frames,
            "confirmed_detections": counters["confirmed_detections"],
            "saved_events": counters["saved_events"],
            "reconnects": counters["reconnects"],
            "read_fps": round(frames_read / elapsed, 3) if elapsed else 0.0,
            "inference_fps": round(inference_frames / elapsed, 3) if elapsed else 0.0,
            "detection_rate": round(detected_frames / inference_frames, 5) if inference_frames else 0.0,
            "read_ms": timing_summary(timings["read_ms"]),
            "inference_ms": timing_summary(timings["inference_ms"]),
            "cycle_ms": timing_summary(timings["cycle_ms"]),
        }
        if not final:
            self.interval_started_at = now
            self.interval.clear()
            for values in self.interval_timings.values():
                values.clear()
        return payload


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(payload, ensure_ascii=False) + "\n")


def print_metrics(payload: dict[str, Any]) -> None:
    inference = payload["inference_ms"]
    print(
        f"[{payload['kind']}] read={payload['read_fps']:.2f} FPS, "
        f"inference={payload['inference_fps']:.2f} FPS, "
        f"YOLO avg/p50/p95={inference['avg']}/{inference['p50']}/{inference['p95']} ms, "
        f"detections={payload['detected_frames']}, events={payload['saved_events']}, "
        f"reconnects={payload['reconnects']}"
    )


def import_rectifier(path: Path):
    spec = importlib.util.spec_from_file_location("plate_rectifier", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import rectifier: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def select_best_detection(result: Any, roi: tuple[int, int, int, int]) -> Detection | None:
    if result.boxes is None or len(result.boxes) == 0:
        return None
    confidences = result.boxes.conf.cpu().numpy()
    best_index = int(confidences.argmax())
    x1, y1, x2, y2 = result.boxes.xyxy[best_index].cpu().numpy().tolist()
    roi_x1, roi_y1, _roi_x2, _roi_y2 = roi
    return Detection((x1 + roi_x1, y1 + roi_y1, x2 + roi_x1, y2 + roi_y1), float(confidences[best_index]))


def clamp_crop(frame: Any, box: tuple[float, float, float, float], padding: int = 10) -> tuple[Any | None, tuple[int, int, int, int]]:
    import numpy as np

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    left = max(0, int(np.floor(x1)) - padding)
    top = max(0, int(np.floor(y1)) - padding)
    right = min(width, int(np.ceil(x2)) + padding)
    bottom = min(height, int(np.ceil(y2)) + padding)
    if right <= left or bottom <= top:
        return None, (left, top, right, bottom)
    return frame[top:bottom, left:right].copy(), (left, top, right, bottom)


def save_jpeg(cv2: Any, path: Path, image: Any, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise RuntimeError(f"Cannot write image: {path}")


def save_event(
    *,
    cv2: Any,
    rectifier: Any,
    output_root: Path,
    frame: Any,
    detection: Detection,
    roi: tuple[int, int, int, int],
    jpeg_quality: int,
    upscale: int,
) -> dict[str, Any]:
    stem = timestamp()
    crop, crop_box = clamp_crop(frame, detection.box)
    if crop is None:
        raise RuntimeError("Invalid detection box")
    ocr_variants = rectifier.ocr_variants(crop, upscale)
    annotated = frame.copy()
    x1, y1, x2, y2 = (int(value) for value in detection.box)
    roi_x1, roi_y1, roi_x2, roi_y2 = roi
    cv2.rectangle(annotated, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 165, 255), 2)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(annotated, f"plate {detection.confidence:.2f}", (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    paths = {
        "original": output_root / "original" / f"{stem}.jpg",
        "annotated": output_root / "annotated" / f"{stem}.jpg",
        "crop": output_root / "crops" / f"{stem}.jpg",
    }
    variant_paths = {
        name: output_root / "rectified" / name / f"{stem}_plate_{name}.jpg"
        for name in ("gray", "clahe", "adaptive")
    }
    save_jpeg(cv2, paths["original"], frame, jpeg_quality)
    save_jpeg(cv2, paths["annotated"], annotated, jpeg_quality)
    save_jpeg(cv2, paths["crop"], crop, jpeg_quality)
    for name, image in ocr_variants.items():
        save_jpeg(cv2, variant_paths[name], image, jpeg_quality)
    return {
        "timestamp": stem,
        "confidence": round(detection.confidence, 6),
        "bbox_xyxy": [round(value, 2) for value in detection.box],
        "crop_bbox_xyxy": list(crop_box),
        "roi": list(roi),
        "ocr_variants_source": "crop",
        "upscale": upscale,
        "files": {
            **{name: str(path) for name, path in paths.items()},
            **{f"rectified_{name}": str(path) for name, path in variant_paths.items()},
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect confirmed licence plates from one Axis stream.")
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.json")
    parser.add_argument("--rectifier", type=Path, default=DEFAULT_RECTIFIER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    output_root = resolve_path(str(config["output_root"]), config_path)
    weights_path = resolve_path(str(config["weights_path"]), config_path)
    roi = tuple(int(value) for value in config["roi"])
    if len(roi) != 4 or roi[2] <= roi[0] or roi[3] <= roi[1]:
        raise ValueError("roi must be [x1, y1, x2, y2] with x2>x1 and y2>y1")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    if not args.rectifier.is_file():
        raise FileNotFoundError(f"Rectifier not found: {args.rectifier}")

    import cv2
    from ultralytics import YOLO

    rectifier = import_rectifier(args.rectifier)
    model = YOLO(str(weights_path))
    tracker = ConfirmationTracker(
        frames_required=int(config["confirmation_frames"]),
        min_iou=float(config["confirmation_iou"]),
        max_center_shift_px=float(config["max_center_shift_px"]),
    )
    metrics = Metrics()
    events_path = output_root / "events.jsonl"
    metrics_path = output_root / "metrics.jsonl"
    active_vehicle = False
    last_confirmed_at: float | None = None
    capture = None
    connected_before = False
    last_metrics_at = time.monotonic()

    print("Collector started. Stop with Ctrl+C.")
    try:
        while True:
            if capture is None or not capture.isOpened():
                if capture is not None:
                    capture.release()
                capture = cv2.VideoCapture(str(config["stream_url"]))
                if not capture.isOpened():
                    print("Stream unavailable; reconnecting...")
                    time.sleep(float(config["reconnect_delay_seconds"]))
                    continue
                if connected_before:
                    metrics.count("reconnects")
                connected_before = True
                print("Stream connected.")

            read_started = time.perf_counter()
            received, frame = capture.read()
            read_ms = (time.perf_counter() - read_started) * 1000
            metrics.observe("read_ms", read_ms)
            if not received:
                print("Stream frame unavailable; reconnecting...")
                capture.release()
                capture = None
                tracker.reset()
                time.sleep(float(config["reconnect_delay_seconds"]))
                continue

            frame_started = time.perf_counter()
            metrics.count("frames_read")
            height, width = frame.shape[:2]
            roi_x1, roi_y1, roi_x2, roi_y2 = roi
            if roi_x1 < 0 or roi_y1 < 0 or roi_x2 > width or roi_y2 > height:
                raise ValueError(f"ROI {roi} is outside stream frame {width}x{height}")
            roi_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            inference_started = time.perf_counter()
            result = model.predict(source=roi_frame, conf=float(config["confidence"]), imgsz=int(config["imgsz"]), verbose=False)[0]
            inference_ms = (time.perf_counter() - inference_started) * 1000
            metrics.count("inference_frames")
            metrics.observe("inference_ms", inference_ms)
            detection = select_best_detection(result, roi)
            if detection is not None:
                x1, y1, x2, y2 = detection.box
                plate_width = x2 - x1
                plate_height = y2 - y1
                if (
                    plate_width < float(config["min_plate_width_px"])
                    or plate_height < float(config["min_plate_height_px"])
                ):
                    detection = None
                else:
                    metrics.count("detected_frames")
            confirmed = tracker.update(detection)
            now = time.monotonic()
            if confirmed is not None:
                metrics.count("confirmed_detections")
                if active_vehicle:
                    last_confirmed_at = now
                else:
                    event = save_event(
                        cv2=cv2,
                        rectifier=rectifier,
                        output_root=output_root,
                        frame=frame,
                        detection=confirmed,
                        roi=roi,
                        jpeg_quality=int(config["jpeg_quality"]),
                        upscale=int(config["upscale"]),
                    )
                    append_jsonl(events_path, event)
                    metrics.count("saved_events")
                    active_vehicle = True
                    last_confirmed_at = now
                    print(f"Saved event {event['timestamp']}.")
            elif active_vehicle and last_confirmed_at is not None and now - last_confirmed_at >= float(config["rearm_after_seconds"]):
                active_vehicle = False
                last_confirmed_at = None
                print("Collector re-armed for the next vehicle.")
            metrics.observe("cycle_ms", (time.perf_counter() - frame_started) * 1000)

            if now - last_metrics_at >= float(config["metrics_interval_seconds"]):
                snapshot = metrics.snapshot()
                append_jsonl(metrics_path, snapshot)
                print_metrics(snapshot)
                last_metrics_at = now
    except KeyboardInterrupt:
        print("Stopping collector.")
    finally:
        if capture is not None:
            capture.release()
        summary = metrics.snapshot(final=True)
        append_jsonl(metrics_path, summary)
        print_metrics(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
