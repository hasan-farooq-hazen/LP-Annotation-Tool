"""Reusable processing stages for the FastALPR Streamlit application."""

from __future__ import annotations

import csv
import io
import json
import math
import re
import shutil
import zipfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import cv2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
ProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True)
class StageResult:
    """Summary returned by every processing stage."""

    input_count: int
    output_count: int
    skipped_count: int
    error_count: int
    output_dir: Path
    manifest_path: Path


def _notify(callback: ProgressCallback | None, current: int, total: int, text: str) -> None:
    if callback is not None:
        callback(current, total, text)


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def image_paths(root: Path) -> list[Path]:
    """Return supported images below a directory in deterministic order."""
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def unpack_image_uploads(
    uploads: Sequence[tuple[str, bytes]], destination: Path
) -> list[Path]:
    """Safely unpack uploaded images and ZIP archives into ``destination``."""
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def unique_path(name: str) -> Path:
        clean_name = Path(name).name
        stem = Path(clean_name).stem or "image"
        suffix = Path(clean_name).suffix.lower()
        candidate = destination / f"{stem}{suffix}"
        number = 2
        while candidate.exists():
            candidate = destination / f"{stem}_{number}{suffix}"
            number += 1
        return candidate

    for upload_name, payload in uploads:
        suffix = Path(upload_name).suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            output = unique_path(upload_name)
            output.write_bytes(payload)
            written.append(output)
            continue
        if suffix != ".zip":
            continue

        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = [member for member in archive.infolist() if not member.is_dir()]
            if sum(member.file_size for member in members) > 2_000_000_000:
                raise ValueError("The ZIP expands beyond the 2 GB safety limit.")
            if len(members) > 50_000:
                raise ValueError("The ZIP contains too many files.")
            for member in members:
                member_path = PurePosixPath(member.filename)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ValueError("The ZIP contains an unsafe file path.")
                if member_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                output = unique_path(member_path.name)
                with archive.open(member) as source, output.open("wb") as target:
                    shutil.copyfileobj(source, target)
                written.append(output)
    return written


def make_zip_bytes(directory: Path) -> bytes:
    """Create an in-memory ZIP containing all files below ``directory``."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(directory).as_posix())
    return buffer.getvalue()


def extract_video_frames(
    video_path: Path,
    output_dir: Path,
    target_fps: int,
    jpeg_quality: int = 92,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    progress: ProgressCallback | None = None,
) -> StageResult:
    """Sample a video at the selected rate and save frames plus a CSV manifest."""
    if target_fps not in {5, 10, 15, 20}:
        raise ValueError("Frame rate must be one of 5, 10, 15, or 20 FPS.")
    if not 60 <= jpeg_quality <= 100:
        raise ValueError("JPEG quality must be between 60 and 100.")

    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("The uploaded video could not be opened.")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(source_fps) or source_fps <= 0:
        capture.release()
        raise ValueError("The video does not report a valid frame rate.")

    duration = source_frames / source_fps if source_frames > 0 else 0.0
    effective_end = duration if end_seconds is None else min(end_seconds, duration)
    if start_seconds < 0 or effective_end <= start_seconds:
        capture.release()
        raise ValueError("The selected trim range is not valid.")
    start_frame = max(0, round(start_seconds * source_fps))
    end_frame = min(source_frames, round(effective_end * source_fps))
    selected_source_frames = max(0, end_frame - start_frame)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    effective_fps = min(float(target_fps), source_fps)
    interval = source_fps / effective_fps
    expected = math.ceil(selected_source_frames / interval) if selected_source_frames else 0
    manifest_path = output_dir / "manifest.csv"
    rows: list[dict[str, object]] = []
    decoded_index = start_frame
    sample_index = errors = 0
    next_sample = start_frame

    try:
        while True:
            if source_frames > 0 and decoded_index >= end_frame:
                break
            ok, frame = capture.read()
            if not ok:
                break
            if decoded_index < next_sample:
                decoded_index += 1
                continue

            filename = f"frame_{sample_index + 1:08d}.jpg"
            timestamp = decoded_index / source_fps
            saved = cv2.imwrite(
                str(output_dir / filename),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            )
            status = "saved" if saved else "error"
            if not saved:
                errors += 1
            rows.append(
                {
                    "source_frame_index": decoded_index,
                    "sample_index": sample_index,
                    "timestamp_seconds": f"{timestamp:.3f}",
                    "width": frame.shape[1],
                    "height": frame.shape[0],
                    "output_relative": filename if saved else "",
                    "status": status,
                }
            )
            sample_index += 1
            next_sample = start_frame + round(sample_index * interval)
            decoded_index += 1
            if sample_index == 1 or sample_index % 10 == 0:
                _notify(progress, sample_index, expected, "Extracting video frames")
    finally:
        capture.release()

    fields = [
        "source_frame_index", "sample_index", "timestamp_seconds", "width",
        "height", "output_relative", "status",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _write_json(
        output_dir / "run_config.json",
        {
            "source_file": video_path.name,
            "requested_fps": target_fps,
            "source_fps": source_fps,
            "effective_fps": effective_fps,
            "source_frame_count": source_frames,
            "video_duration_seconds": duration,
            "trim_start_seconds": start_seconds,
            "trim_end_seconds": effective_end,
            "jpeg_quality": jpeg_quality,
        },
    )
    _notify(progress, sample_index, expected or sample_index, "Frames ready")
    return StageResult(
        input_count=selected_source_frames,
        output_count=sample_index - errors,
        skipped_count=max(0, selected_source_frames - sample_index),
        error_count=errors,
        output_dir=output_dir,
        manifest_path=manifest_path,
    )


def difference_hash(image, hash_size: int = 8) -> int:
    """Return a 64-bit perceptual difference hash."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    differences = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in differences.flat:
        value = (value << 1) | int(bit)
    return value


class _HashNode:
    def __init__(self, image_hash: int, group: int) -> None:
        self.image_hash = image_hash
        self.group = group
        self.children: dict[int, _HashNode] = {}


class _HashTree:
    """BK-tree supporting fast Hamming-distance searches."""

    def __init__(self) -> None:
        self.root: _HashNode | None = None

    def insert(self, image_hash: int, group: int) -> None:
        if self.root is None:
            self.root = _HashNode(image_hash, group)
            return
        node = self.root
        while True:
            distance = (image_hash ^ node.image_hash).bit_count()
            if distance not in node.children:
                node.children[distance] = _HashNode(image_hash, group)
                return
            node = node.children[distance]

    def search(self, image_hash: int, max_distance: int) -> list[tuple[int, int]]:
        if self.root is None:
            return []
        matches: list[tuple[int, int]] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = (image_hash ^ node.image_hash).bit_count()
            if distance <= max_distance:
                matches.append((distance, node.group))
            stack.extend(
                child
                for edge, child in node.children.items()
                if distance - max_distance <= edge <= distance + max_distance
            )
        return matches


def reduce_similar_frames(
    input_dir: Path,
    output_dir: Path,
    max_distance: int = 6,
    keep_per_group: int = 1,
    progress: ProgressCallback | None = None,
) -> StageResult:
    """Group visually similar frames and retain the sharpest representatives."""
    if not 0 <= max_distance <= 20:
        raise ValueError("Similarity distance must be between 0 and 20.")
    if keep_per_group not in {1, 2, 3}:
        raise ValueError("Frames kept per group must be 1, 2, or 3.")

    images = image_paths(input_dir)
    if not images:
        raise ValueError("No JPG or PNG images were provided.")
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, object]] = []
    rows: dict[Path, dict[str, object]] = {}
    errors = 0

    for index, source in enumerate(images, start=1):
        relative = source.relative_to(input_dir)
        row: dict[str, object] = {
            "source_relative": relative.as_posix(),
            "similarity_hash": "",
            "sharpness": "",
            "group": "",
            "duplicate_of": "",
            "output_relative": "",
            "status": "candidate",
            "error": "",
        }
        try:
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Image could not be decoded")
            image_hash = difference_hash(image)
            sharpness = float(
                cv2.Laplacian(
                    cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F
                ).var()
            )
            row["similarity_hash"] = f"{image_hash:016x}"
            row["sharpness"] = f"{sharpness:.3f}"
            candidates.append(
                {"source": source, "relative": relative, "hash": image_hash, "sharpness": sharpness}
            )
        except Exception as exc:  # one corrupt image should not stop a large job
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
            errors += 1
        rows[source] = row
        if index == 1 or index % 10 == 0:
            _notify(progress, index, len(images), "Comparing frames")

    ranked = sorted(candidates, key=lambda item: float(item["sharpness"]), reverse=True)
    groups: list[list[dict[str, object]]] = []
    tree = _HashTree()
    for candidate in ranked:
        matches = tree.search(int(candidate["hash"]), max_distance)
        if matches:
            _, group_index = min(matches)
            groups[group_index].append(candidate)
        else:
            group_index = len(groups)
            groups.append([candidate])
            tree.insert(int(candidate["hash"]), group_index)

    retained: list[dict[str, object]] = []
    for group_index, group in enumerate(groups, start=1):
        selected = group[:keep_per_group]
        retained.extend(selected)
        best_relative = Path(selected[0]["relative"]).as_posix()
        for candidate in group:
            row = rows[Path(candidate["source"])]
            row["group"] = group_index
            if any(candidate is chosen for chosen in selected):
                row["status"] = "saved"
            else:
                row["status"] = "similar_frame_skipped"
                row["duplicate_of"] = best_relative

    copy_errors = 0
    for candidate in retained:
        source = Path(candidate["source"])
        relative = Path(candidate["relative"])
        destination = output_dir / relative
        row = rows[source]
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            row["output_relative"] = relative.as_posix()
        except Exception as exc:
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
            copy_errors += 1

    manifest_path = output_dir / "manifest.csv"
    fields = [
        "source_relative", "similarity_hash", "sharpness", "group",
        "duplicate_of", "output_relative", "status", "error",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows[path] for path in images)
    _write_json(
        output_dir / "run_config.json",
        {
            "input_images": len(images),
            "similarity_method": "64-bit full-frame difference hash",
            "maximum_hamming_distance": max_distance,
            "frames_kept_per_group": keep_per_group,
        },
    )
    _notify(progress, len(images), len(images), "Best frames selected")
    return StageResult(
        input_count=len(images),
        output_count=len(retained) - copy_errors,
        skipped_count=len(candidates) - len(retained),
        error_count=errors + copy_errors,
        output_dir=output_dir,
        manifest_path=manifest_path,
    )


def _safe_label(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_-")[:48]


def _mean_confidence(value: float | Iterable[float] | None) -> float | str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return float(value)
    values = list(value)
    return sum(values) / len(values) if values else ""


def extract_plates(
    input_dir: Path,
    output_dir: Path,
    alpr: object,
    progress: ProgressCallback | None = None,
) -> StageResult:
    """Detect, crop, and OCR plates from a directory of images."""
    images = image_paths(input_dir)
    if not images:
        raise ValueError("No JPG or PNG images were provided.")
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "source_relative", "detection_index", "status", "ocr", "ocr_confidence",
        "detector_confidence", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "crop_width", "crop_height", "output_relative", "error",
    ]
    rows: list[dict[str, object]] = []
    saved = no_detection = errors = plate_number = 0

    def blank_row(relative: str, status: str, error: str = "") -> dict[str, object]:
        row: dict[str, object] = {field: "" for field in fields}
        row.update(source_relative=relative, status=status, error=error)
        return row

    for image_index, source in enumerate(images, start=1):
        relative = source.relative_to(input_dir).as_posix()
        try:
            frame = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Image could not be decoded")
            results = alpr.predict(frame)  # type: ignore[attr-defined]
            if not results:
                rows.append(blank_row(relative, "no_plate_detected"))
                no_detection += 1
            for detection_index, result in enumerate(results, start=1):
                box = result.detection.bounding_box
                x1, y1 = max(int(box.x1), 0), max(int(box.y1), 0)
                x2 = min(int(box.x2), frame.shape[1])
                y2 = min(int(box.y2), frame.shape[0])
                crop = frame[y1:y2, x1:x2]
                ocr = result.ocr
                text = ocr.text if ocr is not None and ocr.text else ""
                plate_number += 1
                label = _safe_label(text) or f"unread_plate_{plate_number:06d}"
                suffix = source.suffix.lower() if source.suffix.lower() in IMAGE_SUFFIXES else ".jpg"
                destination = output_dir / f"{label}{suffix}"
                duplicate = 2
                while destination.exists():
                    destination = output_dir / f"{label}_{duplicate}{suffix}"
                    duplicate += 1
                row: dict[str, object] = {
                    "source_relative": relative,
                    "detection_index": detection_index,
                    "status": "saved",
                    "ocr": text,
                    "ocr_confidence": _mean_confidence(ocr.confidence if ocr else None),
                    "detector_confidence": result.detection.confidence,
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    "crop_width": max(0, x2 - x1),
                    "crop_height": max(0, y2 - y1),
                    "output_relative": destination.name,
                    "error": "",
                }
                if not crop.size or not cv2.imwrite(str(destination), crop):
                    row["status"] = "error"
                    row["output_relative"] = ""
                    row["error"] = "The detected plate crop could not be saved."
                    errors += 1
                else:
                    saved += 1
                rows.append(row)
        except Exception as exc:
            rows.append(blank_row(relative, "error", f"{type(exc).__name__}: {exc}"))
            errors += 1
        if image_index == 1 or image_index % 5 == 0:
            _notify(progress, image_index, len(images), "Finding and reading plates")

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _write_json(
        output_dir / "run_config.json",
        {"input_images": len(images), "output_plate_crops": saved},
    )
    _notify(progress, len(images), len(images), "Plate crops ready")
    return StageResult(
        input_count=len(images),
        output_count=saved,
        skipped_count=no_detection,
        error_count=errors,
        output_dir=output_dir,
        manifest_path=manifest_path,
    )
