"""Focused tests for the three processing stages."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from pipeline import extract_plates, extract_video_frames, reduce_similar_frames


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_video_extraction_honors_trim_and_fps(self) -> None:
        video = self.root / "sample.avi"
        writer = cv2.VideoWriter(
            str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (96, 64)
        )
        if not writer.isOpened():
            self.skipTest("MJPG video encoding is unavailable")
        for index in range(30):
            frame = np.full((64, 96, 3), index * 5, dtype=np.uint8)
            cv2.putText(
                frame, str(index), (16, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2,
            )
            writer.write(frame)
        writer.release()

        result = extract_video_frames(
            video, self.root / "frames", target_fps=5,
            start_seconds=1, end_seconds=2,
        )

        self.assertEqual(result.input_count, 10)
        self.assertEqual(result.output_count, 5)
        self.assertEqual(len(list((self.root / "frames").glob("*.jpg"))), 5)

    def test_similarity_reduction_retains_a_representative(self) -> None:
        source = self.root / "source"
        source.mkdir()
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        cv2.rectangle(image, (10, 15), (100, 65), (255, 255, 255), -1)
        cv2.imwrite(str(source / "first.jpg"), image)
        cv2.imwrite(str(source / "copy.jpg"), image)

        result = reduce_similar_frames(source, self.root / "reduced", keep_per_group=1)

        self.assertEqual(result.input_count, 2)
        self.assertEqual(result.output_count, 1)
        self.assertEqual(result.skipped_count, 1)

    def test_plate_extraction_keeps_unreadable_crops(self) -> None:
        source = self.root / "source"
        source.mkdir()
        cv2.imwrite(
            str(source / "vehicle.jpg"),
            np.full((80, 160, 3), 180, dtype=np.uint8),
        )
        detection = SimpleNamespace(
            bounding_box=SimpleNamespace(x1=20, y1=30, x2=120, y2=60),
            confidence=0.91,
        )
        fake_alpr = SimpleNamespace(
            predict=lambda _frame: [SimpleNamespace(detection=detection, ocr=None)]
        )

        result = extract_plates(source, self.root / "plates", fake_alpr)

        self.assertEqual(result.output_count, 1)
        self.assertEqual(
            len(list((self.root / "plates").glob("unread_plate_*.jpg"))), 1
        )
        with result.manifest_path.open(newline="") as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(row["status"], "saved")


if __name__ == "__main__":
    unittest.main()
