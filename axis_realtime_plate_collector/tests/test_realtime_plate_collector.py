import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "realtime_plate_collector.py"
SPEC = importlib.util.spec_from_file_location("realtime_collector", SCRIPT_PATH)
collector = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = collector
SPEC.loader.exec_module(collector)


class ConfirmationTrackerTests(unittest.TestCase):
    def test_confirms_three_stable_detections(self):
        tracker = collector.ConfirmationTracker(frames_required=3, min_iou=0.5)
        first = collector.Detection((10, 10, 110, 50), 0.8)
        second = collector.Detection((12, 11, 112, 51), 0.9)
        third = collector.Detection((13, 12, 113, 52), 0.95)
        self.assertIsNone(tracker.update(first))
        self.assertIsNone(tracker.update(second))
        self.assertEqual(tracker.update(third), third)

    def test_low_iou_resets_confirmation(self):
        tracker = collector.ConfirmationTracker(frames_required=3, min_iou=0.5)
        tracker.update(collector.Detection((0, 0, 100, 50), 0.8))
        self.assertIsNone(tracker.update(collector.Detection((300, 300, 400, 350), 0.9)))
        self.assertEqual(tracker.consecutive_frames, 1)

    def test_missing_detection_resets_confirmation(self):
        tracker = collector.ConfirmationTracker(frames_required=3, min_iou=0.5)
        tracker.update(collector.Detection((0, 0, 100, 50), 0.8))
        tracker.update(None)
        self.assertEqual(tracker.consecutive_frames, 0)
        self.assertIsNone(tracker.previous)


class MetricTests(unittest.TestCase):
    def test_summary_includes_rates_and_percentiles(self):
        metrics = collector.Metrics()
        metrics.count("frames_read", 4)
        metrics.count("inference_frames", 4)
        metrics.count("detected_frames", 2)
        for value in (10.0, 20.0, 30.0, 40.0):
            metrics.observe("inference_ms", value)
        summary = metrics.snapshot(final=True)
        self.assertEqual(summary["detection_rate"], 0.5)
        self.assertEqual(summary["inference_ms"]["p50"], 25.0)
        self.assertEqual(summary["inference_ms"]["p95"], 38.5)


if __name__ == "__main__":
    unittest.main()
