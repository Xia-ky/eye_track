"""Future-distance early stopping must distinguish progress from noise."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


SOFTWARE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOFTWARE_ROOT))

from see_v1.early_stopping import FutureDistanceEarlyStopping


class FutureDistanceEarlyStoppingTests(unittest.TestCase):
    def _tracker(self) -> FutureDistanceEarlyStopping:
        return FutureDistanceEarlyStopping(patience=5, min_delta=0.1)

    def test_first_finite_value_establishes_baseline(self) -> None:
        tracker = self._tracker()

        self.assertFalse(tracker.update(40.0))
        self.assertEqual(40.0, tracker.best_distance)
        self.assertEqual(0, tracker.epochs_without_improvement)

    def test_meaningful_improvement_resets_counter(self) -> None:
        tracker = self._tracker()
        tracker.update(40.0)
        tracker.update(39.95)
        tracker.update(40.2)

        self.assertFalse(tracker.update(39.8))
        self.assertEqual(39.8, tracker.best_distance)
        self.assertEqual(0, tracker.epochs_without_improvement)

    def test_small_improvement_does_not_reset_counter(self) -> None:
        tracker = self._tracker()
        tracker.update(40.0)

        self.assertFalse(tracker.update(39.95))
        self.assertEqual(40.0, tracker.best_distance)
        self.assertEqual(1, tracker.epochs_without_improvement)

    def test_fifth_consecutive_non_improvement_stops(self) -> None:
        tracker = self._tracker()
        tracker.update(40.0)

        for distance in (40.1, 40.0, 39.95, math.inf):
            self.assertFalse(tracker.update(distance))
        self.assertTrue(tracker.update(41.0))
        self.assertEqual(5, tracker.epochs_without_improvement)

    def test_improvement_restarts_patience_window(self) -> None:
        tracker = self._tracker()
        tracker.update(40.0)
        for distance in (40.1, 40.2, 40.3, 40.4):
            tracker.update(distance)

        self.assertFalse(tracker.update(39.0))
        self.assertEqual(0, tracker.epochs_without_improvement)
        for distance in (39.1, 39.2, 39.3, 39.4):
            self.assertFalse(tracker.update(distance))
        self.assertTrue(tracker.update(39.5))


if __name__ == "__main__":
    unittest.main()
