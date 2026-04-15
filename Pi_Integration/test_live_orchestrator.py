"""
Unit test for LiveTestOrchestrator (defined in test_nav.py).

Verifies the only behavior the subclass adds on top of main.Orchestrator:
- _send_and_wait_esp32 overrides distance_ft to TURN_ONLY_DIST_FT when
  _turn_only is True
- _send_and_wait_esp32 passes distance_ft through unchanged when
  _turn_only is False
- ARRIVED_DURATION is overridden to 5.0 (faster test iteration)

Run from Pi_Integration/:
    python test_live_orchestrator.py
"""

import os
import sys
import unittest
from unittest import mock

# Stub the heavyweight modules main.py imports so this test runs without
# Pi-only dependencies (pyaudio, torch, soundfile, serial, etc.)
_PI_DIR = os.path.dirname(os.path.abspath(__file__))
if _PI_DIR not in sys.path:
    sys.path.insert(0, _PI_DIR)

_STUB_MODULES = [
    "FunctionCalls_OA",
    "FunctionCalls_SL",
    "FunctionCalls_BCD",
    "FunctionCalls_App",
    "record",
    "record_samtry",
]
for _name in _STUB_MODULES:
    sys.modules.setdefault(_name, mock.MagicMock())

# Now safe to import test_nav (which imports main, which imports the stubs)
import test_nav  # noqa: E402
test_nav._ensure_live_test_orchestrator()  # lazy-create the subclass for testing


class LiveTestOrchestratorTest(unittest.TestCase):
    def _make(self, turn_only):
        """Construct a LiveTestOrchestrator without running __init__."""
        obj = test_nav.LiveTestOrchestrator.__new__(
            test_nav.LiveTestOrchestrator
        )
        obj._turn_only = turn_only
        return obj

    def test_arrived_duration_is_overridden(self):
        self.assertEqual(test_nav.LiveTestOrchestrator.ARRIVED_DURATION, 5.0)

    def test_turn_only_overrides_distance(self):
        obj = self._make(turn_only=True)
        with mock.patch(
            "main.Orchestrator._send_and_wait_esp32",
            return_value=True,
        ) as parent:
            result = obj._send_and_wait_esp32(angle_deg=42.0, distance_ft=8.5)
        parent.assert_called_once_with(42.0, test_nav.TURN_ONLY_DIST_FT)
        self.assertTrue(result)

    def test_drive_mode_passes_distance_through(self):
        obj = self._make(turn_only=False)
        with mock.patch(
            "main.Orchestrator._send_and_wait_esp32",
            return_value=True,
        ) as parent:
            result = obj._send_and_wait_esp32(angle_deg=-30.0, distance_ft=8.5)
        parent.assert_called_once_with(-30.0, 8.5)
        self.assertTrue(result)

    def test_turn_only_stubs_predict_ml_distance(self):
        """In turn-only mode, main.predict_ml_distance is monkey-patched
        during the inherited _localize_and_navigate call so the distance
        validation in main.Orchestrator doesn't reject 0/inf/None distances
        even when the angle is valid."""
        import main
        obj = self._make(turn_only=True)

        sentinel = main.predict_ml_distance
        seen_inside = []

        def fake_super(*args, **kwargs):
            # When called by our override, predict_ml_distance should
            # be the stub that returns TURN_ONLY_DIST_FT.
            seen_inside.append(main.predict_ml_distance(0, 0))

        with mock.patch(
            "main.Orchestrator._localize_and_navigate",
            side_effect=fake_super,
        ):
            obj._localize_and_navigate()

        self.assertEqual(seen_inside, [test_nav.TURN_ONLY_DIST_FT])
        # After the call, the original is restored.
        self.assertIs(main.predict_ml_distance, sentinel)

    def test_drive_mode_does_not_stub_predict_ml_distance(self):
        """In drive mode, main.predict_ml_distance is unchanged — the
        full distance pipeline runs and the inherited validation applies
        exactly as in main.py."""
        import main
        obj = self._make(turn_only=False)

        sentinel = main.predict_ml_distance
        unchanged_during = []

        def fake_super(*args, **kwargs):
            unchanged_during.append(main.predict_ml_distance is sentinel)

        with mock.patch(
            "main.Orchestrator._localize_and_navigate",
            side_effect=fake_super,
        ):
            obj._localize_and_navigate()

        self.assertEqual(
            unchanged_during, [True],
            "predict_ml_distance should not be patched in drive mode",
        )
        self.assertIs(main.predict_ml_distance, sentinel)


if __name__ == "__main__":
    unittest.main()
