import json
from types import SimpleNamespace

import numpy as np

from EMG import MindRoveEMG


class _Dependency:
    def __init__(self):
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1


class _Board:
    def __init__(self):
        self.stop_stream_calls = 0
        self.release_session_calls = 0

    def stop_stream(self):
        self.stop_stream_calls += 1

    def release_session(self):
        self.release_session_calls += 1


def _bare_regressor():
    from EMG_regression import EMG_regressor

    regressor = EMG_regressor.__new__(EMG_regressor)
    regressor.handTracker = _Dependency()
    regressor.mindrove = _Dependency()
    regressor.stopProgram = False
    regressor._stopped = False
    regressor.emg = [np.zeros(8) for _ in range(25)]
    regressor.label = []
    regressor.labelTs = []
    regressor.shownPred = []
    regressor.pred = np.zeros(20)
    regressor.model = None
    regressor._last_pose_timestamp = float("-inf")
    return regressor


def _pose(timestamp, value):
    return SimpleNamespace(
        timestamp=timestamp,
        valid=True,
        joint_angles=np.full(20, value, dtype=np.float32),
    )


def test_duplicate_pose_timestamp_is_not_saved_twice():
    regressor = _bare_regressor()

    assert regressor.record_pose_estimate(_pose(7.0, 0.2))
    assert not regressor.record_pose_estimate(_pose(7.0, 0.8))

    assert len(regressor.label) == 1
    assert regressor.labelTs == [25]
    np.testing.assert_array_equal(regressor.label[0], np.full(20, 0.2, dtype=np.float32))


def test_stop_is_idempotent_and_releases_dependencies():
    regressor = _bare_regressor()

    regressor.stop()
    regressor.stop()

    assert regressor.handTracker.stop_calls == 1
    assert regressor.mindrove.stop_calls == 1


def test_mindrove_stop_releases_stream_and_session_once():
    device = MindRoveEMG.__new__(MindRoveEMG)
    device.board_shim = _Board()
    device._stopped = False

    device.stop()
    device.stop()

    assert device.board_shim.stop_stream_calls == 1
    assert device.board_shim.release_session_calls == 1


def test_label_schema_file_is_explicit_and_machine_readable(tmp_path):
    from EMG_regression import LABEL_SCHEMA, write_label_schema

    path = write_label_schema(tmp_path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == LABEL_SCHEMA
    assert payload["units"] == "radians"
    assert len(payload["joint_names"]) == 20
