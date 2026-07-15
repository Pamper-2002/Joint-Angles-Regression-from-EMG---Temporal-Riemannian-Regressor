import subprocess
import sys
from pathlib import Path


FRAMEWORK = Path(__file__).resolve().parents[1]


def test_emg_pose_schema_deduplication_and_lifecycle_in_isolated_process():
    script = r'''
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from EMG import MindRoveEMG
from EMG_regression import EMG_regressor, LABEL_SCHEMA, write_label_schema

class Dependency:
    def __init__(self): self.stop_calls = 0
    def stop(self): self.stop_calls += 1

regressor = EMG_regressor.__new__(EMG_regressor)
regressor.handTracker = Dependency()
regressor.mindrove = Dependency()
regressor.stopProgram = False
regressor._stopped = False
regressor.emg = [np.zeros(8) for _ in range(25)]
regressor.label = []
regressor.labelTs = []
regressor.shownPred = []
regressor.pred = np.zeros(20)
regressor.model = None
regressor._last_pose_timestamp = float("-inf")

pose = SimpleNamespace(
    timestamp=7.0, valid=True,
    joint_angles=np.full(20, 0.2, dtype=np.float32),
)
assert regressor.record_pose_estimate(pose)
assert not regressor.record_pose_estimate(pose)
assert len(regressor.label) == 1
assert regressor.labelTs == [25]

regressor.stop()
regressor.stop()
assert regressor.handTracker.stop_calls == 1
assert regressor.mindrove.stop_calls == 1

class Board:
    def __init__(self):
        self.stop_stream_calls = 0
        self.release_session_calls = 0
    def stop_stream(self): self.stop_stream_calls += 1
    def release_session(self): self.release_session_calls += 1

device = MindRoveEMG.__new__(MindRoveEMG)
device.board_shim = Board()
device._stopped = False
device.stop()
device.stop()
assert device.board_shim.stop_stream_calls == 1
assert device.board_shim.release_session_calls == 1

with tempfile.TemporaryDirectory() as directory:
    path = write_label_schema(directory)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assert payload["schema"] == LABEL_SCHEMA
    assert payload["units"] == "radians"
    assert len(payload["joint_names"]) == 20
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=FRAMEWORK,
        text=True,
        capture_output=True,
        timeout=90,
    )

    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
