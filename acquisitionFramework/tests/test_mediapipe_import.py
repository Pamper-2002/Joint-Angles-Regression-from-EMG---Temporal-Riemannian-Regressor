import json
from pathlib import Path
import subprocess
import sys


FRAMEWORK = Path(__file__).resolve().parents[1]


def test_hand_tracker_import_skips_unused_tasks_tensorflow_and_audio():
    script = r'''
import json
import sys
import time

started = time.perf_counter()
from rbcx.handtracker import mediapipe as tracker_module
duration = time.perf_counter() - started

hands = tracker_module.mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    model_complexity=0,
)
hands.close()
print(json.dumps({
    "duration": duration,
    "tensorflow_loaded": "tensorflow" in sys.modules,
    "sounddevice_loaded": "sounddevice" in sys.modules,
    "tasks_python_loaded": "mediapipe.tasks.python" in sys.modules,
}))
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=FRAMEWORK,
        text=True,
        capture_output=True,
        timeout=45,
    )

    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert not result["tensorflow_loaded"]
    assert not result["sounddevice_loaded"]
    assert not result["tasks_python_loaded"]
    assert result["duration"] < 5.0


def test_emg_module_defers_tensorflow_until_model_use():
    script = r'''
import json
import sys
import EMG_regression
print(json.dumps({
    "tensorflow_loaded": "tensorflow" in sys.modules,
    "sounddevice_loaded": "sounddevice" in sys.modules,
}))
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=FRAMEWORK,
        text=True,
        capture_output=True,
        timeout=45,
    )

    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert not result["tensorflow_loaded"]
    assert not result["sounddevice_loaded"]
