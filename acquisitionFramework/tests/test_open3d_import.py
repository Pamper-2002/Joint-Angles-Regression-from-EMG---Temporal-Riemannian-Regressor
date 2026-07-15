import json
from pathlib import Path
import subprocess
import sys


FRAMEWORK = Path(__file__).resolve().parents[1]


def test_desktop_renderer_import_skips_plotly_dash_and_ipython():
    script = r'''
import json
import sys
import time

started = time.perf_counter()
from rbcx.handtracker import hand_mesh_o3d as renderer_module
duration = time.perf_counter() - started
print(json.dumps({
    "duration": duration,
    "visualizer_available": hasattr(
        renderer_module.o3d.visualization, "VisualizerWithKeyCallback"
    ),
    "dash_loaded": "dash" in sys.modules,
    "ipython_loaded": "IPython" in sys.modules,
    "plotly_loaded": "plotly" in sys.modules,
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
    assert result["visualizer_available"]
    assert not result["dash_loaded"]
    assert not result["ipython_loaded"]
    assert not result["plotly_loaded"]
    assert result["duration"] < 5.0
