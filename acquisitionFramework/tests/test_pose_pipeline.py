from dataclasses import dataclass

import numpy as np

from rbcx.handtracker.pose_pipeline import PosePipeline


@dataclass
class _Estimate:
    timestamp: float
    joint_angles: np.ndarray
    valid: bool = True


class _Solver:
    def __init__(self):
        self.solve_count = 0

    def solve(self, landmarks, timestamp, handedness, raw_angles):
        self.solve_count += 1
        return _Estimate(timestamp, np.full(20, timestamp, dtype=np.float32))


class _Renderer:
    def __init__(self):
        self.update_count = 0
        self.timestamps = []

    def update(self, angles, timestamp=None):
        self.update_count += 1
        self.timestamps.append(timestamp)
        return True


def _state(timestamp):
    return {
        "timestamp": timestamp,
        "world_landmarks": np.ones((21, 3)),
        "handedness": "Right",
        "angles_list": np.zeros(20),
    }


def test_pipeline_solves_and_renders_once_per_source_timestamp():
    solver = _Solver()
    renderer = _Renderer()
    pipeline = PosePipeline(solver, renderer)

    first = pipeline.process(_state(5.0))
    second = pipeline.process(_state(5.0))

    assert first is second
    assert solver.solve_count == 1
    assert renderer.update_count == 1
    assert renderer.timestamps == [5.0]


def test_renderer_timestamp_gate_rejects_duplicate_and_regressing_time():
    from rbcx.handtracker.hand_mesh_o3d import HandMeshVisualizer

    renderer = HandMeshVisualizer.__new__(HandMeshVisualizer)
    renderer._last_timestamp = None

    assert renderer._accept_timestamp(2.0)
    assert not renderer._accept_timestamp(2.0)
    assert not renderer._accept_timestamp(1.0)
    assert renderer._accept_timestamp(3.0)


def test_pipeline_can_solve_without_rendering_for_emg_label_collection():
    solver = _Solver()
    renderer = _Renderer()
    pipeline = PosePipeline(solver, renderer)

    estimate = pipeline.process(_state(6.0), render=False)

    assert estimate.valid
    assert solver.solve_count == 1
    assert renderer.update_count == 0
