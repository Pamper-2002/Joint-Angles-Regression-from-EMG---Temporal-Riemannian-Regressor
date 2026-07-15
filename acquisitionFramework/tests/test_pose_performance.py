import time

import numpy as np

import config
from rbcx.handmodel import joint_limits, landmarks_from_angles, mesh_from_angles
from rbcx.handmodel.hand_ik import HandIKSolver
from rbcx.handmodel.angle_map import JOINT_NAMES


UME_TO_MP = np.array(
    [4, 8, 12, 16, 20, 0, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
)


def _benchmark_pose():
    limits = joint_limits()[:20]
    angles = limits[:, 0] + 0.35 * (limits[:, 1] - limits[:, 0])
    aa = np.array([1, 4, 8, 12, 16])
    angles[aa] = limits[aa, 0] + 0.55 * (limits[aa, 1] - limits[aa, 0])
    return angles.astype(np.float32)


def _observation(angles):
    ume = landmarks_from_angles(angles)
    points = np.zeros((21, 3), dtype=np.float64)
    points[UME_TO_MP] = ume[:20]
    points[1] = 0.5 * (points[0] + points[2])
    return points * 0.002 + np.array([0.2, -0.1, 0.8])


def test_realtime_ik_budget_converges_across_video_frames():
    expected = _benchmark_pose()
    solver = HandIKSolver(
        iterations=config.HAND_IK_ITERATIONS,
        convergence_rmse=config.HAND_IK_CONVERGENCE_RMSE,
    )
    durations = []
    estimates = []
    for frame in range(10):
        started = time.perf_counter()
        estimates.append(solver.solve(_observation(expected), frame + 1.0, "Right"))
        durations.append((time.perf_counter() - started) * 1000.0)

    median_ms = float(np.median(durations[2:]))
    median_error_deg = float(
        np.rad2deg(np.median(np.abs(estimates[-1].joint_angles - expected)))
    )
    print(f"IK median={median_ms:.2f} ms, settled error={median_error_deg:.2f} deg")

    assert median_ms < 27.0
    assert any(estimate.converged for estimate in estimates[:3])
    assert median_error_deg < 5.0


def test_fk_lbs_budget_is_finite_and_realtime():
    angles = _benchmark_pose()
    durations = []
    for _ in range(40):
        started = time.perf_counter()
        vertices, triangles = mesh_from_angles(angles)
        durations.append((time.perf_counter() - started) * 1000.0)

    median_ms = float(np.median(durations[5:]))
    print(f"FK+LBS median={median_ms:.2f} ms")

    assert vertices.shape[1] == 3
    assert triangles.shape[1] == 3
    assert np.all(np.isfinite(vertices))
    assert median_ms < 10.0


def test_five_pose_protocol_settles_after_abrupt_transitions():
    limits = joint_limits()[:20]
    neutral = np.clip(np.zeros(20), limits[:, 0], limits[:, 1])
    fe = np.array([i for i, name in enumerate(JOINT_NAMES) if name.endswith("_FE")])
    aa = np.array([i for i, name in enumerate(JOINT_NAMES) if name.endswith("_AA")])

    fist = neutral.copy()
    fist[fe] = limits[fe, 0] + 0.82 * (limits[fe, 1] - limits[fe, 0])
    pinch = neutral.copy()
    pinch_joints = np.array([0, 2, 3, 5, 6, 7])
    pinch[pinch_joints] = limits[pinch_joints, 0] + 0.65 * (
        limits[pinch_joints, 1] - limits[pinch_joints, 0]
    )
    spread = neutral.copy()
    spread[aa] = limits[aa, 0] + np.array([0.25, 0.15, 0.35, 0.65, 0.85]) * (
        limits[aa, 1] - limits[aa, 0]
    )
    index_flexion = neutral.copy()
    index_joints = np.array([5, 6, 7])
    index_flexion[index_joints] = limits[index_joints, 0] + 0.65 * (
        limits[index_joints, 1] - limits[index_joints, 0]
    )

    solver = HandIKSolver(iterations=config.HAND_IK_ITERATIONS)
    timestamp = 0.0
    for name, expected in {
        "open": neutral,
        "fist": fist,
        "pinch": pinch,
        "spread": spread,
        "index_flexion": index_flexion,
    }.items():
        for _ in range(10):
            timestamp += 1.0 / 30.0
            estimate = solver.solve(_observation(expected), timestamp, "Right")
        error_deg = float(np.rad2deg(np.median(np.abs(estimate.joint_angles - expected))))
        assert estimate.converged, f"{name}: rmse={estimate.rmse:.3f}"
        assert error_deg < 5.0, f"{name}: median error={error_deg:.2f} deg"
