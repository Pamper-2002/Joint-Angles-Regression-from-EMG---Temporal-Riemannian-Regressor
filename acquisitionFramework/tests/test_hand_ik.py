import numpy as np

from rbcx.handmodel import joint_limits, landmarks_from_angles
from rbcx.handmodel.hand_ik import HandIKSolver


UME_TO_MP = np.array(
    [4, 8, 12, 16, 20, 0, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
)


def _rotation(seed):
    rng = np.random.default_rng(seed)
    matrix, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(matrix) < 0:
        matrix[:, 0] *= -1
    return matrix


def _observation_from_angles(angles, seed=11):
    ume = landmarks_from_angles(angles)
    mp = np.zeros((21, 3), dtype=np.float64)
    mp[UME_TO_MP] = ume[:20]
    mp[1] = 0.5 * (mp[0] + mp[2])
    return mp @ _rotation(seed).T * 0.002 + np.array([0.2, -0.1, 0.8])


def _legal_test_pose():
    limits = joint_limits()[:20]
    pose = np.zeros(20, dtype=np.float32)
    fe = np.array([0, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19])
    aa = np.array([1, 4, 8, 12, 16])
    pose[fe] = limits[fe, 0] + 0.35 * (limits[fe, 1] - limits[fe, 0])
    pose[aa] = limits[aa, 0] + 0.55 * (limits[aa, 1] - limits[aa, 0])
    return pose


def test_fk_pose_round_trip_recovers_articulation():
    expected = _legal_test_pose()
    solver = HandIKSolver(iterations=90, learning_rate=0.06, convergence_rmse=0.02)

    estimate = solver.solve(_observation_from_angles(expected), timestamp=1.0, handedness="Right")

    assert estimate.valid
    assert estimate.converged
    assert np.median(np.abs(estimate.joint_angles - expected)) < np.deg2rad(5.0)
    assert estimate.rmse < 0.02


def test_duplicate_timestamp_returns_cached_estimate():
    solver = HandIKSolver(iterations=5)
    first = solver.solve(_observation_from_angles(np.zeros(20)), 1.0, "Right")
    second = solver.solve(_observation_from_angles(_legal_test_pose()), 1.0, "Right")

    assert second is first
    assert solver.solve_count == 1


def test_solution_never_exceeds_model_limits():
    solver = HandIKSolver(iterations=20)
    noisy = _observation_from_angles(_legal_test_pose())
    noisy += np.random.default_rng(3).normal(scale=0.004, size=noisy.shape)

    estimate = solver.solve(noisy, 2.0, "Right")

    limits = joint_limits()[:20]
    assert np.all(estimate.joint_angles >= limits[:, 0] - 1e-7)
    assert np.all(estimate.joint_angles <= limits[:, 1] + 1e-7)


def test_invalid_frame_holds_previous_valid_pose():
    solver = HandIKSolver(iterations=10)
    first = solver.solve(_observation_from_angles(np.zeros(20)), 1.0, "Right")

    held = solver.solve(np.zeros((21, 3)), 1.1, "Right")

    assert held.valid
    assert held.source == "held"
    np.testing.assert_array_equal(held.joint_angles, first.joint_angles)
