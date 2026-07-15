import numpy as np

from rbcx.handmodel import landmarks_from_angles
from rbcx.handmodel.pose_geometry import canonicalize_mediapipe_landmarks


UME_TO_MP = np.array(
    [4, 8, 12, 16, 20, 0, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
)


def _mediapipe_layout_from_umetrack(ume_landmarks):
    mp_landmarks = np.zeros((21, 3), dtype=np.float64)
    mp_landmarks[UME_TO_MP] = ume_landmarks[:20]
    mp_landmarks[1] = 0.5 * (mp_landmarks[0] + mp_landmarks[2])
    return mp_landmarks


def _rotation(seed):
    rng = np.random.default_rng(seed)
    matrix, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(matrix) < 0:
        matrix[:, 0] *= -1
    return matrix


def test_canonicalization_is_rigid_transform_and_scale_invariant():
    reference = landmarks_from_angles(np.zeros(20, dtype=np.float32))
    observed = _mediapipe_layout_from_umetrack(reference)
    transformed = observed @ _rotation(7).T * 0.002 + np.array([0.3, -0.2, 1.1])

    result = canonicalize_mediapipe_landmarks(transformed, reference, "Right")

    assert result.valid
    np.testing.assert_allclose(result.landmarks, reference, atol=1e-3)


def test_canonicalization_preserves_index_to_pinky_lateral_order():
    reference = landmarks_from_angles(np.zeros(20, dtype=np.float32))
    observed = _mediapipe_layout_from_umetrack(reference)

    result = canonicalize_mediapipe_landmarks(observed, reference, "Right")

    assert result.valid
    assert np.all(np.diff(result.landmarks[1:5, 2]) < 0)


def test_canonicalization_rejects_collapsed_palm():
    reference = landmarks_from_angles(np.zeros(20, dtype=np.float32))

    result = canonicalize_mediapipe_landmarks(np.zeros((21, 3)), reference, "Right")

    assert not result.valid
    assert result.reason == "degenerate_palm"


def test_canonicalization_rejects_wrong_shape():
    reference = landmarks_from_angles(np.zeros(20, dtype=np.float32))

    result = canonicalize_mediapipe_landmarks(np.zeros((20, 3)), reference, "Right")

    assert not result.valid
    assert result.reason == "invalid_shape"
