import numpy as np

from rbcx.handmodel import landmarks_from_angles
from rbcx.handmodel.angle_map import AngleMapper
from calibrate_angle_map import _finger_dir_vectors_mp, _finger_dir_vectors_ume


def _neutral_open_angles():
    return np.array(
        [
            150.0, -25.0, 165.0, 178.0,
            -18.0, 112.0, 178.0, 178.0,
            -6.0, 158.0, 178.0, 178.0,
            6.0, 118.0, 178.0, 178.0,
            18.0, 104.0, 178.0, 178.0,
        ],
        dtype=np.float64,
    )


def test_neutral_open_hand_maps_to_umetrack_rest_pose():
    mapped = AngleMapper(params_path="missing-test-params.json").map(_neutral_open_angles())

    np.testing.assert_allclose(mapped, np.zeros(20), atol=np.deg2rad(0.25))


def test_neutral_open_hand_keeps_four_fingertips_ordered():
    mapped = AngleMapper(params_path="missing-test-params.json").map(_neutral_open_angles())
    tips = landmarks_from_angles(mapped)[1:5]

    assert np.all(np.diff(tips[:, 2]) < -5.0)


def test_more_index_spread_moves_index_away_from_middle():
    mapper = AngleMapper(params_path="missing-test-params.json")
    neutral = _neutral_open_angles()
    spread = neutral.copy()
    spread[4] -= 10.0

    neutral_lm = landmarks_from_angles(mapper.map(neutral))
    spread_lm = landmarks_from_angles(mapper.map(spread))

    assert spread_lm[1, 2] > neutral_lm[1, 2] + 1.0


def test_calibration_compares_directions_in_the_model_palm_frame():
    reference = landmarks_from_angles(np.zeros(20))
    ume_to_mp = np.array(
        [4, 8, 12, 16, 20, 0, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
    )
    observed = np.zeros((21, 3))
    observed[ume_to_mp] = reference[:20]
    observed[1] = 0.5 * (observed[0] + observed[2])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    observed = observed @ rotation.T * 0.002 + np.array([0.4, 0.2, -0.3])

    actual = _finger_dir_vectors_mp(observed, reference)

    np.testing.assert_allclose(actual, _finger_dir_vectors_ume(np.zeros(20)), atol=1e-4)
