import numpy as np

from rbcx.handtracker.mediapipe import MediaPipeHandTracker


def _angles(tracker):
    return {name: float(index) for index, name in enumerate(tracker.joint_names)}


def test_realtime_defaults_track_one_hand_with_lightweight_model():
    tracker = MediaPipeHandTracker(show_window=False)

    assert tracker.max_num_hands == 1
    assert tracker.model_complexity == 0


def test_hand_state_is_one_atomic_observation():
    tracker = MediaPipeHandTracker(show_window=False)
    normalized = np.arange(63, dtype=np.float64).reshape(21, 3)
    world = normalized * 0.001

    tracker._publish_observation(
        "Right",
        timestamp=3.0,
        angles=_angles(tracker),
        normalized_landmarks=normalized,
        world_landmarks=world,
    )
    state = tracker.get_hand_state("Right")

    assert state["timestamp"] == 3.0
    assert state["handedness"] == "Right"
    np.testing.assert_array_equal(state["normalized_landmarks"], normalized)
    np.testing.assert_array_equal(state["world_landmarks"], world)
    np.testing.assert_array_equal(state["landmarks"], world)


def test_selected_angle_points_follow_configuration():
    normalized = np.ones((21, 3), dtype=np.float64)
    world = np.full((21, 3), 2.0, dtype=np.float64)

    tracker_2d = MediaPipeHandTracker(show_window=False, use_2D_coord_for_angles=True)
    tracker_3d = MediaPipeHandTracker(show_window=False, use_2D_coord_for_angles=False)

    np.testing.assert_array_equal(tracker_2d._select_angle_points(normalized, world), normalized)
    np.testing.assert_array_equal(tracker_3d._select_angle_points(normalized, world), world)


def test_tracker_exposes_distinct_rate_counters():
    tracker = MediaPipeHandTracker(show_window=False)

    assert tracker.capture_fps == 0.0
    assert tracker.inference_fps == 0.0
    assert tracker.pose_fps == 0.0
