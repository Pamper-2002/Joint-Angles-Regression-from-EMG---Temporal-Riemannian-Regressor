"""Palm-local landmark alignment between MediaPipe and the UmeTrack hand model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


UME_TO_MP = np.array(
    [4, 8, 12, 16, 20, 0, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19],
    dtype=np.int64,
)


@dataclass(frozen=True)
class CanonicalHand:
    landmarks: np.ndarray
    valid: bool
    scale: float
    reason: str = ""


def _invalid(reason: str) -> CanonicalHand:
    return CanonicalHand(np.empty((0, 3), dtype=np.float64), False, 0.0, reason)


def _unit(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < eps:
        return None
    return vector / norm


def palm_basis(
    wrist: np.ndarray,
    index_mcp: np.ndarray,
    middle_mcp: np.ndarray,
    pinky_mcp: np.ndarray,
) -> np.ndarray | None:
    """Return orthonormal columns ``(+longitudinal, +normal, +index-side)``."""
    longitudinal = _unit(middle_mcp - wrist)
    if longitudinal is None:
        return None

    lateral_raw = index_mcp - pinky_mcp
    lateral = _unit(lateral_raw - np.dot(lateral_raw, longitudinal) * longitudinal)
    if lateral is None:
        return None

    normal = _unit(np.cross(lateral, longitudinal))
    if normal is None:
        return None
    lateral = _unit(np.cross(longitudinal, normal))
    if lateral is None:
        return None
    return np.column_stack((longitudinal, normal, lateral))


def _robust_scale(points: np.ndarray, reference: np.ndarray) -> float | None:
    observed_lengths = np.array(
        [np.linalg.norm(points[5] - points[17]), np.linalg.norm(points[9] - points[0])]
    )
    reference_lengths = np.array(
        [np.linalg.norm(reference[8] - reference[17]), np.linalg.norm(reference[11] - reference[5])]
    )
    if np.any(observed_lengths < 1e-8) or not np.all(np.isfinite(observed_lengths)):
        return None
    ratios = reference_lengths / observed_lengths
    scale = float(np.median(ratios))
    if not np.isfinite(scale) or scale <= 0.0 or scale > 1e7:
        return None
    return scale


def canonicalize_mediapipe_landmarks(
    mp_landmarks,
    reference_landmarks,
    handedness: str = "Right",
) -> CanonicalHand:
    """Align MediaPipe world landmarks to the neutral UmeTrack model frame.

    The operation removes global translation, rotation, and scale while retaining
    articulation. Only the right-hand model is supported by the current renderer.
    """
    points = np.asarray(mp_landmarks, dtype=np.float64)
    reference = np.asarray(reference_landmarks, dtype=np.float64)
    if (
        points.shape != (21, 3)
        or reference.shape != (21, 3)
        or not np.all(np.isfinite(points))
        or not np.all(np.isfinite(reference))
    ):
        return _invalid("invalid_shape")
    if handedness != "Right":
        return _invalid("unsupported_handedness")

    observed_basis = palm_basis(points[0], points[5], points[9], points[17])
    reference_basis = palm_basis(reference[5], reference[8], reference[11], reference[17])
    scale = _robust_scale(points, reference)
    if observed_basis is None or reference_basis is None or scale is None:
        return _invalid("degenerate_palm")

    selected = points[UME_TO_MP]
    local_coordinates = (selected - points[0]) @ observed_basis
    target = np.empty((21, 3), dtype=np.float64)
    target[:20] = reference[5] + scale * (local_coordinates @ reference_basis.T)
    # The MediaPipe topology has no direct equivalent of UmeTrack's palm center.
    # It is a rigid palm landmark, so the model's calibrated value is authoritative.
    target[20] = reference[20]
    return CanonicalHand(target, True, scale)


__all__ = ["CanonicalHand", "UME_TO_MP", "canonicalize_mediapipe_landmarks", "palm_basis"]
