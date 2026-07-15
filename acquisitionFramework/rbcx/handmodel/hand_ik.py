"""Stateful, joint-limited inverse kinematics for the UmeTrack hand model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .fk import joint_limits, landmarks_from_angles, landmarks_tensor_from_angles
from .pose_geometry import canonicalize_mediapipe_landmarks


@dataclass(frozen=True)
class PoseEstimate:
    timestamp: float
    joint_angles: np.ndarray
    canonical_landmarks: np.ndarray
    fitted_landmarks: np.ndarray
    rmse: float
    valid: bool
    converged: bool
    source: str
    limit_hits: np.ndarray


class HandIKSolver:
    """Fit MediaPipe landmarks to UmeTrack angles once per source timestamp."""

    def __init__(
        self,
        iterations: int = 2,
        damping: float = 1e-2,
        convergence_rmse: float = 0.08,
        temporal_weight: float = 2e-3,
        neutral_weight: float = 2e-5,
        max_velocity: float = 12.0,
        legacy_mapper=None,
    ):
        self.iterations = int(iterations)
        self.damping = float(damping)
        self.convergence_rmse = float(convergence_rmse)
        self.temporal_weight = float(temporal_weight)
        self.neutral_weight = float(neutral_weight)
        self.max_velocity = float(max_velocity)
        self.legacy_mapper = legacy_mapper

        self.reference = landmarks_from_angles(np.zeros(20, dtype=np.float32)).astype(np.float64)
        self.limits = joint_limits()[:20].astype(np.float32)
        self._limits_t = torch.from_numpy(self.limits)
        self._weights = torch.tensor(
            [2.5] * 5 + [1.0] + [1.5] * 14 + [1.0], dtype=torch.float32
        )
        self._coordinate_weights = self._weights.repeat_interleave(3).sqrt()
        self._identity = torch.eye(20, dtype=torch.float32)
        self._palm_width = float(np.linalg.norm(self.reference[8] - self.reference[17]))
        self._last: Optional[PoseEstimate] = None
        self.solve_count = 0

    @property
    def last_estimate(self) -> Optional[PoseEstimate]:
        return self._last

    def _initial_angles(self) -> np.ndarray:
        if self._last is not None and self._last.valid:
            return self._last.joint_angles.copy()
        return np.clip(np.zeros(20, dtype=np.float32), self.limits[:, 0], self.limits[:, 1])

    def _held(self, timestamp: float, raw_angles=None) -> PoseEstimate:
        if self._last is not None and self._last.valid:
            held = PoseEstimate(
                timestamp=float(timestamp),
                joint_angles=self._last.joint_angles.copy(),
                canonical_landmarks=self._last.canonical_landmarks.copy(),
                fitted_landmarks=self._last.fitted_landmarks.copy(),
                rmse=self._last.rmse,
                valid=True,
                converged=False,
                source="held",
                limit_hits=self._last.limit_hits.copy(),
            )
            self._last = held
            return held

        if self.legacy_mapper is not None and raw_angles is not None:
            angles = self.legacy_mapper.map(raw_angles)
            fitted = landmarks_from_angles(angles)
            fallback = PoseEstimate(
                float(timestamp), angles, fitted.copy(), fitted, np.inf, True, False,
                "legacy_mapper", self._limit_hits(angles),
            )
            self._last = fallback
            return fallback

        invalid = PoseEstimate(
            float(timestamp), np.zeros(20, dtype=np.float32), np.empty((0, 3)),
            np.empty((0, 3)), np.inf, False, False, "held",
            np.zeros(20, dtype=bool),
        )
        self._last = invalid
        return invalid

    def _limit_hits(self, angles: np.ndarray) -> np.ndarray:
        tolerance = np.deg2rad(0.25)
        return np.logical_or(
            angles <= self.limits[:, 0] + tolerance,
            angles >= self.limits[:, 1] - tolerance,
        )

    def _apply_velocity_limit(self, angles: np.ndarray, timestamp: float) -> np.ndarray:
        if self._last is None or not self._last.valid:
            return angles
        dt = float(timestamp) - self._last.timestamp
        if dt <= 0.0:
            return self._last.joint_angles.copy()
        delta = self.max_velocity * min(dt, 0.25)
        return np.clip(
            angles,
            self._last.joint_angles - delta,
            self._last.joint_angles + delta,
        )

    def solve(self, landmarks, timestamp: float, handedness: str = "Right", raw_angles=None) -> PoseEstimate:
        timestamp = float(timestamp)
        if self._last is not None and timestamp <= self._last.timestamp:
            return self._last

        canonical = canonicalize_mediapipe_landmarks(landmarks, self.reference, handedness)
        if not canonical.valid:
            return self._held(timestamp, raw_angles)

        self.solve_count += 1
        target = torch.from_numpy(canonical.landmarks.astype(np.float32))
        initial_np = self._initial_angles()
        initial = torch.from_numpy(initial_np.copy())
        q = initial.clone()
        palm_width_sq = self._palm_width ** 2

        for _ in range(max(1, self.iterations)):
            q = q.detach().requires_grad_(True)
            fitted = landmarks_tensor_from_angles(q)
            residual = (target - fitted).reshape(-1)
            jacobian = torch.autograd.functional.jacobian(
                landmarks_tensor_from_angles, q, vectorize=True
            ).reshape(-1, 20)
            weighted_jacobian = jacobian * self._coordinate_weights[:, None]
            weighted_residual = residual * self._coordinate_weights

            regularization = (self.damping + self.temporal_weight) * palm_width_sq
            lhs = weighted_jacobian.T @ weighted_jacobian + regularization * self._identity
            rhs = weighted_jacobian.T @ weighted_residual
            rhs -= self.temporal_weight * palm_width_sq * (q.detach() - initial)
            rhs -= self.neutral_weight * palm_width_sq * q.detach()
            delta = torch.linalg.solve(lhs, rhs)
            with torch.no_grad():
                q = (q + delta).clamp(self._limits_t[:, 0], self._limits_t[:, 1])
                updated = landmarks_tensor_from_angles(q)
                current_rmse = torch.sqrt(torch.mean(torch.sum((updated - target) ** 2, dim=1)))
                current_rmse /= self._palm_width
            if float(current_rmse) <= self.convergence_rmse:
                break

        angles = q.detach().cpu().numpy().astype(np.float32)
        angles = self._apply_velocity_limit(angles, timestamp)
        angles = np.clip(angles, self.limits[:, 0], self.limits[:, 1]).astype(np.float32)
        fitted_np = landmarks_from_angles(angles).astype(np.float64)
        rmse = float(
            np.sqrt(np.mean(np.sum((fitted_np - canonical.landmarks) ** 2, axis=1)))
            / self._palm_width
        )
        finite = bool(np.all(np.isfinite(angles)) and np.isfinite(rmse))
        estimate = PoseEstimate(
            timestamp=timestamp,
            joint_angles=angles,
            canonical_landmarks=canonical.landmarks,
            fitted_landmarks=fitted_np,
            rmse=rmse,
            valid=finite,
            converged=finite and rmse <= self.convergence_rmse,
            source="ik",
            limit_hits=self._limit_hits(angles),
        )
        self._last = estimate
        return estimate


__all__ = ["HandIKSolver", "PoseEstimate"]
