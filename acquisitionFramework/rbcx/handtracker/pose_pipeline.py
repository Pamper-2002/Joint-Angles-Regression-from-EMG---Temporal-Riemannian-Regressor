"""Timestamp-gated orchestration between hand observations, IK, and rendering."""

from __future__ import annotations


class PosePipeline:
    def __init__(self, solver, renderer):
        self.solver = solver
        self.renderer = renderer
        self.last_timestamp = float("-inf")
        self.last_estimate = None

    def process(self, state):
        if state is None:
            return self.last_estimate
        timestamp = float(state.get("timestamp", float("-inf")))
        if timestamp <= self.last_timestamp:
            return self.last_estimate

        landmarks = state.get("world_landmarks")
        if landmarks is None:
            landmarks = state.get("landmarks")
        estimate = self.solver.solve(
            landmarks,
            timestamp,
            state.get("handedness", "Right"),
            state.get("angles_list"),
        )
        if estimate.valid:
            self.renderer.update(estimate.joint_angles, timestamp=timestamp)
        self.last_timestamp = timestamp
        self.last_estimate = estimate
        return estimate


__all__ = ["PosePipeline"]
