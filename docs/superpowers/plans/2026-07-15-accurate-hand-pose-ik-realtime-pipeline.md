# Accurate Hand Pose IK and Realtime Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the unstable scalar retargeting path with a tested palm-local UmeTrack IK pose pipeline, remove avoidable realtime work, and make EMG/camera shutdown and label schemas deterministic.

**Architecture:** MediaPipe publishes one atomic timestamped observation. `pose_geometry.py` maps that observation into the neutral UmeTrack coordinate frame, and `hand_ik.py` fits a joint-limited 20-DOF pose only once per new timestamp. Rendering and new EMG acquisition consume the resulting UmeTrack radians; the corrected scalar mapper remains an explicit legacy fallback.

**Tech Stack:** Python 3.12, NumPy, PyTorch, MediaPipe 0.10.14, OpenCV, Open3D, pytest, MindRove SDK.

---

## File map

- Create `acquisitionFramework/rbcx/handmodel/pose_geometry.py`: landmark validation, palm frames, scale alignment, and MediaPipe/UmeTrack correspondence.
- Create `acquisitionFramework/rbcx/handmodel/hand_ik.py`: `PoseEstimate`, stateful constrained optimizer, stale/duplicate/fallback policy.
- Modify `acquisitionFramework/rbcx/handmodel/fk.py`: expose differentiable landmark FK without changing the existing NumPy API.
- Modify `acquisitionFramework/rbcx/handmodel/angle_map.py`: correct legacy AA neutral anchors and direction.
- Modify `acquisitionFramework/rbcx/handtracker/mediapipe.py`: atomic observations, correct angle source, one-hand realtime defaults, conditional annotation, distinct FPS counters.
- Modify `acquisitionFramework/rbcx/handtracker/hand_mesh_o3d.py`: timestamp-gated updates, time-based smoothing, realtime topology default.
- Modify `acquisitionFramework/main_o3d.py`: orchestrate observations, IK, rendering, and optional EMG without duplicate work.
- Modify `acquisitionFramework/EMG.py`: idempotent stream/session release.
- Modify `acquisitionFramework/EMG_regression.py`: timestamp-deduplicated labels, schema metadata, tracker/board lifecycle.
- Modify `acquisitionFramework/config.py`: explicit pose, MediaPipe, renderer, and EMG compatibility settings.
- Modify `acquisitionFramework/requirements.txt`: add pytest for reproducible local validation.
- Create `acquisitionFramework/tests/`: focused unit and integration tests.
- Modify `acquisitionFramework/README_采集_中文.md`: authoritative schema, launch, diagnostics, and acceptance instructions.

### Task 1: Palm-local landmark geometry

**Files:**
- Create: `acquisitionFramework/tests/conftest.py`
- Create: `acquisitionFramework/tests/test_pose_geometry.py`
- Create: `acquisitionFramework/rbcx/handmodel/pose_geometry.py`

- [ ] **Step 1: Add the failing rigid-transform and invalid-input tests**

```python
def test_canonicalization_is_rigid_transform_and_scale_invariant():
    reference = landmarks_from_angles(np.zeros(20))
    observed = media_pipe_layout_from_umetrack(reference)
    transformed = observed @ random_rotation(7).T * 0.002 + [0.3, -0.2, 1.1]
    result = canonicalize_mediapipe_landmarks(transformed, reference, "Right")
    assert result.valid
    np.testing.assert_allclose(result.landmarks, reference, atol=1e-4)

def test_canonicalization_rejects_collapsed_palm():
    result = canonicalize_mediapipe_landmarks(np.zeros((21, 3)), reference, "Right")
    assert not result.valid
    assert result.reason == "degenerate_palm"
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `..\.venv-emg\Scripts\python.exe -m pytest acquisitionFramework/tests/test_pose_geometry.py -q`

Expected: collection fails because `rbcx.handmodel.pose_geometry` does not exist.

- [ ] **Step 3: Implement the canonical geometry API**

```python
@dataclass(frozen=True)
class CanonicalHand:
    landmarks: np.ndarray
    valid: bool
    scale: float
    reason: str = ""

def canonicalize_mediapipe_landmarks(mp_landmarks, reference_landmarks, handedness="Right"):
    points = np.asarray(mp_landmarks, dtype=np.float64)
    if points.shape != (21, 3) or not np.all(np.isfinite(points)):
        return CanonicalHand(np.empty((0, 3)), False, 0.0, "invalid_shape")
    observed_basis = palm_basis(points[0], points[5], points[9], points[17])
    reference_basis = palm_basis(reference_landmarks[5], reference_landmarks[8], reference_landmarks[11], reference_landmarks[17])
    if observed_basis is None or reference_basis is None:
        return CanonicalHand(np.empty((0, 3)), False, 0.0, "degenerate_palm")
    scale = robust_palm_scale(points, reference_landmarks)
    target = map_corresponding_points(points, observed_basis, reference_basis, scale, reference_landmarks[5])
    return CanonicalHand(target, True, scale)
```

- [ ] **Step 4: Run geometry tests and full collection**

Run: `..\.venv-emg\Scripts\python.exe -m pytest acquisitionFramework/tests/test_pose_geometry.py -q`

Expected: all geometry tests pass.

- [ ] **Step 5: Commit geometry**

```bash
git add acquisitionFramework/rbcx/handmodel/pose_geometry.py acquisitionFramework/tests
git commit -m "feat: add palm-local hand geometry"
```

### Task 2: Differentiable FK and constrained IK

**Files:**
- Create: `acquisitionFramework/tests/test_hand_ik.py`
- Modify: `acquisitionFramework/rbcx/handmodel/fk.py`
- Modify: `acquisitionFramework/rbcx/handmodel/__init__.py`
- Create: `acquisitionFramework/rbcx/handmodel/hand_ik.py`

- [ ] **Step 1: Add failing FK round-trip, limit, and timestamp tests**

```python
def test_fk_pose_round_trip_recovers_articulation():
    expected = legal_test_pose()
    mp = transformed_mediapipe_observation(expected, rotation_seed=11)
    estimate = HandIKSolver(iterations=60).solve(mp, timestamp=1.0, handedness="Right")
    assert estimate.valid and estimate.converged
    np.testing.assert_allclose(estimate.joint_angles, expected, atol=np.deg2rad(5))

def test_duplicate_timestamp_returns_cached_estimate():
    solver = HandIKSolver()
    first = solver.solve(open_observation(), 1.0, "Right")
    second = solver.solve(closed_observation(), 1.0, "Right")
    assert second is first
    assert solver.solve_count == 1

def test_solution_never_exceeds_model_limits():
    estimate = HandIKSolver().solve(noisy_observation(), 2.0, "Right")
    limits = joint_limits()[:20]
    assert np.all(estimate.joint_angles >= limits[:, 0])
    assert np.all(estimate.joint_angles <= limits[:, 1])
```

- [ ] **Step 2: Run tests and verify RED**

Run: `..\.venv-emg\Scripts\python.exe -m pytest acquisitionFramework/tests/test_hand_ik.py -q`

Expected: import failure for `HandIKSolver` and differentiable FK.

- [ ] **Step 3: Add differentiable FK**

```python
def landmarks_tensor_from_angles(joint_angles_20: torch.Tensor) -> torch.Tensor:
    hm = load_hand_model()
    ja = joint_angles_20.reshape(-1)
    pad = torch.zeros(2, dtype=ja.dtype, device=ja.device)
    wrist = torch.eye(4, dtype=ja.dtype, device=ja.device)
    return skin_landmarks(hm, torch.cat((ja[:20], pad)), wrist)
```

- [ ] **Step 4: Implement `PoseEstimate` and bounded stateful optimization**

```python
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
    def solve(self, landmarks, timestamp, handedness="Right", raw_angles=None):
        if self._last is not None and timestamp <= self._last.timestamp:
            return self._last
        target = canonicalize_mediapipe_landmarks(landmarks, self.reference, handedness)
        if not target.valid:
            return self._held(timestamp, target.reason, raw_angles)
        q = torch.tensor(self._initial_angles(), dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([q], lr=self.learning_rate)
        for _ in range(self.iterations):
            optimizer.zero_grad()
            fitted = landmarks_tensor_from_angles(q)
            loss = self._objective(fitted, target.landmarks, q)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                q.clamp_(self.limits[:, 0], self.limits[:, 1])
        return self._accept(timestamp, q, target.landmarks)
```

- [ ] **Step 5: Run focused tests, tune only documented solver constants, and verify GREEN**

Run: `..\.venv-emg\Scripts\python.exe -m pytest acquisitionFramework/tests/test_hand_ik.py -q`

Expected: round-trip median joint error is below 5 degrees, all limit and timestamp tests pass.

- [ ] **Step 6: Commit IK**

```bash
git add acquisitionFramework/rbcx/handmodel acquisitionFramework/tests/test_hand_ik.py
git commit -m "feat: fit constrained UmeTrack hand pose"
```

### Task 3: Correct and constrain the legacy mapper

**Files:**
- Create: `acquisitionFramework/tests/test_angle_map.py`
- Modify: `acquisitionFramework/rbcx/handmodel/angle_map.py`
- Modify: `acquisitionFramework/calibrate_angle_map.py`

- [ ] **Step 1: Add failing open-hand and abduction-direction tests**

```python
def test_ideal_open_hand_does_not_collapse_four_fingers():
    mapped = AngleMapper().map(ideal_open_geometry_angles())
    tips = landmarks_from_angles(mapped)[:5]
    assert np.all(np.diff(tips[1:, 2]) < -5.0)

def test_more_index_spread_moves_index_away_from_middle():
    mapper = AngleMapper()
    neutral = mapper.map(neutral_angles())
    spread = mapper.map(index_spread_angles())
    neutral_lm = landmarks_from_angles(neutral)
    spread_lm = landmarks_from_angles(spread)
    assert spread_lm[1, 2] > neutral_lm[1, 2]
```

- [ ] **Step 2: Verify RED**

Run: `..\.venv-emg\Scripts\python.exe -m pytest acquisitionFramework/tests/test_angle_map.py -q`

Expected: current mapper collapses or reverses the four-finger spread.

- [ ] **Step 3: Correct neutral anchors and AA direction**

```python
_AA_OPEN = {
    "THUMB_CMC_AA": -25.0,
    "INDEX_MCP_AA": -18.0,
    "MIDDLE_MCP_AA": -6.0,
    "RING_MCP_AA": 6.0,
    "PINKY_MCP_AA": 18.0,
}

self.aa_sign = -np.ones(NUM_JOINTS, dtype=np.float64)
```

Update calibration to estimate neutral AA only from explicit open-hand samples and align MediaPipe directions into the UmeTrack palm frame before residual calculation.

- [ ] **Step 4: Verify mapper tests and IK regression tests**

Run: `..\.venv-emg\Scripts\python.exe -m pytest acquisitionFramework/tests/test_angle_map.py acquisitionFramework/tests/test_hand_ik.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit legacy fallback**

```bash
git add acquisitionFramework/rbcx/handmodel/angle_map.py acquisitionFramework/calibrate_angle_map.py acquisitionFramework/tests/test_angle_map.py
git commit -m "fix: correct legacy hand angle retargeting"
```

### Task 4: Atomic MediaPipe observations and truthful FPS

**Files:**
- Create: `acquisitionFramework/tests/test_mediapipe_state.py`
- Modify: `acquisitionFramework/rbcx/handtracker/mediapipe.py`

- [ ] **Step 1: Add failing state and selected-coordinate tests**

```python
def test_hand_state_is_one_atomic_observation():
    tracker = tracker_without_camera()
    tracker._publish_observation(sample_observation(timestamp=3.0))
    state = tracker.get_hand_state("Right")
    assert state["timestamp"] == 3.0
    np.testing.assert_array_equal(state["world_landmarks"], sample_world())
    np.testing.assert_array_equal(state["normalized_landmarks"], sample_normalized())

def test_selected_angle_points_are_used():
    tracker = tracker_without_camera(use_2D_coord_for_angles=True)
    angles = tracker._angles_for_observation(sample_normalized(), sample_world())
    assert angles == expected_from(sample_normalized())
```

- [ ] **Step 2: Verify RED**

Run: `..\.venv-emg\Scripts\python.exe -m pytest acquisitionFramework/tests/test_mediapipe_state.py -q`

Expected: helper APIs and atomic landmark fields are missing.

- [ ] **Step 3: Implement atomic state and realtime defaults**

Set `max_num_hands=1`, `model_complexity=0`, retain explicit overrides, publish normalized/world data under one lock, move annotation behind `show_window`, and update `capture_fps`, `inference_fps`, and `pose_fps` at their true boundaries. Use the selected `angle_pts` in `_compute_joint_angles`.

- [ ] **Step 4: Verify tests**

Run: `..\.venv-emg\Scripts\python.exe -m pytest acquisitionFramework/tests/test_mediapipe_state.py -q`

Expected: all state tests pass without opening a camera.

- [ ] **Step 5: Commit tracker**

```bash
git add acquisitionFramework/rbcx/handtracker/mediapipe.py acquisitionFramework/tests/test_mediapipe_state.py
git commit -m "perf: publish atomic realtime hand observations"
```

### Task 5: Timestamp-gated renderer and main pipeline

**Files:**
- Create: `acquisitionFramework/tests/test_pose_pipeline.py`
- Modify: `acquisitionFramework/rbcx/handtracker/hand_mesh_o3d.py`
- Modify: `acquisitionFramework/main_o3d.py`
- Modify: `acquisitionFramework/config.py`

- [ ] **Step 1: Add failing duplicate-update pipeline tests**

```python
def test_pipeline_solves_and_renders_once_per_source_timestamp():
    pipeline = PosePipeline(fake_solver(), fake_renderer())
    pipeline.process(observation(timestamp=5.0))
    pipeline.process(observation(timestamp=5.0))
    assert pipeline.solver.solve_count == 1
    assert pipeline.renderer.update_count == 1
```

- [ ] **Step 2: Verify RED**

Run: `..\.venv-emg\Scripts\python.exe -m pytest acquisitionFramework/tests/test_pose_pipeline.py -q`

Expected: `PosePipeline` and timestamp-aware renderer update are missing.

- [ ] **Step 3: Implement orchestration and render gating**

```python
class PosePipeline:
    def process(self, state):
        timestamp = float(state["timestamp"])
        if timestamp <= self.last_timestamp:
            return self.last_estimate
        estimate = self.solver.solve(
            state["world_landmarks"], timestamp, state["handedness"], state["angles_list"]
        )
        if estimate.valid:
            self.renderer.update(estimate.joint_angles, timestamp=timestamp)
        self.last_timestamp = timestamp
        self.last_estimate = estimate
        return estimate
```

Make the renderer realtime default `subdivide=0`; update optional quality mode only for new timestamps; smooth using elapsed source time.

- [ ] **Step 4: Verify pipeline and geometry suites**

Run: `..\.venv-emg\Scripts\python.exe -m pytest acquisitionFramework/tests/test_pose_pipeline.py acquisitionFramework/tests/test_hand_ik.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit pipeline**

```bash
git add acquisitionFramework/main_o3d.py acquisitionFramework/config.py acquisitionFramework/rbcx/handtracker/hand_mesh_o3d.py acquisitionFramework/tests/test_pose_pipeline.py
git commit -m "refactor: drive rendering from timestamped IK poses"
```

### Task 6: EMG schema, deduplication, and lifecycle

**Files:**
- Create: `acquisitionFramework/tests/test_emg_lifecycle.py`
- Modify: `acquisitionFramework/EMG.py`
- Modify: `acquisitionFramework/EMG_regression.py`
- Modify: `acquisitionFramework/numpyToFif.py`

- [ ] **Step 1: Add failing deduplication and stop tests**

```python
def test_duplicate_pose_timestamp_is_not_saved_twice():
    regressor = regressor_with_fakes()
    regressor.record_pose(pose(timestamp=7.0))
    regressor.record_pose(pose(timestamp=7.0))
    assert len(regressor.label) == 1

def test_stop_is_idempotent_and_releases_dependencies():
    regressor = regressor_with_fakes()
    regressor.stop()
    regressor.stop()
    assert regressor.handTracker.stop_calls == 1
    assert regressor.mindrove.stop_calls == 1
```

- [ ] **Step 2: Verify RED**

Run: `..\.venv-emg\Scripts\python.exe -m pytest acquisitionFramework/tests/test_emg_lifecycle.py -q`

Expected: timestamp record and deterministic lifecycle APIs are missing.

- [ ] **Step 3: Implement schema and lifecycle**

Persist `savedData/label_schema.json` with `{"schema": "umetrack20_rad_v1", "units": "radians", "joint_names": [...]}`. Append labels only for advancing timestamps. Add `MindRoveEMG.stop()` that calls `stop_stream()` and `release_session()` once. Make the console input thread daemonized and ensure `EMG_regressor.stop()` delegates once to tracker and board.

- [ ] **Step 4: Verify lifecycle and full non-hardware suite**

Run: `..\.venv-emg\Scripts\python.exe -m pytest acquisitionFramework/tests -q`

Expected: all tests pass without camera or MindRove hardware.

- [ ] **Step 5: Commit EMG reliability**

```bash
git add acquisitionFramework/EMG.py acquisitionFramework/EMG_regression.py acquisitionFramework/numpyToFif.py acquisitionFramework/tests/test_emg_lifecycle.py
git commit -m "fix: version pose labels and release acquisition resources"
```

### Task 7: Documentation and reproducible dependencies

**Files:**
- Modify: `acquisitionFramework/requirements.txt`
- Modify: `acquisitionFramework/README_采集_中文.md`

- [ ] **Step 1: Add pytest and document the authoritative paths**

Document the exact launch command, `umetrack20_rad_v1`, legacy compatibility, realtime/quality profiles, printed diagnostics, five-pose acceptance sequence, network timeout meaning, and clean shutdown behavior. Add `pytest` to requirements.

- [ ] **Step 2: Validate commands and documentation references**

Run: `rg -n "umetrack20_rad_v1|model_complexity|pytest|24.*fps|MindRove" acquisitionFramework/README_采集_中文.md acquisitionFramework/requirements.txt`

Expected: every required concept is present and paths match the implementation.

- [ ] **Step 3: Commit documentation**

```bash
git add acquisitionFramework/README_采集_中文.md acquisitionFramework/requirements.txt
git commit -m "docs: document accurate pose acquisition pipeline"
```

### Task 8: Verification, benchmarks, and hardware smoke test

**Files:**
- Create: `acquisitionFramework/tests/test_pose_performance.py`

- [ ] **Step 1: Add non-flaky benchmark reporting tests**

```python
def test_ik_benchmark_reports_finite_latency(benchmark_observation, capsys):
    solver = HandIKSolver(iterations=12)
    milliseconds = median_solve_time(solver, benchmark_observation, repeats=10)
    assert np.isfinite(milliseconds)
    print(f"IK median: {milliseconds:.2f} ms")
```

- [ ] **Step 2: Run formatting, compile, and complete tests**

Run:

```powershell
..\.venv-emg\Scripts\python.exe -m compileall -q acquisitionFramework
..\.venv-emg\Scripts\python.exe -m pytest acquisitionFramework/tests -q
git diff --check
```

Expected: compile succeeds, all tests pass, and `git diff --check` is silent.

- [ ] **Step 3: Run isolated performance probes**

Run the MediaPipe lightweight camera probe for at least six seconds, the IK benchmark for ten valid observations, and the renderer FK/topology benchmark. Record capture, inference, pose, solve, and render rates in the final report.

Expected on the current machine after warm-up: successful pose rate at least 24 fps in the lightweight profile and no solve/render work for duplicate timestamps.

- [ ] **Step 4: Run the five-pose visual protocol when the camera is available**

Verify open palm, fist, pinch, spread, and individual flexion. Confirm ordered index-to-pinky tips for open/spread poses, no unsupported DIP/PIP bend above 15 degrees, stable tracking for five seconds per pose, and clean exit.

- [ ] **Step 5: Final repository review and commit**

```bash
git status --short --branch
git diff --stat HEAD~7..HEAD
git log --oneline -10
git add acquisitionFramework/tests/test_pose_performance.py
git commit -m "test: verify pose accuracy and realtime performance"
```

- [ ] **Step 6: Push the completed branch to the `pamper` fork**

Run: `git push pamper main`

Expected: all implementation commits are backed up at `Pamper-2002/Joint-Angles-Regression-from-EMG---Temporal-Riemannian-Regressor`.

