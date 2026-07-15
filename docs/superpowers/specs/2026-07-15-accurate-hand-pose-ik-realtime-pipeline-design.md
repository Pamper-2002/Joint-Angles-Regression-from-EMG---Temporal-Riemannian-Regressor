# Accurate Hand Pose IK and Realtime Pipeline Design

## 1. Objective

Replace the fragile landmark-to-angle retargeting path with a measurable, anatomically constrained pipeline that:

- renders an open, closed, pinching, abducted, and individually flexed right hand without systematic finger collapse;
- exposes one authoritative 20-DOF UmeTrack pose in radians;
- preserves an explicit legacy fallback for old MediaPipe-angle models and datasets;
- updates each stage only when a new source timestamp arrives;
- shuts down camera, EMG, renderer, and input resources deterministically;
- reaches at least 24 successful right-hand pose updates per second on the current machine with the recommended lightweight MediaPipe profile;
- provides automated round-trip, invariance, failure, lifecycle, and performance tests.

## 2. Current Failure Model

The current path derives unsigned interior angles and signed abduction angles from MediaPipe landmarks, subtracts hand-written anchors, clips the result to UmeTrack limits, and skins a generic hand. This introduces four independent errors:

1. the configured 2D/3D angle source is computed but ignored;
2. four-finger abduction signs and open-hand anchors disagree with UmeTrack rotation directions;
3. MCP and thumb interior angles do not represent the corresponding UmeTrack axes;
4. rendering and EMG sampling repeat work for unchanged MediaPipe frames.

The existing renderer also rebuilds and Loop-subdivides the mesh every main-loop iteration. MediaPipe uses the two-hand, complexity-1 profile even though only one right hand is consumed. EMG shutdown does not stop the underlying tracker or release the board session, and the console input thread can remain blocked.

## 3. Considered Architectures

### 3.1 Correct the scalar angle mapper only

This is small and fast, but cannot reliably solve coupled thumb motion, MCP flexion, camera rotation, or subject-specific geometry. It remains useful only as a compatibility fallback.

### 3.2 Constrained landmark-to-UmeTrack inverse kinematics

This is the selected architecture. MediaPipe world landmarks are converted into a palm-local, scale-normalized coordinate system. A stateful optimizer fits the UmeTrack 20-DOF pose to corresponding landmarks while enforcing model joint limits and temporal continuity. The result is directly consumable by FK, LBS, and a new EMG label schema.

### 3.3 Direct free-form mesh deformation

This can visually follow landmarks but does not preserve bone lengths or anatomical limits and cannot produce stable joint-angle labels. It is rejected.

## 4. Authoritative Pose Contract

`PoseEstimate` is the only pose value passed between tracking, rendering, and new EMG acquisition code. It contains:

- `timestamp: float`: source MediaPipe timestamp, not render-loop time;
- `joint_angles: np.ndarray[20]`: UmeTrack joint order, radians, clipped to model limits;
- `canonical_landmarks: np.ndarray[21, 3]`: target landmarks in the UmeTrack palm frame;
- `fitted_landmarks: np.ndarray[21, 3]`: FK landmarks for diagnostics;
- `rmse: float`: normalized fit error;
- `valid: bool` and `converged: bool`;
- `source: "ik" | "legacy_mapper" | "held"`;
- `limit_hits: np.ndarray[20]`.

The 20 joint names remain:

1. thumb CMC flexion, thumb CMC abduction, thumb MCP flexion, thumb IP flexion;
2. for index, middle, ring, and pinky: MCP abduction, MCP flexion, PIP flexion, DIP flexion.

New persisted labels use radians and schema name `umetrack20_rad_v1`. Legacy 20-value MediaPipe geometry labels use `mediapipe_geometry20_deg_v1`. A schema metadata file is mandatory when saving new data. Unknown or missing schemas are never silently interpreted as new labels.

## 5. Palm-Local Geometry

The canonical frame is computed from world landmarks:

- origin: wrist landmark;
- longitudinal `+X`: wrist to middle MCP;
- lateral `+Z`: pinky MCP to index MCP, orthogonalized against `+X`;
- normal `+Y`: `Z × X`, with handedness validation;
- scale: robust ratio between UmeTrack and MediaPipe palm widths plus wrist-to-middle-MCP lengths.

The target is rotated and scaled into the fixed UmeTrack model frame. Degenerate axes, non-finite values, a palm width below epsilon, an implausible scale, or an invalid handedness label produce an invalid geometry result rather than a guessed pose.

UmeTrack-to-MediaPipe correspondences cover wrist, five tips, thumb MCP/IP, all four-finger MCP/PIP/DIP landmarks, and a computed palm center. Target weights emphasize tips and PIP/DIP joints while keeping the wrist and palm center stable.

## 6. Stateful Constrained IK

The solver uses differentiable UmeTrack FK in PyTorch. Joint variables are parameterized inside their model limits, preventing invalid optimization states. For each new timestamp it minimizes:

1. weighted landmark position error in the canonical frame;
2. weighted bone-direction error to reduce scale sensitivity;
3. a temporal prior to the previous accepted pose;
4. a small neutral-pose prior when tracking begins or confidence is low.

The previous accepted pose initializes the next solve. The solver uses a fixed iteration budget and returns diagnostics even when it does not converge. Acceptance requires finite output, fit RMSE below the configured threshold, and no implausible timestamp regression.

Failure behavior is deterministic:

- duplicate timestamp: return the cached estimate without solving;
- short invalid interval: hold the previous valid pose and mark `source="held"`;
- prolonged invalid interval: remain held and expose invalid status instead of snapping to zero;
- failed initial IK with valid raw angles: use the corrected legacy mapper and mark the source;
- no valid inputs: return no pose and keep the renderer unchanged.

## 7. MediaPipe Worker

The recommended realtime profile is one hand and model complexity zero. The worker:

- captures and processes only the newest available frame;
- publishes angles, normalized landmarks, world landmarks, handedness, and one common timestamp atomically;
- computes raw geometry angles from the selected coordinate source rather than an ignored temporary;
- records `capture_fps`, `inference_fps`, and successful-right-hand `pose_fps` separately;
- creates annotated frames only when preview is enabled;
- does not call OpenCV GUI functions outside the main thread.

The complexity-1 profile remains selectable for accuracy experiments, but is not the realtime default.

## 8. Renderer

The renderer consumes only accepted UmeTrack radians. It never re-solves pose or interprets MediaPipe angles.

It skips FK and geometry updates when the timestamp and joint vector are unchanged. The realtime default uses the original 788-vertex topology with smooth vertex normals, eliminating per-frame Loop subdivision. Optional subdivision remains an explicit quality mode and only runs for new poses. Render FPS and pose FPS are reported separately.

Smoothing is time-based and source-timestamp-based, not main-loop-iteration-based. A fixed wrist view remains available, while canonical wrist orientation is the default diagnostic view so landmark and mesh alignment can be compared.

## 9. EMG Acquisition and Compatibility

New recordings store authoritative UmeTrack radians and write schema metadata next to the arrays. Label samples are appended only when the pose timestamp advances. The EMG sample index remains the synchronization anchor.

Legacy models may still output MediaPipe geometry degrees. Their metadata selects `AngleMapper`; a model declaring `umetrack20_rad_v1` bypasses the mapper. A missing model schema is treated as legacy with an explicit warning.

`EMG_regressor.stop()` stops the tracker, signals the input thread, and releases the MindRove stream/session. The input thread is daemonized and uses an injectable, interruptible command source for tests. Network failures are rate-limited and do not masquerade as MediaPipe failures.

## 10. Diagnostics and Error Handling

Runtime diagnostics include:

- raw MediaPipe landmarks;
- canonical targets and fitted UmeTrack landmarks;
- IK RMSE, convergence, limit hits, and source;
- capture, inference, pose, solve, render, and end-to-end rates;
- stale-frame age and dropped/duplicate frame counts.

Warnings are rate-limited. Exceptions in worker threads are captured and raised on the main thread. Shutdown is idempotent.

## 11. Test Strategy

### Unit tests

- canonicalization is invariant to translation, scale, and 3D rigid rotation;
- right-hand lateral direction maps index and pinky to opposite sides without crossing;
- FK-generated neutral and articulated poses round-trip through canonicalization and IK;
- recovered angles obey every UmeTrack limit;
- duplicate timestamps do not invoke a second solve;
- invalid and stale landmarks follow the hold/fallback contract;
- legacy abduction mapping has the correct neutral anchors and motion direction;
- renderer update gating ignores unchanged poses;
- EMG label collection rejects duplicate pose timestamps;
- stop is idempotent and releases tracker and board resources.

### Integration tests

- synthetic FK poses with random legal joint angles recover a median absolute joint error below 5 degrees and fingertip error below 5% of palm width;
- neutral/open pose produces ordered, non-crossing index-to-pinky fingertips;
- noisy landmarks remain finite and temporally bounded;
- main pipeline operates without a physical EMG board when EMG is disabled;
- lightweight MediaPipe and render microbenchmarks report results without being flaky pass/fail tests on CI.

### Hardware acceptance on the current machine

- show open palm, fist, pinch, finger spread, and individual finger flexion for at least five seconds each;
- no persistent finger crossing or open-hand DIP/PIP flexion above 15 degrees without corresponding landmark evidence;
- full preview plus mesh achieves at least 24 successful pose updates per second in the lightweight profile after warm-up;
- closing either window or pressing Ctrl+C leaves no MediaPipe, console-input, or board thread alive;
- disconnected MindRove produces a clear, rate-limited error and does not degrade standalone visual tracking.

## 12. Scope Boundaries

This work does not personalize the UmeTrack mesh shape, retrain an EMG model, replace MediaPipe with another detector, or add left-hand rendering. It establishes a correct and testable right-hand pose foundation so those changes can be added later without altering the pose contract.

