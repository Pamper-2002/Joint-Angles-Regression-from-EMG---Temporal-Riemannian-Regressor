# MediaPipe Solutions-Only Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Start hand tracking without importing unused MediaPipe Tasks, TensorFlow, sounddevice, or PortAudio.

**Architecture:** A small adapter temporarily registers placeholder MediaPipe Tasks modules only while MediaPipe exposes its legacy Solutions API, then restores module state. The hand tracker imports through that adapter; a fresh-process regression test verifies the actual import boundary.

**Tech Stack:** Python 3.12, MediaPipe 0.10.14, pytest, subprocess, Windows.

## Global Constraints

- Do not modify `.venv-emg` or any file under `site-packages`.
- Preserve `mp.solutions.hands` and later deliberate imports of the real Tasks package.
- A fresh hand-tracker import must not load `tensorflow` or `sounddevice`.

---

### Task 1: Solutions-only import adapter

**Files:**
- Create: `acquisitionFramework/rbcx/handtracker/mediapipe_import.py`
- Modify: `acquisitionFramework/rbcx/handtracker/mediapipe.py:11`
- Create: `acquisitionFramework/tests/test_mediapipe_import.py`

**Interfaces:**
- Produces: `import_mediapipe_solutions() -> ModuleType`
- Consumes: standard-library `importlib`, `sys`, and `types` only.

- [ ] **Step 1: Write the failing fresh-process test**

The subprocess imports `rbcx.handtracker.mediapipe`, constructs `mp.solutions.hands.Hands`, and asserts `tensorflow` and `sounddevice` are absent from `sys.modules`.

- [ ] **Step 2: Verify RED**

Run: `D:\Project_CJ\rgb2pose\.venv-emg\Scripts\python.exe -m pytest acquisitionFramework/tests/test_mediapipe_import.py -q`

Expected: FAIL because the current top-level import loads TensorFlow and sounddevice.

- [ ] **Step 3: Implement minimal state-restoring adapter**

Use temporary `ModuleType("mediapipe.tasks")` and `ModuleType("mediapipe.tasks.python")` entries, restore previous `sys.modules` values in `finally`, and remove only the temporary `mp.tasks` public attribute.

- [ ] **Step 4: Verify GREEN and tracker state regression**

Run: `D:\Project_CJ\rgb2pose\.venv-emg\Scripts\python.exe -m pytest acquisitionFramework/tests/test_mediapipe_import.py acquisitionFramework/tests/test_mediapipe_state.py -q`

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add acquisitionFramework/rbcx/handtracker/mediapipe_import.py acquisitionFramework/rbcx/handtracker/mediapipe.py acquisitionFramework/tests/test_mediapipe_import.py
git commit -m "fix: avoid unused MediaPipe Tasks imports"
```

### Task 2: Startup and regression verification

**Files:**
- Modify: `acquisitionFramework/README_采集_中文.md`

**Interfaces:**
- Consumes: the Task 1 adapter through the normal `main_o3d.py` import path.
- Produces: documented fast-start behavior and diagnostic command.

- [ ] **Step 1: Time a fresh `main_o3d` import**

Run a fresh subprocess and assert it reports completion without loading TensorFlow through MediaPipe.

- [ ] **Step 2: Run camera smoke test**

Start `MediaPipeHandTracker(show_window=False)` for six seconds; require nonzero capture and inference FPS and deterministic `stop()`.

- [ ] **Step 3: Run isolated regression groups**

Run import/state, IK/performance, renderer pipeline, and EMG lifecycle tests in independent processes. Expected total: 24 passing tests.

- [ ] **Step 4: Document and commit**

Document that startup no longer initializes MediaPipe Audio Tasks, plus the import diagnostic command. Commit with `docs: document lightweight MediaPipe startup`.

- [ ] **Step 5: Push verified main**

Run `git push pamper main` and verify the remote `refs/heads/main` hash matches local `HEAD`.
