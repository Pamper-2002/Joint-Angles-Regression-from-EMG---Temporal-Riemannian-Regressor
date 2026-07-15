# MediaPipe Solutions-Only Import Design

## Problem

The acquisition application uses only the legacy `mp.solutions.hands` API. Importing
`mediapipe` 0.10.14 also imports `mediapipe.tasks.python`, which unconditionally imports
audio tasks, `sounddevice`/PortAudio, and TensorFlow documentation helpers. On this Windows
environment that unused chain takes 12--20 seconds and can block in native initialization;
interrupting it produces a misleading deep traceback before the application starts.

## Decision

Add a repository-local import adapter that temporarily supplies an empty
`mediapipe.tasks.python` module while the MediaPipe top-level package initializes. This keeps
the required `mediapipe.python.solutions` exports but prevents unused Tasks dependencies from
loading. The adapter restores `sys.modules` in `finally`, removes the temporary public
`mediapipe.tasks` attribute, and leaves an already-imported MediaPipe installation untouched.

Do not patch `site-packages`, uninstall TensorFlow, or change MediaPipe itself. Those options
would be environment-specific and would not survive recreation of the virtual environment.

## Integration and failure handling

`rbcx.handtracker.mediapipe` imports MediaPipe through the adapter. If MediaPipe import fails,
the original exception propagates after temporary modules are restored. Code that deliberately
imports MediaPipe Tasks later can still load the real package because the placeholders are not
retained.

## Acceptance

- A fresh subprocess importing the hand tracker does not load `tensorflow` or `sounddevice`.
- `mp.solutions.hands.Hands` can be constructed and closed.
- Fresh hand-tracker import completes within five seconds on the current machine.
- Existing MediaPipe state, IK, renderer, EMG lifecycle, and performance tests remain green.
- A camera smoke test still reports capture and inference frames.
