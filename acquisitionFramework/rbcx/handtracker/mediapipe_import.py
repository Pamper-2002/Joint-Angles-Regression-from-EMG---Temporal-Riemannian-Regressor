"""Import legacy MediaPipe Solutions without initializing unused Tasks APIs."""

from __future__ import annotations

import importlib
import sys
import threading
from types import ModuleType


_IMPORT_LOCK = threading.Lock()
_MISSING = object()


def import_mediapipe_solutions() -> ModuleType:
    """Return MediaPipe with ``solutions`` while keeping optional Tasks lazy.

    MediaPipe 0.10.14 imports its complete Tasks namespace at package import time.
    The hand tracker does not use Tasks, but that namespace initializes audio,
    PortAudio, and TensorFlow. Temporary modules satisfy only that eager import;
    they are removed immediately so a later explicit Tasks import remains possible.
    """
    with _IMPORT_LOCK:
        loaded = sys.modules.get("mediapipe")
        if loaded is not None:
            return loaded

        tasks_name = "mediapipe.tasks"
        python_name = "mediapipe.tasks.python"
        previous_tasks = sys.modules.get(tasks_name, _MISSING)
        previous_python = sys.modules.get(python_name, _MISSING)

        tasks_module = ModuleType(tasks_name)
        tasks_module.__path__ = []
        python_module = ModuleType(python_name)
        python_module.__package__ = tasks_name
        tasks_module.python = python_module
        sys.modules[tasks_name] = tasks_module
        sys.modules[python_name] = python_module

        try:
            mediapipe = importlib.import_module("mediapipe")
        finally:
            if previous_python is _MISSING:
                sys.modules.pop(python_name, None)
            else:
                sys.modules[python_name] = previous_python
            if previous_tasks is _MISSING:
                sys.modules.pop(tasks_name, None)
            else:
                sys.modules[tasks_name] = previous_tasks

        if getattr(mediapipe, "tasks", None) in (tasks_module, python_module):
            delattr(mediapipe, "tasks")
        return mediapipe


__all__ = ["import_mediapipe_solutions"]
