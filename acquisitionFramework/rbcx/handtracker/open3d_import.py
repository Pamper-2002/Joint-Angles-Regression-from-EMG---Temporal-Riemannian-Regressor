"""Import Open3D desktop visualization without optional Plotly/Jupyter UI."""

from __future__ import annotations

import importlib
import sys
import threading
from types import ModuleType


_IMPORT_LOCK = threading.Lock()
_MISSING = object()


def _plotly_not_loaded(*args, **kwargs):
    raise RuntimeError("Plotly rendering is not loaded in the desktop hand visualizer")


def import_open3d_desktop() -> ModuleType:
    """Return Open3D while leaving Dash, Plotly, and IPython unloaded."""
    with _IMPORT_LOCK:
        loaded = sys.modules.get("open3d")
        if loaded is not None:
            return loaded

        module_name = "open3d.visualization.draw_plotly"
        previous = sys.modules.get(module_name, _MISSING)
        placeholder = ModuleType(module_name)
        placeholder.draw_plotly = _plotly_not_loaded
        placeholder.draw_plotly_server = _plotly_not_loaded
        sys.modules[module_name] = placeholder

        try:
            open3d = importlib.import_module("open3d")
        finally:
            if previous is _MISSING:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous
        return open3d


__all__ = ["import_open3d_desktop"]
