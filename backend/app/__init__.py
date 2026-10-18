"""
SIH26187 Backend Application Package.
Initializes runtime environment, including CUDA/cuDNN DLL paths on Windows.
"""
from __future__ import annotations

import os
import sys

if sys.platform == "win32":
    try:
        import importlib.util
        torch_spec = importlib.util.find_spec("torch")
        if torch_spec and torch_spec.submodule_search_locations:
            torch_lib = os.path.join(list(torch_spec.submodule_search_locations)[0], "lib")
            if os.path.isdir(torch_lib):
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(torch_lib)
                os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass
