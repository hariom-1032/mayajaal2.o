"""
SIH26187 Backend Application Package.
Initializes runtime environment, including CUDA/cuDNN DLL paths on Windows.
"""
from __future__ import annotations

import os
import sys

# Automatically configure CUDA and cuDNN DLL paths for Windows
try:
    from app.utils.cuda_setup import setup_cuda_dll_directories
    setup_cuda_dll_directories()
except Exception:
    pass

