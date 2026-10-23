import logging
import os
import sys

logger = logging.getLogger("SIH26187.CUDA")

_DLL_DIRECTORIES_CONFIGURED = False
_DLL_HANDLES: list = []


def setup_cuda_dll_directories() -> list[str]:
    """
    Ensure all NVIDIA CUDA and cuDNN runtime DLL directories are registered
    with Windows OS DLL search path via os.add_dll_directory() and PATH.

    This resolves ONNX Runtime and PyTorch CUDAExecutionProvider failures
    on Windows (e.g. cublasLt64_12.dll, cudnn64_9.dll, cudart64_12.dll).
    """
    global _DLL_DIRECTORIES_CONFIGURED, _DLL_HANDLES
    if _DLL_DIRECTORIES_CONFIGURED:
        return []

    if sys.platform != "win32":
        _DLL_DIRECTORIES_CONFIGURED = True
        return []

    added_dirs: list[str] = []

    def _add_dir(path: str) -> None:
        if not path or not os.path.isdir(path):
            return
        norm = os.path.normpath(path)
        if norm in added_dirs:
            return
        try:
            if hasattr(os, "add_dll_directory"):
                _DLL_HANDLES.append(os.add_dll_directory(norm))
            os.environ["PATH"] = norm + os.pathsep + os.environ.get("PATH", "")
            added_dirs.append(norm)
            logger.debug("Added CUDA DLL directory: %s", norm)
        except Exception as exc:
            logger.warning("Could not add DLL directory %s: %s", norm, exc)


    # 1. Search site-packages for nvidia-* packages (cublas, cudnn, cuda_nvrtc, etc.)
    try:
        import site

        search_roots = []
        if hasattr(site, "getsitepackages"):
            search_roots.extend(site.getsitepackages())
        user_site = getattr(site, "getusersitepackages", lambda: None)()
        if user_site:
            search_roots.append(user_site)

        # Also inspect current sys.prefix site-packages
        prefix_site = os.path.join(sys.prefix, "Lib", "site-packages")
        if os.path.isdir(prefix_site) and prefix_site not in search_roots:
            search_roots.append(prefix_site)

        for sdir in search_roots:
            nvidia_root = os.path.join(sdir, "nvidia")
            if os.path.isdir(nvidia_root):
                for root, _, files in os.walk(nvidia_root):
                    if any(f.lower().endswith(".dll") for f in files):
                        _add_dir(root)
    except Exception as exc:
        logger.debug("Error discovering nvidia pip packages: %s", exc)

    # 2. Search CUDA_PATH environment variable and standard CUDA Toolkit paths
    cuda_env = os.environ.get("CUDA_PATH")
    candidate_cuda_paths = []
    if cuda_env:
        candidate_cuda_paths.append(os.path.join(cuda_env, "bin"))
        candidate_cuda_paths.append(os.path.join(cuda_env, "libnvvp"))

    candidate_cuda_paths.extend([
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
        r"C:\inference\cudnn-8.9.2\bin",
        r"C:\cudnn-8.9.2\bin",
    ])

    for cpath in candidate_cuda_paths:
        _add_dir(cpath)

    # 3. Search torch/lib if PyTorch is installed
    try:
        import importlib.util

        torch_spec = importlib.util.find_spec("torch")
        if torch_spec and torch_spec.submodule_search_locations:
            for loc in torch_spec.submodule_search_locations:
                torch_lib = os.path.join(loc, "lib")
                if os.path.isdir(torch_lib):
                    _add_dir(torch_lib)
    except Exception:
        pass

    _DLL_DIRECTORIES_CONFIGURED = True
    if added_dirs:
        logger.info("Configured %d CUDA/cuDNN DLL director(ies) for GPU inference.", len(added_dirs))
    return added_dirs


def get_cuda_device_count() -> int:
    """Return the number of available NVIDIA CUDA devices."""
    if sys.platform == "win32":
        try:
            import ctypes
            cuda = ctypes.windll.nvcuda
            if cuda.cuInit(0) == 0:
                count = ctypes.c_int()
                if cuda.cuDeviceGetCount(ctypes.byref(count)) == 0:
                    return count.value
        except Exception:
            pass
    return 0


def get_cuda_device_name(device_id: int = 0) -> str:
    """Return the name of the NVIDIA CUDA device (e.g. 'NVIDIA T400 4GB')."""
    if sys.platform == "win32":
        try:
            import ctypes
            cuda = ctypes.windll.nvcuda
            if cuda.cuInit(0) == 0:
                dev = ctypes.c_int()
                if cuda.cuDeviceGet(ctypes.byref(dev), device_id) == 0:
                    buf = ctypes.create_string_buffer(256)
                    if cuda.cuDeviceGetName(buf, 256, dev) == 0:
                        return buf.value.decode("utf-8")
        except Exception:
            pass
    return "NVIDIA GPU"


# Automatically execute DLL setup upon import
setup_cuda_dll_directories()

