"""Startup diagnostics for ONNX Runtime GPU support."""
import glob
import importlib
import os
import platform
import sys

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover - Python < 3.8 fallback
    import importlib_metadata


_HAS_PRINTED_STARTUP_DIAGNOSTICS = False


def _print_line(message: str) -> None:
    print(f"[Deepness GPU Diagnostics] {message}")


def _get_distribution_version(package_name: str) -> str:
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return "not installed"
    except Exception as exc:
        return f"unknown ({type(exc).__name__}: {exc})"


def _unique_existing_dirs(paths):
    seen = set()
    for path in paths:
        if not path:
            continue

        normalized_path = os.path.normpath(path)
        if normalized_path in seen or not os.path.isdir(normalized_path):
            continue

        seen.add(normalized_path)
        yield normalized_path


def _iter_windows_gpu_lib_paths():
    cuda_env_candidates = [
        os.path.join(os.environ.get('CUDA_PATH', ''), 'bin'),
        os.path.join(os.environ.get('CUDA_HOME', ''), 'bin'),
    ]
    cudnn_env_candidates = [
        os.path.join(os.environ.get('CUDNN_PATH', ''), 'bin'),
        os.environ.get('CUDNN_BIN_PATH', ''),
    ]

    cuda_glob_candidates = []
    if not any(os.path.isdir(path) for path in cuda_env_candidates):
        cuda_glob_candidates = sorted(glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin"), reverse=True)

    cudnn_glob_candidates = []
    for dll_path in glob.glob(r"C:\Program Files\NVIDIA\CUDNN\**\cudnn*.dll", recursive=True):
        cudnn_glob_candidates.append(os.path.dirname(dll_path))

    env_candidates = [
        *cuda_env_candidates,
        *cudnn_env_candidates,
    ]
    glob_candidates = cuda_glob_candidates + sorted(cudnn_glob_candidates, reverse=True)

    yield from _unique_existing_dirs(env_candidates + glob_candidates)


def _find_windows_gpu_dlls(directory: str):
    dll_patterns = [
        "cudart64*.dll",
        "cublas64*.dll",
        "cublasLt64*.dll",
        "cudnn*.dll",
        "zlibwapi.dll",
    ]

    found = {}
    for pattern in dll_patterns:
        found[pattern] = sorted(os.path.basename(path) for path in glob.glob(os.path.join(directory, pattern)))
    return found


def print_startup_gpu_diagnostics() -> None:
    """Print ONNX Runtime/CUDA diagnostics once per plugin load."""
    global _HAS_PRINTED_STARTUP_DIAGNOSTICS
    if _HAS_PRINTED_STARTUP_DIAGNOSTICS:
        return

    _HAS_PRINTED_STARTUP_DIAGNOSTICS = True

    _print_line("begin")
    _print_line(f"platform: {platform.platform()}")
    _print_line(f"python: {sys.version.split()[0]} ({sys.executable})")
    onnxruntime_version = _get_distribution_version('onnxruntime')
    onnxruntime_gpu_version = _get_distribution_version('onnxruntime-gpu')
    _print_line(f"onnxruntime package: {onnxruntime_version}")
    _print_line(f"onnxruntime-gpu package: {onnxruntime_gpu_version}")
    if onnxruntime_version != "not installed" and onnxruntime_gpu_version != "not installed":
        if onnxruntime_version != onnxruntime_gpu_version:
            _print_line("warning: both onnxruntime and onnxruntime-gpu are installed with different versions")
            _print_line("warning: uninstall the CPU package or align package versions to avoid ambiguous imports")
        else:
            _print_line("warning: both onnxruntime and onnxruntime-gpu are installed")
    _print_line(f"CUDA_PATH: {os.environ.get('CUDA_PATH', '<not set>')}")
    _print_line(f"CUDA_HOME: {os.environ.get('CUDA_HOME', '<not set>')}")
    _print_line(f"CUDNN_PATH: {os.environ.get('CUDNN_PATH', '<not set>')}")
    _print_line(f"CUDNN_BIN_PATH: {os.environ.get('CUDNN_BIN_PATH', '<not set>')}")

    try:
        ort = importlib.import_module('onnxruntime')
        _print_line(f"onnxruntime import version: {getattr(ort, '__version__', '<unknown>')}")
        _print_line(f"onnxruntime available providers: {ort.get_available_providers()}")
        _print_line(f"onnxruntime preload_dlls available: {callable(getattr(ort, 'preload_dlls', None))}")
        if not callable(getattr(ort, 'preload_dlls', None)):
            _print_line("info: onnxruntime.preload_dlls is available in newer ONNX Runtime releases")
    except Exception as exc:
        _print_line(f"onnxruntime import failed: {type(exc).__name__}: {exc}")

    if sys.platform == "win32":
        gpu_lib_paths = list(_iter_windows_gpu_lib_paths())
        if gpu_lib_paths:
            _print_line(f"discovered GPU runtime directories: {gpu_lib_paths}")
        else:
            _print_line("discovered GPU runtime directories: none")

        for directory in gpu_lib_paths:
            dlls = _find_windows_gpu_dlls(directory)
            for pattern, file_names in dlls.items():
                visible_files = file_names[:5]
                suffix = "" if len(file_names) <= 5 else f" (+{len(file_names) - 5} more)"
                _print_line(f"{directory} | {pattern}: {visible_files or 'not found'}{suffix}")

        has_cudnn = any(
            _find_windows_gpu_dlls(directory)["cudnn*.dll"]
            for directory in gpu_lib_paths
        )
        if not has_cudnn:
            _print_line("warning: no cudnn*.dll found in discovered GPU runtime directories")

    _print_line("end")
