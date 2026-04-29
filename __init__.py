"""Main plugin module - entry point for the plugin."""
import glob
import os
import sys


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
    """Yield candidate CUDA/cuDNN directories on Windows."""
    env_candidates = [
        os.path.join(os.environ.get('CUDA_PATH', ''), 'bin'),
        os.path.join(os.environ.get('CUDA_HOME', ''), 'bin'),
        os.path.join(os.environ.get('CUDNN_PATH', ''), 'bin'),
        os.environ.get('CUDNN_BIN_PATH', ''),
    ]
    glob_candidates = sorted(glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin"), reverse=True)
    glob_candidates += sorted(glob.glob(r"C:\Program Files\NVIDIA\CUDNN\*\bin"), reverse=True)

    yield from _unique_existing_dirs(env_candidates + glob_candidates)


if sys.platform == "win32":
    # Auto-detect CUDA/cuDNN directories. QGIS often starts with an incomplete DLL search path.
    for path in _iter_windows_gpu_lib_paths():
        current_path = os.environ.get('PATH', '')
        if path not in current_path:
            os.environ['PATH'] = path + os.pathsep + current_path
            print(f"[Deepness] Added GPU runtime directory to PATH: {path}")

        if sys.version_info >= (3, 8) and hasattr(os, 'add_dll_directory'):
            try:
                os.add_dll_directory(path)
            except OSError:
                # Keep going so a missing/invalid directory does not break plugin startup.
                pass

# increase limit of pixels (2^30), before importing cv2.
# We are doing it here to make sure it will be done before importing cv2 for the first time
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()


# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """Load Deepness class from file Deepness.
    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    from deepness.dialogs.packages_installer import packages_installer_dialog
    packages_installer_dialog.check_required_packages_and_install_if_necessary(iface=iface)

    from deepness.deepness import Deepness
    return Deepness(iface)
