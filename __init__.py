"""Main plugin module - entry point for the plugin."""
import os
import sys

# Auto-detect CUDA/cuDNN paths. ONNX Runtime needs them in PATH for GPU support.
import glob
_cuda_paths = glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin")
_cudnn_paths = glob.glob(r"C:\Program Files\NVIDIA\CUDNN\*\bin\*")

for lib_path, lib_name in [(_cuda_paths, "CUDA"), (_cudnn_paths, "cuDNN")]:
    for path in lib_path:
        if os.path.exists(path):
            current_path = os.environ.get('PATH', '')
            if path not in current_path:
                os.environ['PATH'] = path + os.pathsep + current_path
                print(f"[Deepness] Added to PATH: {path}")

            if sys.version_info >= (3, 8) and hasattr(os, 'add_dll_directory'):
                try:
                    os.add_dll_directory(path)
                except Exception:
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