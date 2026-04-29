"""Main plugin module - entry point for the plugin."""
import os
import sys

# CRITICAL FIX: Add CUDA and cuDNN to PATH before any imports
# ONNX Runtime needs CUDA + cuDNN DLLs in PATH to use GPU
_cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin"
_cudnn_bin = r"C:\Program Files\NVIDIA\CUDNN\v9.17\bin\12.9"

for lib_path, lib_name in [(_cuda_bin, "CUDA"), (_cudnn_bin, "cuDNN")]:
    if os.path.exists(lib_path):
        # Method 1: Add to PATH environment variable
        current_path = os.environ.get('PATH', '')
        if lib_path not in current_path:
            os.environ['PATH'] = lib_path + os.pathsep + current_path
            print(f"[Deepness {lib_name} Fix] Added to PATH: {lib_path}")
        
        # Method 2: Use os.add_dll_directory() for Python 3.8+ (more reliable)
        if sys.version_info >= (3, 8) and hasattr(os, 'add_dll_directory'):
            try:
                os.add_dll_directory(lib_path)
                print(f"[Deepness {lib_name} Fix] Added DLL directory: {lib_path}")
            except Exception as e:
                print(f"[Deepness {lib_name} Fix] Warning: Could not add DLL directory: {e}")
    else:
        print(f"[Deepness {lib_name} Fix] WARNING: {lib_name} not found at {lib_path}")

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