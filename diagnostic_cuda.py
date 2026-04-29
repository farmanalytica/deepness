"""
Diagnostic script to check CUDA setup for ONNX Runtime
Run this in the QGIS Python console
"""

import sys
import subprocess

print("=" * 80)
print("CUDA DIAGNOSTIC FOR ONNXRUNTIME")
print("=" * 80)

# 1. Check onnxruntime version and build info
print("\n1. ONNX Runtime Version & Build:")
try:
    import onnxruntime as ort
    print(f"   Version: {ort.__version__}")
    print(f"   Build: {ort.__full_version__}")
except AttributeError:
    print("   Full version info not available")

# 2. Check available providers
print("\n2. Available Providers:")
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    for i, p in enumerate(providers, 1):
        print(f"   {i}. {p}")
except Exception as e:
    print(f"   ERROR: {e}")

# 3. Check for CUDA installations
print("\n3. System CUDA/cuDNN Search:")
import os
import glob

cuda_search_paths = [
    "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\*",
    "C:\\cuda\\*",
    "C:\\CUDA\\*",
    os.environ.get('CUDA_HOME', ''),
    os.environ.get('CUDA_PATH', ''),
]

cuda_found = []
for path_pattern in cuda_search_paths:
    if path_pattern:
        matches = glob.glob(path_pattern)
        cuda_found.extend(matches)

if cuda_found:
    print(f"   Found CUDA installations:")
    for p in cuda_found:
        print(f"   - {p}")
else:
    print("   WARNING: No CUDA installations found in standard paths!")

# 4. Check PATH for CUDA DLLs
print("\n4. CUDA DLLs in PATH:")
path_env = os.environ.get('PATH', '')
cuda_dlls_found = []
for p in path_env.split(';'):
    if 'cuda' in p.lower() or 'nvidia' in p.lower():
        cuda_dlls_found.append(p)

if cuda_dlls_found:
    print(f"   Found CUDA-related PATH entries:")
    for p in cuda_dlls_found:
        print(f"   - {p}")
else:
    print("   WARNING: No CUDA-related entries in PATH!")

# 5. Environment variables
print("\n5. Relevant Environment Variables:")
cuda_env_vars = ['CUDA_HOME', 'CUDA_PATH', 'CUDA_TOOLKIT_ROOT_DIR', 'LD_LIBRARY_PATH', 'PATH']
for var in cuda_env_vars:
    val = os.environ.get(var)
    if var == 'PATH':
        # Just show if CUDA is in PATH, not the full PATH
        cuda_in_path = any('cuda' in p.lower() for p in val.split(';'))
        print(f"   {var}: CUDA paths present = {cuda_in_path}")
    else:
        if val:
            print(f"   {var}: {val}")
        else:
            print(f"   {var}: NOT SET")

# 6. Try to import CUDA libraries
print("\n6. CUDA Library Import Test:")
try:
    import ctypes
    # Try to load CUDA runtime
    try:
        ctypes.CDLL('cudart64_12.dll')
        print("   ✓ cudart64_12.dll loaded successfully")
    except OSError as e:
        print(f"   ✗ cudart64_12.dll failed: {e}")
    
    # Try other common CUDA DLL names
    for dll in ['cudart64_11.dll', 'cudart64.dll', 'cudart.dll']:
        try:
            ctypes.CDLL(dll)
            print(f"   ✓ {dll} loaded successfully")
            break
        except OSError:
            continue
except Exception as e:
    print(f"   Error testing CUDA libraries: {e}")

# 7. Check onnxruntime package location
print("\n7. ONNX Runtime Package Location:")
try:
    import onnxruntime as ort
    ort_path = ort.__file__
    print(f"   {ort_path}")
    # Check if it's the right package
    parent_dir = os.path.dirname(ort_path)
    if os.path.exists(os.path.join(parent_dir, 'cudnn_version.py')):
        print("   ✓ Has cuDNN support files")
    else:
        print("   ⚠ Missing cuDNN support files (might be CPU-only build)")
except Exception as e:
    print(f"   {e}")

print("\n" + "=" * 80)
print("RECOMMENDATIONS:")
print("=" * 80)
print("""
If CUDA is not working:

1. INSTALL CUDA:
   - Download from: https://developer.nvidia.com/cuda-toolkit
   - Install CUDA Toolkit (e.g., CUDA 12.x)
   - Install cuDNN (https://developer.nvidia.com/cudnn)
   - Add CUDA bin folder to PATH (e.g., C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x\\bin)

2. REINSTALL ONNXRUNTIME WITH GPU SUPPORT:
   - pip uninstall onnxruntime
   - pip install onnxruntime-gpu
   
   OR if you need a specific version:
   - pip install onnxruntime-gpu==1.17.0

3. VERIFY INSTALLATION:
   - Restart QGIS
   - Check console output again
   - Should see CUDAExecutionProvider being used

4. CHECK COMPATIBILITY:
   - CUDA 12.x works with onnxruntime-gpu 1.16+
   - CUDA 11.x works with onnxruntime-gpu 1.15+
""")
