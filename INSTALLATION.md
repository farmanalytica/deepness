# Deepness Plugin Installation Guide

## Prerequisites (in order)

### 1. QGIS 3.44.3 or later
- Download from https://qgis.org/download/
- **Custom install path (optional):** 
  - Choose custom install location: `C:\QGIS\`
  - Or use default: `C:\Program Files\QGIS 3.44.3\`
- Complete installation
- Verify: Open QGIS and confirm it loads

### 2. CUDA + cuDNN (Optional - for GPU acceleration)

**CUDA 12.4 or 13.x:**
- Download from https://developer.nvidia.com/cuda-toolkit
- Install to default location
- Verify: Check `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin` exists

**cuDNN 9.x:**
- Download from https://developer.nvidia.com/cudnn (requires account)
- Extract to `C:\Program Files\NVIDIA\CUDNN\`
- Verify DLLs exist in `C:\Program Files\NVIDIA\CUDNN\v9.17\bin\`

> Without CUDA/cuDNN, plugin works on CPU (slower but functional)

---

## Install Deepness Plugin

### Step 1: Open QGIS Plugin Manager
- QGIS → Plugins → Manage and Install Plugins

### Step 2: Install from ZIP
- Click **"Install from ZIP"** tab
- Select `deepness.zip` file
- Click **"Install Plugin"**
- Wait for installation to complete

### Step 3: Restart QGIS
- Close QGIS completely
- Reopen QGIS

### Step 4: Install Python Packages
- When QGIS starts, a "Packages Installer" dialog appears
- Click **"Install packages"** button
- Wait for installation to complete (takes 2-3 minutes)
- Click **"Test and Close"**

---

## Verify Installation

Open QGIS Python Console and run:
```python
import cv2
import onnxruntime
print("✓ Deepness dependencies installed successfully")
```

---

## GPU Support (Optional)

After CUDA/cuDNN installation, GPU will be auto-detected on next QGIS restart.

Check in QGIS console for:
```
[Deepness GPU Diagnostics] onnxruntime available providers: [...'CUDAExecutionProvider',...]
[SUCCESS] Using CUDAExecutionProvider
```

If still using CPU: reinstall `onnxruntime-gpu`
```python
!pip uninstall onnxruntime-gpu
!pip install onnxruntime-gpu
```

---

## Troubleshooting

### Plugin won't load
- Check QGIS version ≥ 3.44.3
- Verify folder structure: `plugins/deepness/__init__.py` exists
- Check file paths don't have special characters

### CUDA not detected
- Restart QGIS after CUDA install
- Verify CUDA path: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`
- Check cuDNN extracted correctly

### Package installation fails
- Try again in QGIS console:
  ```python
  !pip install --upgrade onnxruntime-gpu opencv-python-headless
  ```

---

## Support
Report issues at: https://github.com/farmanalytica/deepness/issues
