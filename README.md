# GigaLearnCPP Setup Guide

> [!IMPORTANT]
> **Prerequisites**
> *   **Visual Studio 2022** with "Desktop development with C++" workload
> *   **Windows 10/11** (64-bit)
> *   **Git** installed and available in PATH
> *   **CUDA 12.8** (Recommended for NVIDIA GPUs)

## System Requirements

| | Minimum | Recommended |
|---|---|---|
| **CPU** | Intel i5 / Ryzen 5 | Intel i7/i9 or Ryzen 7/9 |
| **RAM** | 8 GB | 16–32 GB |
| **GPU** | Optional | NVIDIA GPU (6GB+ VRAM) |
| **Storage** | 5 GB SSD | 20 GB+ SSD |

---

## Installation Steps

### 1. Clone Repository
Open your terminal (Command Prompt / PowerShell) and run:
```powershell
cd %USERPROFILE%\Downloads
git clone https://github.com/ramgd1/GigalearnCPP.git --recurse-submodules
```

### 2. Install CUDA (Recommended)
If you have an NVIDIA GPU, this will significantly speed up training.
1.  Download **CUDA 12.8** (or compatible any compatible CUDA Verison(DO THIS IF YOU KNOW WHATA RE YOU DOING)) from the [NVIDIA Archive](https://developer.nvidia.com/cuda-12-8-0-download-archive).
2.  Select **Windows** -> **x86_64** -> **exe (local)**.
3.  Install with default settings.

### 3. Install LibTorch
1.  Visit the [PyTorch Start Locally](https://pytorch.org/get-started/locally/) page.
2.  Select configuration:
    *   **Build**: Stable
    *   **OS**: Windows
    *   **Package**: LibTorch
    *   **Language**: C++
    *   **Compute Platform**: CUDA 12.8 (or CPU if no GPU)
3.  **Download** the zip file.
4.  **Extract** the `libtorch` folder into:
    GigaLearnCPP\GigaLearnCPP\`
    *(Structure should be: `GigaLearnCPP\libtorch\include`, `GigaLearnCPP\libtorch\lib`, etc.)*

### 4. Configure in Visual Studio 2022
1.  Open Visual Studio 2022.
2.  Select **"Open a local folder"**.
3.  Navigate to and select the `GigaLearnCPP` folder.
4.  VS will detect the `CMakeLists.txt` and start configuration.

### 5. Build Configuration
1.  Locate the **Configuration Dropdown** in the top toolbar.
2.  Select **x64-Release** from the dropdown. CMake will regenerate the cache.
(this step should be already done automaticaly)

### 6. Build & Run
1.  Go to **Build** -> **Build All** (or press `Ctrl+Shift+B`).
2.  Once built, your executable will be in:
    `out\build\x64-Release`
---

## Troubleshooting

> [!TIP]
> **CMake Cache Issues**
> If the build fails or settings don't apply, go to **Project** -> **Delete Cache and Reconfigure**.
> If that fails, manually delete the `.vs` and `out` folders and restart Visual Studio.

> [!WARNING]
> **Missing DLLs**
> If the bot crashes immediately, ensure all LibTorch DLLs are accessible. They should be copied automatically during build, but if not, verify your `libtorch` placement in Step 3.

> [!NOTE]
> **Support**
> For help, contact `ramgd` on Discord.
