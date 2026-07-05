# Differentiable TDA (Topological Data Analysis) for PyTorch 🚀

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.0.0-EE4C2C.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

A hardware-accelerated, memory-efficient, and fully differentiable C++/CUDA backend for Topological Data Analysis (TDA) in PyTorch. 

Designed specifically for **high-dimensional financial time-series** and **anomaly detection**, this package replaces PyTorch's native $O(N^2)$ dense distance computations with $O(E)$ Sparse Radius Graphs, achieving up to **25x speedups** while preventing catastrophic gradient explosions.

## 🔥 Key Features

* **Sparse Radius Graphs:** Computes distances only for topologically relevant points ($D_{ij} \le R$), reducing VRAM usage from tens of Gigabytes to mere Megabytes.
* **Topological Deadzone:** A custom hardware-level threshold that drops gradients for identical nodes, completely eliminating `NaN` gradient explosions caused by zero-distance divisions.
* **Batched 3D Execution:** Native support for `(B, N, D)` tensors, seamlessly integrating with modern Deep Learning pipelines (BERT, EfficientNet) and production Triton servers.
* **Zero-Copy Memory:** Bypasses standard ATen overhead by directly accessing contiguous physical memory blocks on the GPU.

## ⚡ Benchmark (Native vs. Custom CUDA)

Tested on a 64-dimensional financial vector space on an RTX-class GPU (Radius = 3.2):

| $N$ Points | PyTorch Time | Custom TDA | Speedup | PyTorch VRAM | Custom VRAM | VRAM Saved |
|-----------:|-------------:|-----------:|--------:|-------------:|------------:|-----------:|
| **1,000** | 24.82 ms     | **16.66 ms**| ~1.5x   | 4.12 MB      | **2.23 MB** | ~45%       |
| **5,000** | 232.88 ms    | **4.69 ms** | **~49x**| 97.28 MB     | **11.20 MB**| ~88%       |
| **10,000** | 166.09 ms    | **17.16 ms**| **~9x** | 384.50 MB    | **23.08 MB**| ~94%       |
| **20,000** | 613.77 ms    | **67.11 ms**| **~9x** | 1,530.95 MB  | **49.25 MB**| ~96%       |
| **40,000** | 2,459.53 ms  | **268.74 ms**| **~9x** | 6,113.83 MB  | **111.84 MB**| **~98%** |

## 🛠️ Installation

Ensure you have a C++ compiler and CUDA Toolkit installed (CUDA 12.x recommended).

**Option 1: Direct Install (Recommended for Users)**
Install directly from GitHub without cloning the repository:
```bash
pip install --upgrade --no-cache-dir git+https://github.com/AbdulbakiAsur/differentiable_tda.git
```

**Option 2: Install from Source (For Development)**
If you want to modify the custom CUDA kernels or contribute to the project:
```bash
git clone https://github.com/AbdulbakiAsur/differentiable_tda.git
cd differentiable_tda
pip install -e .
```

## 🚀 Quick Start
```code
import torch
import differentiable_tda as dtda

# Batch of 4, 10,000 nodes, 64 features
x = torch.rand((4, 10000, 64), dtype=torch.float32, device='cuda', requires_grad=True)

# Returns only edges where distance <= 3.2
indices, values = dtda.radius_graph(x, radius=3.2, max_edges=500000)

loss = values.sum()
loss.backward() # Safe, NaN-free sparse gradients!
```