# Architecture & Design Decisions

This document outlines the core engineering challenges faced when implementing differentiable Topological Data Analysis (TDA) for high-dimensional data, and how the custom C++/CUDA backend resolves them.

## 1. The VRAM Bottleneck & Sparse Radius Graphs

**The Problem:** Standard topological formulations (like Vietoris-Rips complexes) require computing pairwise distance matrices. For a point cloud of $N=50,000$, a dense distance matrix requires storing $2.5 \times 10^9$ elements. In Float32, this consumes ~10 GB of VRAM per matrix, leading to immediate Out-of-Memory (OOM) errors.

**The Architecture:**
We implemented a custom CUDA kernel that bypasses dense matrix allocation entirely. 
* **Atomic Counters:** The `radius_graph_forward_kernel` utilizes CUDA's `atomicAdd` to dynamically populate a flat 1D array on the GPU. Threads only write to global memory if the distance $D_{ij}$ is strictly less than the topological radius $R$.
* **Zero-Padding Slice:** The C++ dispatcher uses `.contiguous()` slicing to return an exact, dynamically sized `(2, E)` edge index tensor, dropping VRAM consumption by over 95% for typical high-dimensional financial datasets.

## 2. Gradient Explosions & The "Topological Deadzone"

**The Problem:** The derivative of the Euclidean distance $D_{ij}$ with respect to the input coordinates $x_{ik}$ is given by the chain rule:

$$\frac{\partial L}{\partial x_{ik}} \propto \frac{x_{ik} - x_{jk}}{D_{ij} + \epsilon}$$

In financial datasets (e.g., fraud rings), transactions are often identical ($x_i \approx x_j$), meaning $D_{ij} \to 0$. Even with a small $\epsilon$, this division creates a massive gradient multiplier, cascading into `NaN` (Not a Number) values during the backward pass and destroying model weights.

**The Architecture:**
We introduced a hardware-level **Topological Deadzone**. Inside the `radius_graph_backward_kernel`, before any floating-point division occurs, the thread checks if the distance falls below the `deadzone` threshold (default: $10^{-5}$). If true, the thread returns early, applying **zero gradient force**. Topologically, points that occupy the exact same simplex space do not exert repulsive forces on each other.

## 3. Race Conditions in Sparse Backward Passes

**The Problem:** In a sparse graph, multiple valid edges (e.g., node $A \to B$ and node $A \to C$) share the same source node. If parallel CUDA threads attempt to accumulate gradients for node $A$ simultaneously using standard `+=` operators, memory overwrites (race conditions) occur, silently corrupting the gradients.

**The Architecture:**
Our backward kernel abandons standard grid mapping. Instead, **each thread represents a valid Edge ($E$), not a matrix cell.** We strictly enforce `atomicAdd` on the physical memory addresses of `grad_pc_data[i * D + k]` and `grad_pc_data[j * D + k]` to guarantee deterministic and mathematically precise gradient accumulation across millions of edges.

## 4. Floating-Point Precision vs. PyTorch `cdist`

During benchmarking, we discovered that PyTorch's native `torch.cdist` diverges from pure L2 Euclidean math for large matrices due to its reliance on the `baddbmm` (Matrix Multiplication) expansion:

$$||x_i - x_j||_2^2 = ||x_i||_2^2 + ||x_j||_2^2 - 2\langle x_i, x_j \rangle$$

While fast, this binomial expansion suffers from catastrophic cancellation in Float32. Our kernel uses the naive but numerically stable raw loop formulation, resulting in gradients that are mathematically purer than PyTorch's default behavior.