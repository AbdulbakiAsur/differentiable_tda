import torch
import warnings

# Version Compatibility Check
if torch.cuda.is_available():
    cuda_version = torch.version.cuda
    if cuda_version is None:
        warnings.warn("PyTorch is compiled without CUDA, but a GPU is present. dtda will fallback to CPU.")
    elif "12." not in cuda_version and "11." not in cuda_version:
        warnings.warn(f"dtda is optimized for CUDA 11.x/12.x. Found CUDA {cuda_version}. You may experience instability.")

# Critical: Loads libc10.so and libtorch.so into process memory first
from . import _C
from .autograd import pdist, radius_graph

__all__ = ['_C', 'pdist', 'radius_graph']