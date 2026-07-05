import os
import subprocess
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

# Helper function to dynamically check for CUDA and nvcc presence in the environment
def check_cuda():
    try:
        nvcc_path = subprocess.check_output(['which', 'nvcc']).decode('utf-8').strip()
        os.environ['CUDA_HOME'] = os.path.dirname(os.path.dirname(nvcc_path))
        # Verify that PyTorch itself was compiled with CUDA support
        return torch.cuda.is_available()
    except Exception:
        return False

use_cuda = check_cuda()

# Dynamically construct the extension modules based on hardware capabilities
if use_cuda:
    print("🚀 CUDA detected! Building GPU-accelerated (C++ & CUDA) module...")
    ext_modules = [
        CUDAExtension(
            name='differentiable_tda._C',
            sources=[
                'csrc/tda_core.cpp',
                'csrc/tda_kernel.cu',
            ],
            # Pass 'WITH_CUDA' macro to C++ code to enable CUDA-specific code paths
            extra_compile_args={
                'cxx': ['-O3', '-g', '-DWITH_CUDA'], 
                'nvcc': ['-O3', '-g', '--use_fast_math', '-DWITH_CUDA']
            }
        )
    ]
else:
    print("⚠️ CUDA not found! Building CPU-only (C++) fallback module...")
    ext_modules = [
        CppExtension(
            name='differentiable_tda._C',
            sources=[
                'csrc/tda_core.cpp',
                # Dropped .cu file compilation since nvcc is unavailable
            ],
            extra_compile_args=['-O3', '-g']
        )
    ]

setup(
    name='differentiable_tda',
    version='0.1.0',
    packages=['differentiable_tda'],
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    }
)