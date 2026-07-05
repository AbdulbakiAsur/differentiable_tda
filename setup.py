import os
import subprocess
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

def check_cuda():
    try:
        nvcc_path = subprocess.check_output(['which', 'nvcc']).decode('utf-8').strip()
        os.environ['CUDA_HOME'] = os.path.dirname(os.path.dirname(nvcc_path))
        return torch.cuda.is_available()
    except Exception:
        return False

use_cuda = check_cuda()

# [CRITICAL FIX]: PyTorch C++11 ABI Dualism and GCC Standards
abi_val = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
abi_flag = f'-D_GLIBCXX_USE_CXX11_ABI={abi_val}'

# We force C++17 to ensure compatibility with PyTorch 2.x and avoid legacy GCC syntax errors
cxx_args = ['-O3', '-g', '-std=c++17', abi_flag]

if use_cuda:
    print("🚀 CUDA detected! Building GPU-accelerated (C++ & CUDA) module...")
    nvcc_args = ['-O3', '-g', '-std=c++17', '--use_fast_math', abi_flag, '-DWITH_CUDA']
    cxx_args.append('-DWITH_CUDA')
    
    ext_modules = [
        CUDAExtension(
            name='differentiable_tda._C',
            sources=[
                'csrc/tda_core.cpp',
                'csrc/tda_kernel.cu',
            ],
            extra_compile_args={
                'cxx': cxx_args, 
                'nvcc': nvcc_args
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
            ],
            extra_compile_args=cxx_args
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