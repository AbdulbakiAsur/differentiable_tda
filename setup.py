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

# ESKİ GCC VE ABI KORUMASI: PyTorch 2.x kesinlikle C++17 ve doğru ABI bayrağı ister!
abi_version = "1" if torch._C._GLIBCXX_USE_CXX11_ABI else "0"
cxx_args = ['-O3', '-g', '-std=c++17', f'-D_GLIBCXX_USE_CXX11_ABI={abi_version}']

if check_cuda():
    ext_modules = [
        CUDAExtension(
            name='differentiable_tda._C',
            sources=['csrc/tda_core.cpp', 'csrc/tda_kernel.cu'],
            extra_compile_args={
                'cxx': cxx_args + ['-DWITH_CUDA'],
                'nvcc': ['-O3', '-g', '-std=c++17', '--use_fast_math', f'-D_GLIBCXX_USE_CXX11_ABI={abi_version}', '-DWITH_CUDA']
            }
        )
    ]
else:
    ext_modules = [
        CppExtension(
            name='differentiable_tda._C',
            sources=['csrc/tda_core.cpp'],
            extra_compile_args=cxx_args
        )
    ]

setup(
    name='differentiable_tda',
    version='0.1.0',
    packages=['differentiable_tda'],
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)