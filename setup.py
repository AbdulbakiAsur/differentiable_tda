from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import subprocess

# Production-ready CUDA_HOME resolution
try:
    nvcc_path = subprocess.check_output(['which', 'nvcc']).decode('utf-8').strip()
    os.environ['CUDA_HOME'] = os.path.dirname(os.path.dirname(nvcc_path))
except Exception:
    print("WARNING: nvcc not found in PATH. Build might fail if CUDA is not standard.")

cxx_args = ['-O3', '-g']
nvcc_args = ['-O3', '-g', '--use_fast_math']

setup(
    name='differentiable_tda',
    version='0.1.0',
    packages=['differentiable_tda'],
    ext_modules=[
        CUDAExtension(
            name='differentiable_tda._C',
            sources=[
                'csrc/tda_core.cpp',
                'csrc/tda_kernel.cu',
            ],
            extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
