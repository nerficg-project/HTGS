import os
from glob import glob
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

__author__ = 'Florian Hahlbohm'
__description__ = 'Provides various CUDA-accelerated functionality for the HTGS method.'

ENABLE_NVCC_LINEINFO = False  # set to True for profiling kernels with Nsight Compute (overhead is minimal)

module_root = Path(__file__).parent.absolute()
extension_name = module_root.name
extension_root = module_root / extension_name
cuda_modules = [d.name for d in Path(extension_root).iterdir() if d.is_dir() and d.name not in ['utils', 'torch_bindings']]

# gather source files
all_sources = []
for module in cuda_modules:
    all_sources += glob(str(extension_root / module / 'src' / '**'/ '*.cpp'), recursive=True)
    all_sources += glob(str(extension_root / module / 'src' / '**' / '*.cu'), recursive=True)

base_sources = [str(extension_root / 'torch_bindings' / 'bindings.cpp')]
fast_inference_sources = [
    str(extension_root / 'torch_bindings' / 'bindings_benchmarking.cpp'),
    str(extension_root / 'rasterization' / 'src' / 'shared_kernels.cu')
]
for src in all_sources:
    if 'fast' in Path(src).name:
        fast_inference_sources.append(src)
    else:
        base_sources.append(src)

# gather include directories
include_dirs = [str(extension_root / 'utils')]
for module in cuda_modules:
    include_dirs.append(str(extension_root / module / 'include'))

# set up compiler flags
cxx_flags = ['/std:c++17' if os.name == 'nt' else '-std=c++17']
nvcc_flags = ['-std=c++17']
if ENABLE_NVCC_LINEINFO:
    nvcc_flags.append('-lineinfo')

benchmark_cxx_flags = ['-O3']
benchmark_nvcc_flags = ['-O3', '-use_fast_math']

# define the CUDA extensions
base_extension = CUDAExtension(
    name=f'{extension_name}._C',
    sources=base_sources,
    include_dirs=include_dirs,
    extra_compile_args={
        'cxx': cxx_flags,
        'nvcc': nvcc_flags
    }
)
fast_inference_extension = CUDAExtension(
    name=f'{extension_name}._C_benchmarking',
    sources=fast_inference_sources,
    include_dirs=include_dirs,
    extra_compile_args={
        'cxx': cxx_flags + benchmark_cxx_flags,
        'nvcc': nvcc_flags + benchmark_nvcc_flags
    }
)

# set up the package
setup(
    name=extension_name,
    author=__author__,
    packages=[f'{extension_name}.torch_bindings'],
    ext_modules=[base_extension, fast_inference_extension],
    description=__description__,
    cmdclass={'build_ext': BuildExtension}
)
