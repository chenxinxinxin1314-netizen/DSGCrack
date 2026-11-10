# Adapted from VMamba / state-spaces/mamba. See ../../third_party/README.md.
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

ROOT = Path(__file__).resolve().parent
if CUDA_HOME is None:
    raise RuntimeError("CUDA toolkit not found. Set CUDA_HOME to your CUDA toolkit directory.")
source = ROOT/"csrc/selective_scan"
setup(
    name="dsgcrack-selective-scan",
    version="0.1.0",
    description="The oflex selective scan kernel used by DSGCrack",
    packages=[],
    ext_modules=[CUDAExtension(
        name="selective_scan_cuda_oflex",
        sources=[str(source/"cusoflex"/p) for p in [
            "selective_scan_oflex.cpp", "selective_scan_core_fwd.cu",
            "selective_scan_core_bwd.cu"]],
        include_dirs=[str(source)],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": ["-O3", "-std=c++17", "--use_fast_math",
                     "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                     "-U__CUDA_NO_BFLOAT16_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                     "-U__CUDA_NO_BFLOAT162_OPERATORS__", "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                     "--expt-relaxed-constexpr", "--expt-extended-lambda"]})],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
)
