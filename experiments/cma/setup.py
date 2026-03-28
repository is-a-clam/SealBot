"""Build the CMA-ES minimax wrapper.

Usage:  python setup.py build_ext --inplace
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    name="cma_minimax_cpp",
    ext_modules=[
        Pybind11Extension(
            "cma_minimax_cpp",
            ["cma_wrapper.cpp"],
            cxx_std=17,
            extra_compile_args=["-O3", "-march=native", "-DNDEBUG"],
            include_dirs=["../../best"],
        ),
    ],
    cmdclass={"build_ext": build_ext},
)
