from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        Pybind11Extension(
            'volt_spatial_assembler._spatial_assembler',
            ['src/glb_python.cpp'],
            include_dirs=['src'],
            cxx_std=17,
            extra_compile_args=['-O3', '-pthread'],
        ),
    ],
)
