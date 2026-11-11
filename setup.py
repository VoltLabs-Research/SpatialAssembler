from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

MESHOPTIMIZER = [
    f'third_party/meshoptimizer/{name}.cpp'
    for name in ('allocator', 'vertexcodec', 'indexcodec', 'vertexfilter', 'vfetchoptimizer')
]

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        Pybind11Extension(
            'volt_spatial_assembler._spatial_assembler',
            ['src/python_binding.cpp', 'src/glb_core.cpp', *MESHOPTIMIZER],
            include_dirs=['src', 'third_party/meshoptimizer', 'third_party/cgltf'],
            cxx_std=17,
            extra_compile_args=['-O3', '-pthread'],
        ),
    ],
)
