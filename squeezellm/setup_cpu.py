from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

os.environ["CXX"] = "dpcpp"
os.environ["CC"] = "dpcpp"

setup(
    name='quant_cpu',
    ext_modules=[cpp_extension.CppExtension(
        'quant_cpu',
        ['quant_cpu.cpp', 'quant_cpu_kernel.dp.cpp'],
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
