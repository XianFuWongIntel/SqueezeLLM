from setuptools import setup
from intel_extension_for_pytorch.xpu.cpp_extension import DPCPPExtension, DpcppBuildExtension
import os

os.environ["CXX"] = "dpcpp"
os.environ["CC"] = "dpcpp"

setup(
    name='quant_xpu',
    ext_modules=[DPCPPExtension(
        'quant_xpu',
        ['quant_xpu.cpp', 'quant_xpu_kernel.dp.cpp']
    )],
    cmdclass={'build_ext': DpcppBuildExtension}
)
