from setuptools import setup, Extension
from Cython.Build import cythonize

ext = Extension(
    name="fast_fill",
    sources=["fast_fill.pyx"],
    language="c++",
    extra_compile_args=["-O3", "-march=native"],
)

setup(
    name="fast_fill",
    ext_modules=cythonize([ext], annotate=True),
)