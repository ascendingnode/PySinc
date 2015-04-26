from distutils.core import setup
from Cython.Build import cythonize

# The Cython build call
setup(
        name = "PySinc",
        ext_modules = cythonize(('PySinc.pyx')),
        )
