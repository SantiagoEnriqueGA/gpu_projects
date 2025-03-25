from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("misc/monte_carlo_pi_cython.pyx", compiler_directives={"language_level": "3"}),
    include_dirs=[np.get_include()],  # Add NumPy include directory

)

# To build the cython code, run the following command in the terminal:
# The inplace flag is used to build the extension in the same directory as the source code.
# python setup.py build_ext --inplace
