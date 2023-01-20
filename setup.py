from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

setup(
    name = 'DTSNE',
    ext_modules=[
        Extension('dt_sne',
                  sources=['dt_sne.pyx'],
                  #extra_compile_args=['-O3'],
                  language='c++')
        ],
    include_dirs=[np.get_include()],
    cmdclass = {'build_ext': build_ext}
)
