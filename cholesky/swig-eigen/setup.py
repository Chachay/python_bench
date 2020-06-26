from distutils.core import *
from distutils      import sysconfig

import numpy

try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

_SwigMod = Extension("_swig_eigen_mod",
                   ["swig_mod.i","swig_mod.cpp"],
                   swig_opts=['-c++', '-py3'],
                   include_dirs = [numpy_include, "./eigen"]
                   )

setup(  name        = "SwigMod function",
        description = "Swiged Calc",
        author      = "Chachay",
        version     = "1.0",
        ext_modules = [_SwigMod]
        )
