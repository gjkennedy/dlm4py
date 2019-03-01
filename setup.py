from numpy.distutils.core import setup
from numpy.distutils.core import Extension

libs = ['lapack', 'blas']
ext = Extension(name='dlm4py.dlm', sources=['src/dlm.f90'], libraries=libs)

setup(name='dlm4py',
      description='A simple DLM implementation',
      author='Graeme J. Kennedy',
      author_email='graeme.kennedy@ae.gatech.edu',
      packages=['dlm4py'], ext_modules=[ext])
