# DLM4PY: A simple DLM implementation #

dlm4py is a simple Doublet Lattice Method (DLM) and flutter code implemented in Python and Fortran that implements exact derivatives. To compile dlm4py you will need Python, a Fortran compiler and f2py. To run flutter analyses you will also need a working version of TACS.

# Installation #

Installation utilizes f2py through numpy distutils. Use

python setup.py build

I recommend a local installation:

python setup.py install --user --prefix=
