
FF90=gfortran

default: all

all: dlm.f90
	f2py -c -m dlm dlm.f90

# This should only be called when you want to destroy the existing
# .pyf file!
overwrite_pyf_file:
	f2py --overwrite-signature -m dlm -h dlm.pyf dlm.f90

%.f90.o:
	${FF90} -c $< -o $*.o
