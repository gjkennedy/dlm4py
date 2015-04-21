
FF90=gfortran

%.f90.o:
	${FF90} -c $< -o $*.o

default: all

all: 
	gfortran -c precision.f90 -o precision.o
	gfortran -c dlm.f90 -o dlm.o
	f2py dlm.pyf

# This should only be called when you want to destroy the existing
# .pyf file!
overwrite_pyf_file:
	f2py --overwrite-signature -m dlm -h dlm.pyf dlm.f90
