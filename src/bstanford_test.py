'''
Analyze Bret Stanford's straight and swept flat-plate wings for
comparison against his results.
'''

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from tacs import TACS, elements, constitutive
from dlm4py import DLM

# Create the DLM object and add the mesh
dlm_solver = DLM(is_symmetric=1)

# Dimensions of the mesh
nspan = 20
nchord = 12

# Geometric parameters
semi_span = 1
chord = 0.3
taper = 1.0
dihedral = 0.0

sweep = 0.0
if 'swept' in sys.argv:
    sweep = np.arctan(0.6) 

dlm_solver.addMeshSegment(nspan, nchord, semi_span, chord,
                          sweep=sweep, dihedral=dihedral, 
                          taper_ratio=taper)

# Set the material properties for the shell
rho = 2800.0
E = 70e9
nu = 0.3
kcorr = 0.8333
ys = 400e6
t = 0.001

# Set the size of the mesh (nx, ny) finite-elements
nx = 12
ny = 40

# Set the dimensions for the wing mesh
Lx = 0.21
Ly = 0.85

# Compute the chord offset
x_off = 0.5*(chord - Lx) - 0.25*chord

# Finally, now we can start to assemble TACS
comm = MPI.COMM_WORLD
num_nodes = (nx+1)*(ny+1)
num_elements = nx*ny
csr_size = 4*num_elements
num_load_cases = 1

# There are 6 degrees of freedom per node
vars_per_node = 6

# Create the TACS assembler object
tacs = TACS.TACSAssembler(comm, num_nodes, vars_per_node,
                          num_elements, num_nodes,
                          csr_size, num_load_cases)

# Since we're doing everything ourselves, we have to add nodes
# elements and finalize the mesh. This is a serial case, therefore
# global and local node numbers are the same (or they could be some
# other unique mapping)
for i in xrange(num_nodes):
    tacs.addNode(i, i)

# Create the shell element class 
stiff = constitutive.isoFSDTStiffness(rho, E, nu, kcorr, ys, t)
stiff.setRefAxis([0.0, 1.0, 0.0])
shell_element = elements.MITCShell2(stiff)

# Add all the elements
for j in xrange(ny):
    for i in xrange(nx):
        elem_conn = np.array([i + (nx+1)*j, i+1 + (nx+1)*j,
                              i + (nx+1)*(j+1), i+1 + (nx+1)*(j+1)], dtype=np.intc)
        tacs.addElement(shell_element, elem_conn)

# Finalize the mesh
tacs.finalize()

# Set the boundary conditions and nodal locations
bcs = np.arange(vars_per_node, dtype=np.intc)
Xpts = np.zeros(3*num_nodes)

for j in xrange(ny+1):
    for i in xrange(nx+1):
        node = i + (nx+1)*j

        y = Ly*(1.0*j/ny)
        x = x_off + Lx*(1.0*i/nx) + np.tan(sweep)*y

        Xpts[3*node] = x
        Xpts[3*node+1] = y

        if j == 0:
            tacs.addBC(node, bcs)

tacs.setNodes(Xpts)

# Initialize the structural part of the solver
dlm_solver.initStructure(tacs)

# Set the density and Mach number
rho = 1.225
Mach = 0.0

# Set the span of velocities that we'll use
nvals = 4
U = np.linspace(2, 5, nvals)

# nvals = 25
# U = np.linspace(2, 20, nvals)

# Set the number of structural modes to use
nmodes = 5
pvals = dlm_solver.velocitySweep(rho, Mach, U, nmodes)

# Plot the results using
symbols = ['-ko', '-ro', '-go', '-bo', '-mo']
plt.figure()
for kmode in xrange(nmodes):
    plt.plot(U, pvals[kmode,:].imag, symbols[kmode], 
             label='mode %d'%(kmode))
plt.legend()
plt.grid()

plt.figure()
for kmode in xrange(nmodes):
    plt.plot(U, pvals[kmode,:].real, symbols[kmode], 
             label='mode %d'%(kmode))
plt.legend()
plt.grid()
plt.show()


