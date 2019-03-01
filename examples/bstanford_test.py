'''
Analyze the straight and swept flat-plate wings for comparison against
the results provided by Bret Stanford.
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

min_t = 0.001
max_t = 0.01

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

# Set the finite-element size
num_nodes = (nx+1)*(ny+1)
num_elements = nx*ny

# There are 6 degrees of freedom per node
vars_per_node = 6

# Create the TACS assembler object
tacs = TACS.Assembler.create(comm, vars_per_node,
                             num_nodes, num_elements)

elems = []
elem_conn = []

# Add all the elements
for j in xrange(ny):
    for i in xrange(nx):
        # Create the shell element class
        dv_num = i + nx*j
        stiff = constitutive.isoFSDT(rho, E, nu, kcorr, ys, t,
                                     dv_num, min_t, max_t)

        # stiff.setRefAxis([0.0, 1.0, 0.0])
        shell_element = elements.MITCShell(2, stiff)
        elems.append(shell_element)

        # Set the element connectivity
        elem_conn.append([i + (nx+1)*j, 
                          i+1 + (nx+1)*j,
                          i + (nx+1)*(j+1), 
                          i+1 + (nx+1)*(j+1)])
        
# Create the connectivity array
conn = np.array(elem_conn, dtype=np.intc).flatten()
ptr = np.arange(0, len(conn)+1, 4, dtype=np.intc)

# Set the connectivity and the elements
tacs.setElements(elems)
tacs.setElementConnectivity(conn, ptr)

# Add the nodal boundary conditions
bc_nodes = np.arange(nx+1, dtype=np.intc)
tacs.addBCs(bc_nodes)

# Initialize the mesh
tacs.initialize()

# Set the node locations
X = tacs.createNodeVec()
Xpts = X.getArray()
for j in xrange(ny+1):
    for i in xrange(nx+1):
        node = i + (nx+1)*j

        y = Ly*(1.0*j/ny)
        x = x_off + Lx*(1.0*i/nx) + np.tan(sweep)*y

        Xpts[3*node] = x
        Xpts[3*node+1] = y

# Set the nodes into TACS
tacs.setNodes(X)
tacs.getNodes(X)

# Initialize the structural part of the solver
dlm_solver.initStructure(tacs)

# Set the density and Mach number
rho = 1.225
Uval = 4.0
Mach = 0.0

if 'velocity_sweep' in sys.argv:
    # Set the span of velocities that we'll use
    nvals = 25
    U = np.linspace(2, 20, nvals)
    
    # Set the number of structural modes to use
    nmodes = 1
    pvals = dlm_solver.velocitySweep(rho, Mach, U, nmodes, omega)
    
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

    sys.exit(0)

# Set up the subspace that we'll use for the flutter analysis
use_modes = False # True
msub = 50
rsub = 15

# Compute the derivative of tihs mode
kmode = 0

# Maximum number of determinant iterations
max_iters = 15

# Solve the problem using det-iteration
dlm_solver.setUpSubspace(msub, rsub, use_modes=use_modes)
# p = dlm_solver.computeFlutterMode(rho, Uval, Mach, 
#                                   kmode, max_iters=max_iters)

p = dlm_solver.computeFlutterModeEig(rho, Uval, Mach, 
                                     kmode, max_iters=max_iters)

# Set up the information for the derivatives
num_design_vars = nx*ny
x = np.zeros(num_design_vars)
px = -1.0 + 2.0*np.random.uniform(size=x.shape)
tacs.getDesignVars(x)

# Test the derivatives of the matrix
dlm_solver.testMatDeriv(x)

# Compute the derivative based on the frozen-mode approximation
pderiv = dlm_solver.computeFrozenDeriv(rho, Uval, Mach, p,
                                       num_design_vars)

# Compute the derivative along the direction px
pd = np.dot(pderiv, px)

# Set up the higher-order finite-difference
dh = 1e-5
offset = [2, 1, -1, -2]
weights = np.array([-1.0/12.0, 2.0/3.0, -2.0/3.0, 1.0/12.0])
pvals = np.zeros(4, dtype=np.complex)

for i in xrange(len(offset)):
    # Set the design variables at x[0] + dh
    xnew = x + offset[i]*dh*px
    tacs.setDesignVars(xnew)
    dlm_solver.setUpSubspace(msub, rsub, use_modes=use_modes)
    # pvals[i] = dlm_solver.computeFlutterMode(rho, Uval, Mach,
    #                                          kmode, max_iters=max_iters)

    pvals[i] = dlm_solver.computeFlutterModeEig(rho, Uval, Mach,
                                                kmode, max_iters=max_iters)

pfd = np.dot(pvals, weights)/dh

print 'FD =    %20.10e %20.10e'%(pfd.real, pfd.imag)
print 'Deriv = %20.10e %20.10e'%(pd.real, pd.imag)
