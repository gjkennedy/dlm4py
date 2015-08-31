'''
Perform either the Blair or Rodden test cases
'''

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from tacs import TACS, elements, constitutive, functions
from dlm4py import DLM
from tacs_opt_tools import StructureOptTools 

# Create the DLM object and add the mesh
dlm_solver = DLM(is_symmetric=1)

# Dimensions of the mesh
nspan = 15
nchord = 12

# Geometric parameters
span = 58.7629
semi_span = 0.5*span

# The taper ratio for the overall wing
taper = 0.275

# The 1/4 chord sweep
sweep = 35.0/180.0*np.pi

# The dihedral angle
dihedral = 3.0/180.0*np.pi

# The le symmetry location
xle = [ 22.29691, 0.0, 4.4228 ]

# Set the semi-span location
ysemi = 0.37*semi_span

rchord = 13.619
ychord = 7.26
tchord = taper*rchord

xroot = [xle[0] + 0.25*rchord, 
         0.0, xle[2]]
xyehudi = [ xroot[0] + np.tan(sweep)*ysemi,
            ysemi, 
            xroot[2] + np.tan(dihedral)*ysemi ]

# Compute the taper ratio for the first segment
ytaper = ychord/rchord
ztaper = tchord/ychord

# Set the segments in the solver
dlm_solver.addMeshSegment(nspan, nchord, ysemi, rchord,
                          x0=xroot, sweep=sweep, dihedral=dihedral, 
                          taper_ratio=ytaper)
dlm_solver.addMeshSegment(nspan, nchord, semi_span-ysemi, ychord,
                          x0=xyehudi, sweep=sweep, dihedral=dihedral,
                          taper_ratio=ztaper)

# Solve and write the solution to a file
Cp = dlm_solver.solve(1.0)
dlm_solver.writeToFile(Cp, filename='crm_dlm_solution.dat')

# Create the mesh loader class
comm = MPI.COMM_WORLD
bdf_file = 'CRM_box_2nd.bdf'
mesh = TACS.TACSMeshLoader(comm)
mesh.scanBdfFile(bdf_file)

# Create a constitutive dictionary indexed by component name
con_dict = {}

# Keep track of the design variable number
t_num = 0

# Set the reference axes to use for visualizing output
x_axis = np.array([1.0, 0.0, 0.0])
s = np.sin(np.pi/6.0)
c = np.sin(np.pi/6.0)
z_axis = np.array([s, 0.0, c])

# Set the material properties for the shell
rho = 2800.0
E = 70e9
poisson = 0.3
kcorr = 0.8333
ys = 400e6

for k in xrange(mesh.getNumComponents()):
    comp = mesh.getComponentDescript(k)

    # The thickness values
    t = 0.05
    t_lb = 0.002
    t_ub = 0.1

    # Set the constitutive object
    con = constitutive.isoFSDTStiffness(rho, E, poisson, kcorr, ys,
                                        t, t_num, t_lb, t_ub)
    con_dict[comp] = con

    if 'RIB' in comp:
        con.setRefAxis(x_axis)
    else:
        con.setRefAxis(z_axis)

    t_num += 1

# Create the TACSAssembler object
num_load_cases = 1
tacs = StructureOptTools.set_up_tacs(mesh, con_dict, num_load_cases)

# Initialize the structural part of the solver
dlm_solver.initStructure(tacs)

# Set the density and Mach number
rho = 0.379597
Mach = 0.85

# Set the span of velocities that we'll use
nvals = 10
a0 = 296.535
U = np.linspace(0.7*a0, 0.9*a0, nvals)

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

