'''
This is the python-level interface for the dlm code.
'''
import numpy as np
import sys
from tacs import TACS, elements, constitutive
from mpi4py import MPI
import dlm

try:
    import matplotlib.pyplot as plt
except:
    pass

class DLM:
    def __init__(self, is_symmetric=1, epstol=1e-12):
        '''
        Initialize the internal mesh.
        '''

        # A flag that indicates whether this geometry is symmetric or
        # not. Symmetric by default.
        self.is_symmetric = is_symmetric
        self.use_steady_kernel = True
        self.epstol = epstol
        
        # The influence coefficient matrix
        self.Dtrans = None

        # The total number of panels and nodes
        self.npanels = 0
        self.nnodes = 0

        # The points required for analysis
        self.Xi = None
        self.Xo = None
        self.Xr = None
        self.dXav = None

        # The surface connectivity - required for visualization and
        # load/displacement transfer
        self.X = None
        self.conn = None

        return

    def addMeshSegment(self, n, m, span, root_chord, x0=[0, 0, 0], 
                       sweep=0.0, dihedral=0.0, taper_ratio=1.0):
        '''
        Add a segment to the current set of mesh points. Note that
        once a segment is added, you cannot delete it.
        
        input:
        n:           number of panels along the spanwise direction
        m:           number of panels along the chord-wise direction
        root_chord:  segment root chord (at x0)
        x0:          (x, y, z) position
        sweep:       sweep angle in radians
        dihedral:    dihedral angle in radians
        taper_ratio: the segment taper ratio
        '''

        npanels = n*m
        x0 = np.array(x0)
        Xi = np.zeros((npanels, 3))
        Xo = np.zeros((npanels, 3))
        Xr = np.zeros((npanels, 3))
        dXav = np.zeros(npanels)

        # Compute the inboard/outboard and receiving point locations
        dlm.computeinputmeshsegment(n, m, x0, span, dihedral, sweep,
                                    root_chord, taper_ratio, 
                                    Xi.T, Xo.T, Xr.T, dXav)

        conn = np.zeros((n*m, 4), dtype=np.intc)
        X = np.zeros(((n+1)*(m+1), 3))

        # Compute the x,y,z surface locations
        dlm.computesurfacesegment(n, m, x0, span, dihedral, sweep,
                                  root_chord, taper_ratio, X.T, conn.T)

        # Append the new mesh components
        if self.Xi is None:
            # Set the mesh locations if they do not exist already
            self.Xi = Xi
            self.Xo = Xo
            self.Xr = Xr
            self.dXav = dXav

            # Set the surface mesh locations/connectivity
            self.X = X
            self.conn = conn
        else:
            # Append the new mesh locations
            self.Xi = np.vstack(self.Xi, Xi)
            self.Xo = np.vstack(self.Xo, Xo)
            self.Xr = np.vstack(self.Xr, Xr)
            self.dXav = np.vstack(self.dXav, dXav)

            # Append the new connectivity and nodes
            self.X = np.vstack(self.X, X)
            self.conn = np.vstack(self.conn, conn + self.nnodes)

        # Set the new number of nodes/panels
        self.npanels = self.Xi.shape[0]
        self.nnodes = self.X.shape[0]

        return

    def computeFlutterDeterminant(self, U, p, qinf, Mach,
                                  nvecs, omega, vwash, dwash, modes):
        '''
        Compute the flutter determinant given as follows:

        det(F(p))

        F = p**2 + omega**2 - qinf*modes^{T}*D^{-1}*wash  
        '''

        # Compute the contribution from the reduced structural problem
        F = np.zeros((nvecs, nvecs), dtype=np.complex)
        for i in xrange(nvecs):
            F[i, i] = p**2 + omega[i]**2 

        # Compute the influence coefficient matrix
        self.computeInfluenceMatrix(U, p.imag, Mach)

        # Compute the boundary condition: -1/U*(dh/dt + U*dh/dx)
        # dwash = -dh/dx, vwash = -dh/dt 
        wash = p*vwash/U + dwash

        # Solve for the normal wash due to the motion of the wing
        # through the flutter mode
        Cp = np.linalg.solve(self.Dtrans.T, wash)

        # Compute the forces due to the flutter motion
        forces = np.zeros((self.nnodes, 3), dtype=np.complex)

        # Add the forces to the vector
        for i in xrange(nvecs):
            forces[:,:] = 0.j
            dlm.addcpforces(qinf, Cp[:,i], self.X.T, self.conn.T, forces.T)
            F[:,i] += np.dot(modes.T, forces.flatten())
            
        return np.linalg.det(F)/(omega[0]**(2*nvecs))

    def computeStaticLoad(self, aoa, U, qinf, Mach, 
                          nvecs, omega, modes, filename=None):
        '''
        Compute the static loads due 
        '''

        # Compute the influence coefficient matrix
        omega_aero = 0.0
        self.computeInfluenceMatrix(U, omega_aero, Mach)

        # Evaluate the right-hand-side
        w = np.zeros(self.npanels, dtype=np.complex)
            
        # Compute a right hand side
        dlm.computeperiodicbc(w, aoa, omega_aero, self.Xi.T, self.Xo.T)

        # Solve the resulting right-hand-side
        Cp = np.linalg.solve(self.Dtrans.T, w)

        # Compute the forces
        forces = np.zeros((self.nnodes, 3), dtype=np.complex)
        dlm.addcpforces(qinf, Cp, self.X.T, self.conn.T, forces.T)

        # Compute the generalized displacements
        u = np.dot(modes.T, forces.flatten())/omega**2

        # Compute the full set of diplacements
        udisp = np.dot(modes, u).real

        if not filename is None:
            self.writeToFile(Cp, filename, udisp.reshape(self.nnodes, 3))
        
        return

    def computeInfluenceMatrix(self, U, omega_aero, Mach):
        '''
        Compute the influence coefficient matrix
        '''

        if self.Dtrans is None or self.Dtrans.shape[0] < self.npanels:
            # Allocate the influence coefficient matrix
            self.Dtrans = np.zeros((self.npanels, self.npanels), dtype=np.complex)

        # Compute the influence coefficient matrix
        dlm.computeinfluencematrix(self.Dtrans.T, omega_aero, U, Mach,
                                   self.Xi.T, self.Xo.T, self.Xr.T, self.dXav,
                                   self.is_symmetric, self.use_steady_kernel, 
                                   self.epstol)
        return
    
    def solve(self, U, aoa=0.0, omega=0.0, Mach=0.0, w=None):
        '''
        Solve the linear system (in the frequency domain)
        '''

        # Compute the influence coefficient matrix
        self.computeInfluenceMatrix(U, omega, Mach)

        if w is None:
            # Evaluate the right-hand-side
            w = np.zeros(self.npanels, dtype=np.complex)
            
            # Compute a right hand side
            dlm.computeperiodicbc(w, aoa, omega, self.Xi.T, self.Xo.T)

        Cp = np.linalg.solve(self.Dtrans.T, w)

        return Cp

    def getModeBCs(self, mode):
        '''
        Transfer the displacements specified at the surface
        coordinates to the normal-component at the receiving points.
        '''

        vwash = np.zeros(self.npanels)
        dwash = np.zeros(self.npanels)
        dlm.getmodebcs(mode.T, self.X.T, self.conn.T, vwash, dwash)

        return vwash, dwash

    def addAeroForces(self, qinf, Cp):
        '''
        Compute the forces on the aerodynamic surface mesh.
        '''

        Cp = np.array(Cp, dtype=np.complex)
        forces = np.zeros((self.nnodes, 3), dtype=np.complex)
        dlm.addcpforces(qinf, Cp, self.X.T, self.conn.T, forces.T)

        return forces        

    def writeToFile(self, Cp, filename='solution.dat', u=None):
        '''
        Write the Cp solution (both the real and imaginary parts)
        to a file for visualization
        '''

        fp = open(filename, 'w')

        if fp:
            fp.write('Title = \"Solution\"\n')
            fp.write('Variables = X, Y, Z, Re(Cp), Im(Cp)\n')
            fp.write('Zone T=wing n=%d e=%d '%(self.nnodes, self.npanels))
            fp.write('datapacking=block ')
            fp.write('zonetype=fequadrilateral ')
            fp.write('varlocation=([4,5]=cellcentered)\n')

            # Write out the panel locations
            if u is None:
                for j in xrange(3):
                    for i in xrange(self.nnodes):
                        fp.write('%e\n'%(self.X[i, j]))
            else:
                for j in xrange(3):
                    for i in xrange(self.nnodes):
                        fp.write('%e\n'%(self.X[i, j] + u[i, j]))

            # Write out the real/imaginary Cp values
            for i in xrange(self.npanels):
                fp.write('%e\n'%(Cp[i].real))
            for i in xrange(self.npanels):
                fp.write('%e\n'%(Cp[i].imag))

            for i in xrange(self.npanels):
                fp.write('%d %d %d %d\n'%(
                        self.conn[i,0]+1, self.conn[i,1]+1,
                        self.conn[i,2]+1, self.conn[i,3]+1))
            
            fp.close()

        return

# Create the DLM object and add the mesh
dlm_solver = DLM(is_symmetric=1)

if 'blair' in sys.argv:
    # Perform the highly-simplified test from Blair
    nchord = 3
    nspan = 3

    # Set the geometry
    semi_span = 12.0
    chord = 12.0
    taper = 1.0
    sweep = 0.0
    dihedral = 0.0

    dlm_solver.addMeshSegment(nspan, nchord, semi_span, chord,
                              sweep=sweep, dihedral=dihedral, 
                              taper_ratio=taper)

    # Set the values of things
    U = 1.0
    b = 6.0
    kr = 1.0
    Mach = 0.5
    # Compute the circular frequency from the reduced frequency:
    # kr = omega*b/U
    omega = U*kr/b

    # dlm_solver.use_steady_kernel = False
    dlm_solver.computeInfluenceMatrix(U, omega, Mach)

    w = np.zeros(nchord*nspan, dtype=np.complex)
    w[:] = -1j
    Cp = np.linalg.solve(dlm_solver.Dtrans.T, w)
    Cp_blair = [ -5.4900e-01 + 6.2682e+00j,
                  -3.8862e+00 + 2.4495e+00j,
                  -3.8736e+00 + 1.1745e+00j,
                  -5.9144e-01 + 5.8092e+00j,
                  -3.6405e+00 + 2.1530e+00j,
                  -3.6234e+00 + 1.0281e+00j,
                  -5.8286e-01 + 4.5474e+00j,
                  -2.8983e+00 + 1.4663e+00j,
                  -2.8893e+00 + 7.1186e-01j ]

    print '   %12s %12s %12s %12s %12s'%(
        'Re(Cp_B)', 'Im(Cp_B)', 'Re(Cp)', 'Im(Cp)', 'Rel err')

    for j in xrange(nchord*nspan):
        print '%2d %12.4e %12.4e %12.4e %12.4e %12.4e'%(
            j, Cp_blair[j].real, Cp_blair[j].imag,
            Cp[j].real, Cp[j].imag, abs(Cp[j] - Cp_blair[j])/abs(Cp[j]))

    print 'CL = ', np.sum(Cp)/(nchord*nspan)
    sys.exit(0)

elif 'rodden' in sys.argv:
    # Set up an aspect-ratio 20 wing
    nchord = 10
    nspan = 40

    # The wing aspect ratio
    Ar = 20.0

    # Set the geometry
    b = 10.0
    semi_span = 0.5*b
    chord = b/Ar

    # Compute the area of the rectangular wing
    Area = semi_span*chord

    # Set other geometric parameters
    taper = 1.0
    sweep = 0.0
    dihedral = 0.0
    dlm_solver.addMeshSegment(nspan, nchord, semi_span, chord,
                              sweep=sweep, dihedral=dihedral, 
                              taper_ratio=taper)
    Mach = 0.0
    U = 1.0

    # Set the downwash over the wing
    npanels = dlm_solver.npanels
    w = np.zeros(npanels, dtype=np.complex)

    # Set space for the frequencies
    nfreq = 25
    kr = np.linspace(0.0, 2, nfreq)
    Cl = np.zeros(nfreq, dtype=np.complex)
    Clalpha = np.zeros(nfreq, dtype=np.complex)

    # Set the angle of attack to use
    aoa = 0.5/180.0*np.pi

    for k in xrange(nfreq):
        # kr = omega*b/U
        omega = kr[k]*U/(0.5*chord)

        # Compute the downwash
        for i in xrange(npanels):
            # Compute the contribution to the boundary condition:
            # -1/U*(dh/dt + U*dh/dx), where h = (x - 0.25c)*e^{j*omega*t}
            w[i] = -1.0 - 1j*(omega/U)*(dlm_solver.Xr[i, 0] - 0.25*chord)
            
        # Solve the resulting system
        dlm_solver.computeInfluenceMatrix(U, omega, Mach)
        Cp = np.linalg.solve(dlm_solver.Dtrans.T, w)

        # Compute the coefficient of pressure
        Cl[k] = np.sum(Cp)/(nchord*nspan)
        Clalpha[k] = Cl[k]

        print '%3d %12.5e %12.5e %12.5e %12.5e %12.5e'%(
            k, kr[k], Cl[k].real, Cl[k].imag, Clalpha[k].real, Clalpha[k].imag)

    plt.figure()
    plt.plot(kr, Cl.imag, '-ko', label='Cl imaginary')
    plt.plot(kr, Cl.real, '-ro', label='Cl real')
    plt.legend()
    plt.grid()
    plt.show()
    sys.exit(0)                     

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

# Now, set up the load and displacement transfer object
struct_root = 0
aero_root = 0
aero_member = 1

# Get the aerodynamic mesh connectivity
aero_pts = dlm_solver.X.flatten()
aero_conn = dlm_solver.conn.flatten()

# Specify the load/displacement transfer data
max_h_size = 0.5
gauss_order = 2
transfer = TACS.LDTransfer(comm, struct_root, aero_root, aero_member,
                           aero_pts, aero_conn, tacs, max_h_size, gauss_order)

# Set the aerodynamic surface nodes
transfer.setAeroSurfaceNodes(aero_pts)

# Set up the frequency analysis object
sigma = 0.1

mat = tacs.createFEMat()
mmat = tacs.createFEMat()

# Create the preconditioner and the solver object
lev = 10000
fill = 10.0
reorder_schur = 1
pc = TACS.PcScMat(mat, lev, fill, reorder_schur)

# Create the GMRES object 
gmres_iters = 30
nrestart = 0
isflexible = 0
gmres = TACS.GMRES(mat, pc, gmres_iters, nrestart, isflexible)

# Set up the solver object
maxeigvecs = 40
neigvals = 5
eigtol = 1e-16

# Set the load case number
load_case = 0
freq = TACS.TACSFrequencyAnalysis(tacs, load_case, sigma,
                                  mat, mmat, gmres, 
                                  maxeigvecs, neigvals, eigtol)
freq.solve(TACS.KSMPrintStdout('Freq', comm.rank, 1))

# Extract the eignvectors and eigenvalues
vec = tacs.createVec()

# Store information related to the surface modes
nnodes = dlm_solver.nnodes
npanels = dlm_solver.npanels

# Get the surface modes and the corresponding normal wash
modes = np.zeros((3*nnodes, neigvals))
vwash = np.zeros((npanels, neigvals))
dwash = np.zeros((npanels, neigvals))

# Store the natural frequencies of vibration
omega = np.zeros(neigvals)

for k in xrange(neigvals):
    # Extract the eigenvector
    eigval, error = freq.extractEigenvector(k, vec)
    omega[k] = np.sqrt(eigval)

    # Transfer the eigenvector to the aerodynamic surface
    disp = np.zeros(3*dlm_solver.nnodes)
    transfer.setDisplacements(vec)    
    transfer.getDisplacements(disp)

    # Compute the normal wash on the aerodynamic mesh
    npts = dlm_solver.nnodes
    vk, dk = dlm_solver.getModeBCs(disp.reshape(nnodes, 3))
    
    # Store the normal wash and the surface displacement
    vwash[:,k] = vk
    dwash[:,k] = dk
    modes[:,k] = disp

print 'omega = ', omega

# Set the flight conditions
Mach = 0.0
rho = 1.225

# Set up the number of values
nvals = 25
U = np.linspace(2, 20, nvals)
pvals = np.zeros((neigvals, nvals), dtype=np.complex)
omega_aero = np.zeros(nvals)
damping = np.zeros(nvals)

for keig in xrange(neigvals):
    for i in xrange(nvals):
        qinf = 0.5*rho*U[i]**2
    
        # Compute an estimate of p based on the lowest natural frequency
        if i == 0:
            eps = 1e-3
            p1 = -0.1 + 1j*omega[keig]
            p2 = p1 + (eps + 1j*eps)
        elif i == 1:
            eps = 1e-3
            p1 = 1.0*pvals[keig,0]
            p2 = p1 + (eps + 1j*eps)

        # The following code tries to extrapolate the next point
        elif i == 2:
            eps = 1e-3
            p1 = 2.0*pvals[keig,i-1] - pvals[keig,i-2]
            p2 = p1 + (eps + 1j*eps)
        else: 
            eps = 1e-3
            p1 = 3.0*pvals[keig,i-1] - 3.0*pvals[keig,i-2] + pvals[keig,i-3]
            p2 = p1 + (eps + 1j*eps)

        det1 = dlm_solver.computeFlutterDeterminant(U[i], p1, qinf, Mach,
                                                    neigvals, omega, vwash, dwash, modes)
        det2 = dlm_solver.computeFlutterDeterminant(U[i], p2, qinf, Mach,
                                                    neigvals, omega, vwash, dwash, modes)

        # Perform the flutter determinant iteration
        max_iters = 50
        det0 = 1.0*det1
        for k in xrange(max_iters):
            # Compute the new value of p
            pnew = (p2*det1 - p1*det2)/(det1 - det2)

            # Move p2 to p1
            p1 = 1.0*p2
            det1 = 1.0*det2

            # Move pnew to p2 and compute pnew
            p2 = 1.0*pnew
            det2 = dlm_solver.computeFlutterDeterminant(U[i], p2, qinf, Mach,
                                                        neigvals, omega, vwash, dwash, modes)
            if k == 0:
                print '%4s %10s %10s %10s'%('Iter', 'Det', 'Re(p)', 'Im(p)') 
            print '%4d %10.2e %10.6f %10.6f'%(k, abs(det2), p2.real, p2.imag)

            if abs(det2) < 1e-6*abs(det0):
                break

        # Store the final value of p
        pvals[keig, i] = p2

        print '%4s %10s %10s %10s'%('Iter', 'U', 'Re(p)', 'Im(p)')
        print '%4d %10.6f %10.6f %10.6f'%(i, U[i], pvals[keig,i].real, pvals[keig,i].imag)

symbols = ['-ko', '-ro', '-go', '-bo', '-mo']
plt.figure()
for keig in xrange(neigvals):
    plt.plot(U, pvals[keig,:].imag, symbols[keig], label='mode %d'%(keig))
plt.legend()
plt.grid()

plt.figure()
for keig in xrange(neigvals):
    plt.plot(U, pvals[keig,:].real, symbols[keig], label='mode %d'%(keig))
plt.legend()
plt.grid()
plt.show()

