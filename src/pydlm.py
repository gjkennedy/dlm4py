'''
This is the python-level interface for the dlm code.
'''

import numpy as np
import sys
from tacs import TACS, elements, constitutive
from mpi4py import MPI
import dlm

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

    def initStructure(self, tacs, max_h_size=0.5,
                      gauss_order=2):
        '''
        Set up the load and displacement transfer object for a general
        TACS finite-element model.
        '''

        # Get the communicator from the TACSAssembler object
        self.tacs = tacs
        comm = self.tacs.getMPIComm()

        # Now, set up the load and displacement transfer object
        struct_root = 0
        aero_root = 0

        # Only use the first rank as an aerodynamic processor
        aero_member = 0
        if comm.rank == 0:
            aero_member = 1

        # Get the aerodynamic mesh connectivity
        aero_pts = self.X.flatten()
        aero_conn = self.conn.flatten()

        # Specify the load/displacement transfer data
        self.transfer = TACS.LDTransfer(comm, struct_root, 
                                        aero_root, aero_member,
                                        aero_pts, aero_conn, 
                                        self.tacs, max_h_size, gauss_order)

        # Set the aerodynamic surface nodes
        self.transfer.setAeroSurfaceNodes(aero_pts)

        # Set up the matrices/pc/Krylov solver that will be required
        # for the flutter analysis
        self.kmat = tacs.createFEMat()
        self.mmat = tacs.createFEMat()

        # Create the preconditioner and the solver object.  Note that
        # these settings are best for a shell-type finite-element
        # model.
        lev = 10000
        fill = 10.0
        reorder_schur = 1
        self.pc = TACS.PcScMat(self.kmat, lev, fill, reorder_schur)

        # Create the GMRES object 
        gmres_iters = 10
        nrestart = 0
        isflexible = 0
        self.gmres = TACS.GMRES(self.kmat, self.pc, 
                                gmres_iters, nrestart, isflexible)

        return

    def velocitySweep(self, rho, Mach, U, nmodes,
                      sigma=0.1):
        
        # First, set up an eigenvalue solver
        maxeigvecs = 40
        eigtol = 1e-14

        # Set the load case number
        load_case = 0
        freq = TACS.TACSFrequencyAnalysis(self.tacs, load_case, sigma,
                                          self.kmat, self.mmat, self.gmres, 
                                          maxeigvecs, nmodes, eigtol)
        print_obj = TACS.KSMPrintStdout('Freq', 
                                        self.tacs.getMPIComm().rank, 1)
        freq.solve()

        # Extract the eignvectors and eigenvalues
        vec = self.tacs.createVec()

        # Get the surface modes and the corresponding normal wash
        modes = np.zeros((3*self.nnodes, nmodes))
        vwash = np.zeros((self.npanels, nmodes))
        dwash = np.zeros((self.npanels, nmodes))

        # Store the natural frequencies of vibration
        omega = np.zeros(nmodes)

        # Extract the natural frequencies of vibration
        for k in xrange(nmodes):
            # Extract the eigenvector
            eigval, error = freq.extractEigenvector(k, vec)
            omega[k] = np.sqrt(eigval)

            # Transfer the eigenvector to the aerodynamic surface
            disp = np.zeros(3*self.nnodes)
            self.transfer.setDisplacements(vec)    
            self.transfer.getDisplacements(disp)

            # Compute the normal wash on the aerodynamic mesh
            vk, dk = self.getModeBCs(disp.reshape(self.nnodes, 3))
    
            # Store the normal wash and the surface displacement
            vwash[:,k] = vk
            dwash[:,k] = dk
            modes[:,k] = disp

        # Allocate the eigenvalue at all iterations
        nvals = len(U)
        pvals = np.zeros((nmodes, nvals), dtype=np.complex)

        # Now, evalue the flutter determinant at all iterations
        for kmode in xrange(nmodes):
            for i in xrange(nvals):
                qinf = 0.5*rho*U[i]**2
    
                # Compute an estimate of p based on the lowest natural
                # frequency
                if i == 0:
                    eps = 1e-3
                    p1 = -0.1 + 1j*omega[kmode]
                    p2 = p1 + (eps + 1j*eps)
                elif i == 1:
                    eps = 1e-3
                    p1 = 1.0*pvals[kmode,0]
                    p2 = p1 + (eps + 1j*eps)

                # The following code tries to extrapolate the next
                # point
                elif i == 2:
                    eps = 1e-3
                    p1 = 2.0*pvals[kmode,i-1] - pvals[kmode,i-2]
                    p2 = p1 + (eps + 1j*eps)
                else: 
                    eps = 1e-3
                    p1 = 3.0*pvals[kmode,i-1] - 3.0*pvals[kmode,i-2] + pvals[kmode,i-3]
                    p2 = p1 + (eps + 1j*eps)

                # Compute the flutter determinant
                det1 = self.computeFlutterDeterminant(U[i], p1, qinf, Mach,
                                                      nmodes, omega, 
                                                      vwash, dwash, modes)
                det2 = self.computeFlutterDeterminant(U[i], p2, qinf, Mach,
                                                      nmodes, omega, 
                                                      vwash, dwash, modes)

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
                    det2 = self.computeFlutterDeterminant(U[i], p2, qinf, Mach,
                                                          nmodes, omega, 
                                                          vwash, dwash, modes)
                    
                    # Print out the iteration history for impaitent people
                    if k == 0:
                        print '%4s %10s %10s %10s'%(
                            'Iter', 'Det', 'Re(p)', 'Im(p)') 
                    print '%4d %10.2e %10.6f %10.6f'%(
                        k, abs(det2), p2.real, p2.imag)

                    if abs(det2) < 1e-6*abs(det0):
                        break

                # Store the final value of p
                pvals[kmode, i] = p2

            print '%4s %10s %10s %10s'%(
                'Iter', 'U', 'Re(p)', 'Im(p)')
            print '%4d %10.6f %10.6f %10.6f'%(
                i, U[i], pvals[kmode,i].real, pvals[kmode,i].imag)

        # Return the final values
        return pvals

