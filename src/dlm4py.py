'''
This is the python-level interface for the dlm code.
'''

import numpy as np
import sys
from tacs import TACS
from funtofem import FUNtoFEM
from mpi4py import MPI
import dlm

class JDVec:
    def __init__(self, xr, xc=None):
        '''
        The vectors that form the M-orthogonal JD subspace.  Note that
        some of the vectors may just consist of real or complex
        components. From the outside, this vector is designed to
        operate using complex values, but internally, it works with
        separate real and complex components. Each real/complex pair
        is combined using 
        '''
        
        self.xr = xr
        self.xc = xc

        return
    
    def copy(self, vec):
        '''Copy the values from vec to self'''

        # Copy the real components
        self.xr.copyValues(vec.xr)

        # Copy the values from the complex components
        if vec.xc is not None:
            self.xc.copyValues(vec.xc)
        else:
            self.xc.zero()

        return
    
    def dot(self, vec):
        '''
        Compute the dot product
        
        self.x^{H}*vec.x
        = (self.xr - j*self.xc)^{T}*(vec.xr + j*vec.xc) = 
        = (self.xr^{T}*vec.xr + self.xc^{T}*vec.xc) + 
        j*(self.xr^{T}*vec.xc - self.xc^{T}*vec.xr)
        '''

        # Do different things, depending on whether both real and
        # complex vector components are defined
        if (self.xc is None) and (vec.xc is None):
            return self.xr.dot(vec.xr) + 0*1j
        elif (self.xc is None):
            return self.xr.dot(vec.xr) + 1j*self.xr.dot(vec.xc)
        elif (vec.xc is None):
            return self.xr.dot(vec.xr) - 1j*self.xc.dot(vec.xr)

        # Everything is defined, compute the full inner product
        dot = (self.xr.dot(vec.xr) + self.xc.dot(vec.xc)
               + 1j*(self.xr.dot(vec.xc) - self.xc.dot(vec.xr)))
               
        return dot

    def axpy(self, alpha, vec):
        '''
        Add self <- self + alpha*vec

        self.xr + 1j*self.xc += (alphar + 1j*alphac)*(vec.xr + 1j*vec.xc)
        '''

        self.xr.axpy(alpha.real, vec.xr)

        # Add the complex part from alpha
        if alpha.imag != 0.0:
            self.xc.axpy(alpha.imag, vec.xr)

        # Add any contributions from the complex vector components
        if vec.xc is not None:
            self.xc.axpy(alpha.real, vec.xc)

            if alpha.imag != 0.0:
                self.xr.axpy(-alpha.imag, vec.xc)

        return

    def scale(self, alpha):
        '''
        Scale the vector of values by a real scalar
        '''
        self.xr.scale(alpha.real)
        if self.xc is not None:
            self.xc.scale(alpha.real)

        return

    def zero(self):
        '''
        Zero the values in the array
        '''

        self.xr.zeroEntries()
        if self.xc is not None:
            self.xc.zeroEntries()

        return

class GMRES:
    def __init__(self, mat, pc, msub):
        '''
        Initialize the GMRES object for the Jacobi--Davidson method
        '''

        # Copy over the problem definitions
        self.mat = mat
        self.pc = pc
        self.msub = msub

        # Allocate the Hessenberg - this allocates a full matrix
        self.H = np.zeros((self.msub+1, self.msub), dtype=np.complex)

        # Allocate small arrays of size m
        self.res = np.zeros(self.msub+1, dtype=np.complex)

        # Store the normal rotations
        self.Qsin = np.zeros(self.msub, dtype=np.complex)
        self.Qcos = np.zeros(self.msub, dtype=np.complex)

        # Allocate the subspaces
        self.W = []
        self.Z = []

        return

    def solve(self, b, x):
        '''
        Solve the linear system
        '''

        # Perform the initialization: copy over b to W[0] and
        # normalize the result - store the entry in res[0]
        self.W[0].copy(b)
        self.res[0] = np.sqrt(self.W[0].dot(self.W[0]))
        self.W[0].scale(1.0/self.res[0])

        # Perform the matrix-vector products
        for i in xrange(self.msub):
            # Apply the preconditioner
            self.pc.apply(self.W[i], self.Z[i])

            # Compute the matrix-vector product
            self.mat.mult(self.Z[i], self.W[i+1])

            # Perform modified Gram-Schmidt orthogonalization
            for j in xrange(i+1):
                self.H[j,i] = self.W[j].dot(self.W[i+1])
                self.W[i+1].axpy(-self.H[j,i], self.W[j])

            # Compute the norm of the orthogonalized vector and
            # normalize it
            self.H[i+1,i] = np.sqrt(self.W[i+1].dot(self.W[i+1]))
            self.W[i+1].scale(1.0/H[i+1,i])

            # Apply the Givens rotations
            for j in xrange(i):
                h1 = self.H[j, i]
                h2 = self.H[j+1, i]
                self.H[j, i] = h1*self.Qcos[j] + h2*self.Qsin[j]
                self.H[j+1, i] = -h1*self.Qsin[j] + h2*self.Qcos[j]

            # Compute the contribution to the Givens rotation
            # for the current entry
            h1 = self.H[i, i]
            h2 = self.H[i+1, i]

            # Modification for complex from Saad pg. 193
            sq = np.sqrt(abs(h1)**2 + h2*h2)
            self.Qsin[i] = h2/sq
            self.Qcos[i] = h1/sq

            # Apply the newest Givens rotation to the last entry
            self.H[i, i] = h1*self.Qcos[i] + h2*self.Qsin[i]
            self.H[i+1, i] = -h1*self.Qsin[i] + h2*self.Qcos[i]

            # Update the residual
            h1 = res[i]
            res[i] = h1*self.Qcos[i]
            res[i+1] = -h1*self.Qsin[i]

        # Compute the linear combination
        for i in xrange(niters-1, -1, -1):
            for j in xrange(i+1, niters):
                res[i] -= self.H[i, j]*res[j]
            res[i] /= self.H[i, i]

        # Form the linear combination
        x.zero()
        for i in xrange(niters):
            x.axpy(res[i], self.Z[i])

        return niters

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

        # Set placeholder objects for flutter objects
        self.temp = None
        self.Vm = None

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
            self.Xi = np.vstack((self.Xi, Xi))
            self.Xo = np.vstack((self.Xo, Xo))
            self.Xr = np.vstack((self.Xr, Xr))
            self.dXav = np.hstack((self.dXav, dXav))

            # Append the new connectivity and nodes
            self.X = np.vstack((self.X, X))
            self.conn = np.vstack((self.conn, conn + self.nnodes))

        # Set the new number of nodes/panels
        self.npanels = self.Xi.shape[0]
        self.nnodes = self.X.shape[0]

        return

    def computeFlutterMat(self, U, p, qinf, Mach,
                          nvecs, Kr, vwash, dwash, modes):
        '''
        Compute the (reduced) flutter matrix given as follows:

        Fr(p) = p**2*Ir + Kr - qinf*modes^{T}*D^{-1}*wash  
        '''

        # Compute the contribution from the reduced structural problem
        F = np.zeros((nvecs, nvecs), dtype=np.complex)
        F[:,:] = Kr[:,:]

        # Add the term I*p**2 from the M-orthonormal subspace
        for i in xrange(nvecs):
            F[i, i] += p**2

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
            
        return F

    def computeFlutterDet(self, U, p, qinf, Mach,
                          nvecs, Kr, vwash, dwash, modes, omega):
        '''
        Compute the determinant of the reduced flutter matrix:
        
        det Fr(p)
        
        This code first calls the code to compute the flutter matrix,
        then evaluates the determinant.
        '''

        F = self.computeFlutterMat(U, p, qinf, Mach, nvecs, 
                                   Kr, vwash, dwash, modes)
        return np.linalg.det(F)/(omega**(2*nvecs))
    
    
    def computeRigidMat(self, U, rho, Mach, aoa, omega, cref, m, Iyy, xcm, theta_0=0.0, W0=0.0):
        '''
        Compute 'A' matrix for rigid a/c motion
        '''

        g = 9.81
        c = np.cos(theta_0)
        s = np.sin(theta_0)
        
        dF_array = self.computeAeroForceDerivs(U, rho, Mach, cref, aoa, omega, xcm)
        Xu = dF_array[0]
        Xw = dF_array[1]
        Xq = dF_array[2]
        Zu = dF_array[3]
        Zw = dF_array[4]
        Zq = dF_array[5]
        Mu = dF_array[6]
        Mw = dF_array[7]
        Mq = dF_array[8]
        
        a = -U*s + W0*c
        b = -U*s - W0*c
                
        A = np.array([[Xu/m   , Xw/m  , ((Xq/m)-W0), -g*c, 0.0, 0.0],
                      [Zu/m   , Zw/m  , ((Zq/m)+U) , -g*s, 0.0, 0.0],
                      [Mu/Iyy , Mw/Iyy, Mq/Iyy     , 0.0 , 0.0, 0.0],
                      [0.0    , 0.0   , 1.0        , 0.0 , 0.0, 0.0],
                      [c      , s     , 0.0        , a   , 0.0, 0.0],
                      [-s     , c     , 0.0        , b   , 0.0, 0.0]])

        #w, v = np.linalg.eig(A)

        return A #, w, v

    def computeRigidForceVec(self, U, rho, Mach, omega, aoa, m, Iyy, xcm,
                             theta_0=0.0, W0=0.0, w=None):
        '''
        compute force vector for rigid a/c motion
        '''

        g = 9.81
        qinf = 0.5*rho*U**2
        c = np.cos(theta_0)
        s = np.sin(theta_0)

        Cp = self.solve(U, aoa=aoa, omega=omega, Mach=Mach, w=w)
        X, _, Z = self.computeCGForces(qinf, Cp)
        M = self.computeCGMoment(qinf, Cp, xcm)
        
        f = np.array([ (X/m) - g*s + theta_0*g*c,
                       (Z/m) + g*c + theta_0*g*s,
                       M/Iyy,
                       0,
                       theta_0*(U*s - W0*s),
                       theta_0*(-U*c + W0*s)])
        
        return f
    
    def computeAeroForceDerivs(self, U, rho, Mach, cref, aoa, omega, xcm):
        '''
        Compute derivatives of aero forces/moment wrt u, w, q using FD 
        for symmetric longitudinal rigid motion
        '''

        qinf = 0.5*rho*U**2
        du = 1.0e-1
        dw = 1.0e-1
        dq = 1.0e-1
        
        # compute aero forces for +du case
        x = np.array([du, 0.0, 0.0, 0.0, 0.0, 0.0])
        #xdot = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        w = self.computeRigidDownwash(U+du, cref, omega, x, xcm) # update Dtrans i think
        Cp = self.solve(U, aoa=aoa, omega=omega, Mach=Mach, w=w)
        X_pdu, _, Z_pdu = self.computeCGForces(qinf, Cp)
        M_pdu = self.computeCGMoment(qinf, Cp, xcm)

        # compute aero forces for -du case
        x = np.array([-du, 0.0, 0.0, 0.0, 0.0, 0.0])
        #xdot = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        w = self.computeRigidDownwash(U-du, cref, omega, x, xcm)
        Cp = self.solve(U, aoa=aoa, omega=omega, Mach=Mach, w=w)
        X_mdu, _, Z_mdu = self.computeCGForces(qinf, Cp)
        M_mdu = self.computeCGMoment(qinf, Cp, xcm)

        # compute aero forces for +dw case
        x = np.array([0.0, dw, 0.0, 0.0, 0.0, 0.0])
        #xdot = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        w = self.computeRigidDownwash(U, cref, omega, x, xcm)
        Cp = self.solve(U, aoa=aoa, omega=omega, Mach=Mach, w=w)
        X_pdw, _, Z_pdw = self.computeCGForces(qinf, Cp)
        M_pdw = self.computeCGMoment(qinf, Cp, xcm)
        
        # compute aero forces for -dw case
        x = np.array([0.0, -dw, 0.0, 0.0, 0.0, 0.0])
        #xdot = np.array([ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        w = self.computeRigidDownwash(U, cref, omega, x, xcm)
        Cp = self.solve(U, aoa=aoa, omega=omega, Mach=Mach, w=w)
        X_mdw, _, Z_mdw = self.computeCGForces(qinf, Cp)
        M_mdw = self.computeCGMoment(qinf, Cp, xcm)
        
        # compute aero forces for +dq case
        x = np.array([0.0, 0.0, dq, 0.0, 0.0, 0.0])
        #xdot = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        w = self.computeRigidDownwash(U, cref, omega, x, xcm)
        Cp = self.solve(U, aoa=aoa, omega=omega, Mach=Mach, w=w)
        X_pdq, _, Z_pdq = self.computeCGForces(qinf, Cp)
        M_pdq = self.computeCGMoment(qinf, Cp, xcm)
        
        # compute aero forces for -dq case
        x = np.array([0.0, 0.0, -dq, 0.0, 0.0, 0.0])
        #xdot = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        w = self.computeRigidDownwash(U, cref, omega, x, xcm)
        Cp = self.solve(U, aoa=aoa, omega=omega, Mach=Mach, w=w)
        X_mdq, _, Z_mdq = self.computeCGForces(qinf, Cp)
        M_mdq = self.computeCGMoment(qinf, Cp, xcm)

        # perform FD to get dX/du, dZ/du, dM/du
        Xu = (X_pdu - X_mdu)/(2*du) # dX/du
        Zu = (Z_pdu - Z_mdu)/(2*du) # dZ/du
        Mu = (M_pdu - M_mdu)/(2*du) # dM/du

        # perform FD to get dX/dw, dZ/dw, dM/dw
        Xw = (X_pdw - X_mdw)/(2*dw) # dX/dw
        Zw = (Z_pdw - Z_mdw)/(2*dw) # dZ/dw
        Mw = (M_pdw - M_mdw)/(2*dw) # dM/dw

        # perform FD to get dX/dq, dZ/dq, dM/dq
        Xq = (X_pdq - X_mdq)/(2*dq) # dX/dq
        Zq = (Z_pdq - Z_mdq)/(2*dq) # dZ/dq
        Mq = (M_pdw - M_mdq)/(2*dq) # dM/dq

        dF_vec = np.array([Xu, Xw, Xq, Zu, Zw, Zq, Mu, Mw, Mq])
        
        return dF_vec

    def computeElasticMotion(self, U, omega, qinf, Mach,
                            nvecs, Kr, vwash, dwash, modes,
                            W0,
                            aoa=0.0, tol=1e-4, max_iters=100):
        '''
        Compute the forced motion q due to sinusoidal gust
        '''
        # Compute the influence coefficient matrix
        self.computeInfluenceMatrix(U, omega, Mach)

        # Set the mode coefficients
        q = np.zeros(nvecs, dtype=np.complex)
        
        # Compute the boundary condition: -1/U*(dh/dt + U*dh/dx)
        # dwash = -dh/dx, vwash = -dh/dt
        mode_wash = 1j*omega*vwash/U + dwash
        
        aoa_total = 1j*omega*W0/U + aoa

        for k in xrange(max_iters):

            # Compute the wash as a function of time
            wash = aoa_total + np.dot(mode_wash, q)
                                       
            # Solve for the normal wash due to the motion of the wing
            # through the flutter mode
            Cp = np.linalg.solve(self.Dtrans.T, wash)

            # Compute the forces due to the flutter motion
            forces = np.zeros((self.nnodes, 3), dtype=np.complex)

            # Add the forces to the vector
            dlm.addcpforces(qinf, Cp, self.X.T, self.conn.T, forces.T)

            # Compute the aerodynamic forces
            Fa = np.dot(modes.T, forces.flatten())

            A = Kr - omega**2*np.eye(nvecs)

            # Solve for the new values of q
            q = np.linalg.solve(A, Fa)

            #print q
            # if np.absolute((wash-mode_wash)/mode_wash) > tol:
            #    break
                                       

        return q

    def computeRigidMotion(self, U, rho, qinf, Mach, omega,
                           aoa, cref, m, Iyy, xcm,
                           theta_0=0.0, W0=0.0, x0=None, max_iters=10):
        
        if x0 is None:
            x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        A = self.computeRigidMat(U, rho, Mach, aoa, omega, cref, m, Iyy, xcm, theta_0=theta_0, W0=W0)
        f = self.computeRigidForceVec(U, rho, Mach, omega, aoa, m, Iyy, xcm, theta_0=theta_0, W0=W0)
        x = np.linalg.solve(1j*omega*np.eye(6) - A, x0 + f)
        xdot = np.dot(A, x) + f

        for i in range(max_iters):
            w = self.computeRigidDownwash(U, cref, omega, x, xdot, xcm)
            f = self.computeRigidForceVec(U, rho, Mach, omega, aoa, m, Iyy, xcm, theta_0=theta_0, W0=W0, w=w)
            x = np.linalg.solve(1j*omega*np.eye(6) - A, x0 + f)
            xdot = np.dot(A, x) + f
        
        return x, xdot

    def computeFullMotion(self, U, qinf, Mach, omega, m, I, xcm,
                                   aoa, W0, Kr, modes, nvecs,
                                   vwash, dwash, max_iters=10):

        g = 9.81
        # Set the mode coefficients
        q = np.zeros(nvecs, dtype=np.complex)
        
        # Compute the boundary condition: -1/U*(dh/dt + U*dh/dx)
        # dwash = -dh/dx, vwash = -dh/dt
        mode_wash = 1j*omega*vwash/U + dwash
       
        aoa_total = 1j*omega*W0/U + aoa

        for i in xrange(max_iters):

            Cp = self.solve(U, aoa=aoa_total, omega=omega, Mach=Mach)
            forces = self.addAeroForces(qinf, Cp)
            force = np.sum(forces)
            moment = self.computeCGMoment(qinf, Cp, xcm)

            z0 = (-1.0/omega**2)*((force/m)-g)
            a0 = (-1.0/omega**2)*(moment/I)       

            dzdt = 1j*omega*z0/U
            dzdx = a0

            aoa_total = 1j*omega*W0/U + aoa + dzdt/U + a0 # check this
            
            mode_wash = 1j*omega*vwash/U + dwash + dzdt/U + dzdx

            # Compute the wash as a function of time
            wash = aoa_total + np.dot(mode_wash, q)
                                       
            # Solve for the normal wash due to the motion of the wing
            # through the flutter mode
            #Cp = self.solve(U, aoa=aoa, omega=omega, Mach=Mach)

            # Compute the forces due to the flutter motion
            forces = np.zeros((self.nnodes, 3), dtype=np.complex)

            # Add the forces to the vector
            dlm.addcpforces(qinf, Cp, self.X.T, self.conn.T, forces.T)

            # Compute the aerodynamic forces
            Fa = np.dot(modes.T, forces.flatten())

            A = Kr - omega**2*np.eye(nvecs)

            # Solve for the new values of q
            q = np.linalg.solve(A, Fa)

        return q

    def computeStaticLoad(self, aoa, U, qinf, Mach, nvecs,
                          omega, modes, filename=None):
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

    def computeRigidDownwash(self, U, cref, omega, x, xcm, W0=0.0):
        '''
        Compute downwash vector for a given rigid body motion
        where:
        x = [u w q theta x z]
        xdot = (d/dt)x_cm
        '''

        u = x[0]
        w = x[1]
        q = x[2]
        theta = x[3]

        # better way to solve for xdot?
        A = self.computeRigidMat(U, rho, Mach, aoa, omega, cref, m, Iyy, xcm, W0=W0)
        xdot = np.dot(A, x)
        
        zdot = xdot[5]
        
        #k = (cref*omega)/(2*U)
        w = np.zeros(self.npanels, dtype=np.complex)
        #D1jk = np.array([0, 0, 0, 0, 1, 0], dtype=np.complex)
        
        for i in xrange(self.npanels):
            #D2jk = (-2.0/self.dXav[i])*np.array([0, 0, 1, 0, -self.dXav[i]/4.0, 0], dtype=np.complex)
            #w[i] = np.dot((D1jk + 1j*k*D2jk), x_cm)
            xbar = xcm - self.Xr[i, 0]
            w[i] = -zdot - q*xbar - (U+u)*theta
            w[i] += W0*(-1.0 - 1j*(omega/U)*self.Xr[i, 0]) # sinusoidal gust term, check this
            
        return w
    
    def solve(self, U, aoa=0.0, omega=0.0, Mach=0.0, w=None): # aoa not used?
        '''
        Solve the linear system (in the frequency domain)
        '''

        # Compute the influence coefficient matrix
        self.computeInfluenceMatrix(U, omega, Mach)

        if w is None:
            # Evaluate the right-hand-side
            w = np.zeros(self.npanels, dtype=np.complex)
            
            # Compute the normalized downwash
            for i in xrange(self.npanels):
                w[i] = -1.0 - 1j*(omega/U)*self.Xr[i, 0]

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

    def computeCGForces(self, qinf, Cp):
        '''
        Compute total aero forces about cg
        '''

        forces = self.addAeroForces(qinf, Cp)
        x_forces = forces[:, 0::3]
        y_forces = forces[:, 1::3]
        z_forces = forces[:, 2::3]
        X = np.sum(x_forces.flatten())
        Y = np.sum(y_forces.flatten())
        Z = np.sum(z_forces.flatten())
        
        return X, Y, Z

    def computeCGMoment(self, qinf, Cp, xcm):
        '''
        Compute the aero moment about a chordwise location xcm
        '''
        
        forces = self.addAeroForces(qinf, Cp)
        #x_forces = forces[:, 0::3]
        #y_forces = forces[:, 1::3]
        z_forces = forces[:, 2::3]
        x_arm = self.X[:, 0::3] - xcm
        z_forces = z_forces.flatten()
        x_arm = x_arm.flatten()
        y_moments = np.dot(z_forces, x_arm)
        y_moment = np.sum(y_moments)

        return y_moment

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
    
    def testMatDeriv(self, x, dh=1e-6):
        '''
        Test the derivatives within TACS with respect to the mass and
        stiffness matrices 
        '''

        # Set the design variables and create a random perturbation
        # vector
        self.tacs.setDesignVars(x)
        p = np.random.uniform(size=x.shape)

        # Form random vectors
        ur = self.tacs.createVec()
        vr = self.tacs.createVec()
        ur.setRand(-1.0, 1.0)
        vr.setRand(-1.0, 1.0)
        self.tacs.applyBCs(ur)
        self.tacs.applyBCs(vr)

        # Assemble the stiffness and mass matrices
        self.tacs.assembleJacobian(1.0, 0.0, 0.0, None, self.kmat)
        self.tacs.assembleMatType(TACS.PY_MASS_MATRIX,
                                  self.mmat, TACS.PY_NORMAL)

        # Compute the inner product: vr^{T}*K*ur
        self.kmat.mult(ur, self.temp)
        k1 = self.temp.dot(vr)

        self.mmat.mult(ur, self.temp)
        m1 = self.temp.dot(vr)

        # Compute the derivatives w.r.t. the mass/stiffness matrix
        krr = np.zeros(x.shape)
        mrr = np.zeros(x.shape)
        mtype = TACS.PY_STIFFNESS_MATRIX
        self.tacs.addMatDVSensInnerProduct(1.0, mtype, vr, ur, krr)
        mtype = TACS.PY_MASS_MATRIX
        self.tacs.addMatDVSensInnerProduct(1.0, mtype, vr, ur, mrr)
        
        # Evaluate the stiffness matrix at the new point
        xnew = x + dh*p
        self.tacs.setDesignVars(xnew)

        # Assemble the stiffness and mass matrices
        self.tacs.assembleJacobian(1.0, 0.0, 0.0, None, self.kmat)
        self.tacs.assembleMatType(TACS.PY_MASS_MATRIX,
                                  self.mmat, TACS.PY_NORMAL)
        
        # Compute the inner product: vr^{T}*K*ur
        self.kmat.mult(ur, self.temp)
        k2 = self.temp.dot(vr)

        self.mmat.mult(ur, self.temp)
        m2 = self.temp.dot(vr)

        # Form the approximate directional derivatives
        fdk = (k2 - k1)/dh
        fdm = (m2 - m1)/dh

        # Compute the direction derivatives
        pdk = np.dot(krr, p)
        pdm = np.dot(mrr, p)

        print '          %12s %12s %12s %12s'%(
            'Deriv', 'FD', 'Rel', 'Abs')
        print 'Stiffness %12.4e %12.4e %12.4e %12.4e'%(
            pdk, fdk, abs((fdk - pdk)/pdk), abs(fdk - pdk))
        print 'Mass      %12.4e %12.4e %12.4e %12.4e'%(
            pdm, fdm, abs((fdm - pdm)/pdm), abs(fdm - pdm))

        return

    def initStructure(self, tacs):
        '''
        Set up the load and displacement transfer object for a general
        TACS finite-element model.
        '''

        # Get the communicator from the TACSAssembler object
        self.tacs = tacs
        comm = MPI.COMM_WORLD

        # Now, set up the load and displacement transfer object
        struct_root = 0
        aero_root = 0

        # Get the aerodynamic mesh connectivity
        aero_pts = self.X.flatten()
        aero_conn = self.conn.flatten()

        # Specify the load/displacement transfer data
        isymm = -1
        self.funtofem = FUNtoFEM.pyFUNtoFEM(comm, comm, struct_root,
                                            comm, aero_root,
                                            FUNtoFEM.PY_LINEAR, isymm)
        self.funtofem.setAeroNodes(aero_pts)

        # Set the structural points into FUNtoFEM
        X = self.tacs.createNodeVec()
        self.tacs.getNodes(X)
        self.funtofem.setStructNodes(X.getArray())

        # Initialize the load/displacement transfer
        num_nearest = 25
        self.funtofem.initialize(num_nearest)
            
        # Set up the matrices/pc/Krylov solver that will be required
        # for the flutter analysis
        self.mat = tacs.createFEMat()

        # The raw stiffness/mass matrices 
        self.kmat = tacs.createFEMat()
        self.mmat = tacs.createFEMat()

        # Create the preconditioner and the solver object.  Note that
        # these settings are best for a shell-type finite-element
        # model.
        self.pc = TACS.Pc(self.mat)

        # Create the GMRES object 
        gmres_iters = 10
        nrestart = 0
        self.ksm = TACS.KSM(self.mat, self.pc, gmres_iters, nrestart)

        # Create a temporary tacs vector
        self.temp = self.tacs.createVec()

        return

    def setUpSubspace(self, m, r, sigma=0.0, tol=1e-12,
                      max_iters=5, use_modes=False):
        '''
        Build a subspace for the flutter analysis using a Lanczos
        method. You can specify to either use the Lanczos subspace
        basis or use the eigenvector basis.

        Input:
        m:         the size of the Lanczos subspace
        r:         the number of eigenvectors that must converge
        sigma:     estimate of the frequency 
        tol:       tolerance for the eigenvector solution
        max_iters: maximum number of iterations to use
        use_modes: reduce the subspace to the eigenvectors
        '''

        # Assemble the mass and stiffness matrices
        self.tacs.assembleJacobian(1.0, 0.0, 0.0, None, self.kmat)
        self.tacs.assembleMatType(TACS.PY_MASS_MATRIX,
                                  self.mmat, TACS.PY_NORMAL)
        
        # Create a list of vectors
        if self.Vm is None:
            self.Vm = []

        # Allocate the Lanczos subspace vectors
        if len(self.Vm) < m:
            lvm = len(self.Vm)
            for i in xrange(lvm, m):
                self.Vm.append(self.tacs.createVec())

        # Initialize Vm as a random set of initial vectors
        self.Vm[0].setRand(-1.0, 1.0)

        # Iterate until we have sufficient accuracy
        eigvecs = np.zeros((m-1, m-1))
        for i in xrange(max_iters):
            alpha, beta = self.lanczos(self.Vm, sigma)

            print alpha, beta

            # Compute the final coefficient
            b0 = beta[-1]

            # Compute the eigenvalues and eigenvectors
            info = dlm.tridiageigvecs(alpha, beta, eigvecs.T)
            
            if info != 0:
                print 'Error in the tri-diagonal eigenvalue solver'

            # Compute the true eigenvalues/vectors
            omega = np.sqrt(1.0/alpha + sigma)
            
            # Argsort the array, and test for convergence of the
            # r-lowest eigenvalues
            indices = np.argsort(omega)
            omega = omega[indices]
            eigvecs = eigvecs[indices,:]

            convrg = True
            for j in xrange(r):
                if np.fabs(b0*eigvecs[j,-1]) > tol:
                    convrg = False

            if convrg:
                break
            else:
                # Make a better guess for sigma
                sigma = 0.95*omega[0]**2
                
                # Form a linear combination of the best r eigenvectors
                weights = np.sum(eigvecs[:r,:], axis=0)
                self.Vm[0].scale(weights[0])
                for j in xrange(m-1):
                    self.Vm[0].axpy(weights[j], self.Vm[j])

                self.tacs.applyBCs(Vm[0])

        # Now that we've built Vm, compute the inner product with the
        # K matrix for later useage
        if use_modes:
            self.Qm = []
            for i in xrange(r):
                # Normalize the eigenvectors so that they remain 
                # M-orthonormal
                eigvecs[i,:] /= np.sqrt(np.sum(eigvecs[i,:]**2))
            
                # Compute the full eigenvector
                qr = self.tacs.createVec()
                for j in xrange(m-1):
                    qr.axpy(eigvecs[i,j], self.Vm[j])
                self.Qm.append(qr)

            # Set the values of the stiffness matrix
            self.Kr = np.zeros((r,r))

            # Set the values of stiffness
            for k in xrange(r):
                self.Kr[k,k] = omega[k]**2
        else:
            # Set the stiffness matrix
            self.Kr = np.zeros((m,m))

            for i in xrange(m):
                self.kmat.mult(self.Vm[i], self.temp)
                for j in xrange(i+1):
                    self.Kr[i,j] = self.temp.dot(self.Vm[j])
                    self.Kr[j,i] = self.Kr[i,j]

            # Set the Qm as the subspace
            self.Qm = self.Vm

        # Get the surface modes and the corresponding normal wash
        self.Qm_modes = np.zeros((3*self.nnodes, len(self.Qm)))
        self.Qm_vwash = np.zeros((self.npanels, len(self.Qm)))
        self.Qm_dwash = np.zeros((self.npanels, len(self.Qm)))

        # Extract the natural frequencies of vibration
        for k in xrange(len(self.Qm)):
            # Transfer the eigenvector to the aerodynamic surface
            self.funtofem.transferDisps(self.Qm[k].getArray())
            disp = self.funtofem.getAeroDisps()

            # Compute the normal wash on the aerodynamic mesh
            vk, dk = self.getModeBCs(disp.reshape(self.nnodes, 3))
    
            # Store the normal wash and the surface displacement
            self.Qm_vwash[:,k] = vk
            self.Qm_dwash[:,k] = dk
            self.Qm_modes[:,k] = disp

        # Set the values of omega
        self.omega = omega[:r]

        print 'omega = ', self.omega[:r]

        return

    def lanczos(self, Vm, sigma):
        '''
        Build an M-orthogonal Lanczos subspace using full
        orthogonalization. The full-orthogonalization makes this
        equivalent to Arnoldi, but only the tridiagonal coefficients
        are retained.

        Input:
        Vm:     list of vectors empty vectors except for Vm[0]
        sigma:  estimate of the first natural frequency

        Output:
        Vm:     an M-orthogonal subspace
        '''

        # Allocate space for the symmetric tri-diagonal system
        alpha = np.zeros(len(Vm)-1)
        beta = np.zeros(len(Vm)-1)

        # Compute (K - sigma*M)
        self.mat.copyValues(self.kmat)
        self.mat.axpy(-sigma, self.mmat)
        
        # Factor the stiffness matrix (K - sigma*M)
        self.pc.factor()

        # Apply the boundary conditions to make sure that the 
        # initial vector satisfies them
        self.tacs.applyBCs(Vm[0])

        # Scale the initial vector
        self.mmat.mult(Vm[0], self.temp)
        b0 = np.sqrt(Vm[0].dot(self.temp))
        Vm[0].scale(1.0/b0)

        # Execute the orthogonalization
        for i in xrange(len(Vm)-1):
            # Compute V[i+1] = (K - sigma*M)^{-1}*M*V[i]
            self.mmat.mult(Vm[i], self.temp)
            self.ksm.solve(self.temp, Vm[i+1])
            
            # Make sure that the boundary conditions are enforced
            # fully
            self.tacs.applyBCs(Vm[i+1])

            # Perform full modified Gram-Schmidt orthogonalization
            # with mass-matrix inner products
            for j in xrange(i, -1, -1):
                # Compute the inner product
                self.mmat.mult(Vm[i+1], self.temp)
                h = Vm[j].dot(self.temp)
                Vm[i+1].axpy(-h, Vm[j])

                if i == j:
                    alpha[i] = h
            
            # Compute the inner product w.r.t. itself
            self.mmat.mult(Vm[i+1], self.temp)
            beta[i] = np.sqrt(Vm[i+1].dot(self.temp))
            Vm[i+1].scale(1.0/beta[i])

        return alpha, beta

    def computeFrozenDeriv(self, rho, Uval, Mach, p, 
                           num_design_vars, ortho_check=True):
        '''
        Compute the frozen derivative: First, find the (approx) left
        and right eigenvectors associated with the solution. This
        involves assembling the matrix:
        
        Fr(p) = p^2*Ir + Kr - qinf*Ar(p).

        Then finding the left- and right-eigenvectors associated with
        the eigenvalue closest to zero.
        '''

        # Evaluate the flutter matrix
        qinf = 0.5*rho*Uval**2
        Fr = self.computeFlutterMat(Uval, p, qinf, Mach, len(self.Qm), 
                                    self.Kr, self.Qm_vwash, self.Qm_dwash, 
                                    self.Qm_modes)

        # Duplicate the values stored in the matrix Fr
        Fr_destroyed = np.array(Fr)

        # Compute all of the left- and right- eigenvectors
        m = len(self.Qm)
        eigs = np.zeros(m, dtype=np.complex) 
        Zl = np.zeros((m, m), dtype=np.complex) 
        Zr = np.zeros((m, m), dtype=np.complex) 
        dlm.alleigvecs((Fr.transpose()).T, eigs, Zl.T, Zr.T)

        # Determine what vectors we should use - this is an educated
        # guess, the smallest eigenvalue/eigenvector triplet 
        k = np.argmin(abs(eigs))
        zl = Zl[k,:]
        zr = Zr[k,:]

        # Using the eigenvectors compute the real/complex left
        # eigenvectors
        vr = self.tacs.createVec()
        vc = self.tacs.createVec()
        for i in xrange(m):
            vr.axpy(zl[i].real, self.Qm[i])
            vc.axpy(zl[i].imag, self.Qm[i])

        # Compute the linear combination for the right eigenvector
        ur = self.tacs.createVec()
        uc = self.tacs.createVec()
        for i in xrange(m):
            ur.axpy(zr[i].real, self.Qm[i])
            uc.axpy(zr[i].imag, self.Qm[i])

        # Do an error check here - is this any good???
        if ortho_check:
            self.mmat.mult(vr, self.temp)
            err = ur.dot(self.temp) + 1j*self.temp.dot(uc)
            self.mmat.mult(vc, self.temp)
            err += uc.dot(self.temp) + 1j*self.temp.dot(ur)
            err -= 1.0
            print 'Orthogonality error ', err

        # Compute all of the derivatives
        krr = np.zeros(num_design_vars)
        krc = np.zeros(num_design_vars)
        kcr = np.zeros(num_design_vars)        
        kcc = np.zeros(num_design_vars)
        mtype = TACS.PY_STIFFNESS_MATRIX
        self.tacs.addMatDVSensInnerProduct(1.0, mtype, vr, ur, krr)
        self.tacs.addMatDVSensInnerProduct(1.0, mtype, vc, ur, kcr)
        self.tacs.addMatDVSensInnerProduct(1.0, mtype, vr, uc, krc)
        self.tacs.addMatDVSensInnerProduct(1.0, mtype, vc, uc, kcc)

        # Compute all of the derivatives
        mrr = np.zeros(num_design_vars)
        mrc = np.zeros(num_design_vars)
        mcr = np.zeros(num_design_vars)        
        mcc = np.zeros(num_design_vars)
        mtype = TACS.PY_MASS_MATRIX
        self.tacs.addMatDVSensInnerProduct(1.0, mtype, vr, ur, mrr)
        self.tacs.addMatDVSensInnerProduct(1.0, mtype, vc, ur, mcr)
        self.tacs.addMatDVSensInnerProduct(1.0, mtype, vr, uc, mrc)
        self.tacs.addMatDVSensInnerProduct(1.0, mtype, vc, uc, mcc)
        
        # Evaluate the (approximate) derivative of F(p) w.r.t. p
        dh = 1e-5
        F1 = self.computeFlutterMat(Uval, p + dh, qinf, Mach, len(self.Qm), 
                                    self.Kr, self.Qm_vwash, self.Qm_dwash, 
                                    self.Qm_modes)
        F2 = self.computeFlutterMat(Uval, p - dh, qinf, Mach, len(self.Qm), 
                                    self.Kr, self.Qm_vwash, self.Qm_dwash, 
                                    self.Qm_modes)
        dFdp = 0.5*((F1.real - F2.real)/dh + 1j*(F1.imag - F2.imag)/dh)

        # Compute the inner product of the left and right reduced
        # eigenvectors
        fact = np.dot(zl.conjugate(), np.dot(dFdp, zr))

        # Evaluate the entire derivative
        deriv = -(p**2*((mrr + mcc) + 1j*(mrc - mcr)) + 
                  (krr + kcc) + 1j*(krc - kcr))/fact

        return deriv

    def computeFlutterMode(self, rho, Uval, Mach, 
                           kmode, pinit=None, 
                           max_iters=20, tol=1e-12):
        '''
        Given the density, velocity and Mach number, compute the
        damping/frequency of the k-th mode using Hassig's determinant
        iteration technique.
        '''

        # Provide an initial estimate of the frequency
        if pinit is None:
            p1 = -1.0 + 1j*self.omega[kmode]
            p2 = -1.0 + 1j*(1e-3 + self.omega[kmode])
        else:
            p1 = 1.0*pinit
            p2 = 1.0*pinit + 1j*1e-3

        # Compute the dynamic pressure
        qinf = 0.5*rho*Uval**2

        # Compute the flutter determinant
        det1 = self.computeFlutterDet(Uval, p1, qinf, Mach,
                                      len(self.Qm), self.Kr,
                                      self.Qm_vwash, self.Qm_dwash, 
                                      self.Qm_modes, self.omega[kmode])
        det2 = self.computeFlutterDet(Uval, p2, qinf, Mach,
                                      len(self.Qm), self.Kr,
                                      self.Qm_vwash, self.Qm_dwash, 
                                      self.Qm_modes, self.omega[kmode])

        # Perform the flutter determinant iteration
        det0 = 1.0*det1
        for k in xrange(max_iters):
            # Compute the new value of p
            pnew = (p2*det1 - p1*det2)/(det1 - det2)

            # Move p2 to p1
            p1 = 1.0*p2
            det1 = 1.0*det2

            # Move pnew to p2 and compute pnew
            p2 = 1.0*pnew
            det2 = self.computeFlutterDet(Uval, p2, qinf, Mach,
                                          len(self.Qm), self.Kr,
                                          self.Qm_vwash, self.Qm_dwash, 
                                          self.Qm_modes, self.omega[kmode])
                    
            # Print out the iteration history for impaitent people
            if k == 0:
                print '%4s %10s %15s %15s'%(
                    'Iter', 'Det', 'Re(p)', 'Im(p)') 
            print '%4d %10.2e %15.10f %15.10f'%(
                k, abs(det2), p2.real, p2.imag)

            if abs(det2) < tol*abs(det0):
                break

        return p2

    def computeFlutterModeEig(self, rho, Uval, Mach, 
                              kmode, pinit=None, 
                              max_iters=20, tol=1e-8, dh=1e-5):
        '''
        Given the density, velocity, and Mach number, compute the
        frequency/damping of the k-th flutter mode using an eigenvalue
        iteration procedure. This code attempts to use a secant method
        on the k-th eigenvalue. It may fail to converge if there are
        cross-over modes at the given point, but it has advantages
        over the determinant iteration technique.

        The code solves the problem:

        eig_{k}(F(p)) = 0
        d(eig_{k})/dp = - zl^{H}*dF/dp*zr
        '''

        # Provide an initial estimate of the frequency
        if pinit is None:
            p = -1.0 + 1j*self.omega[kmode]
        else:
            p = 1.0*pinit

        # Compute the dynamic pressure
        qinf = 0.5*rho*Uval**2

        # Allocate space fot the eigenvalue problem
        m = len(self.Qm)
        eigs = np.zeros(m, dtype=np.complex) 
        Zl = np.zeros((m, m), dtype=np.complex) 
        Zr = np.zeros((m, m), dtype=np.complex) 

        # Iterate until convergence
        for i in xrange(max_iters):
            # Compute the flutter matrix at the current point
            F1 = self.computeFlutterMat(Uval, p, qinf, Mach,
                                        m, self.Kr, self.Qm_vwash, 
                                        self.Qm_dwash, self.Qm_modes)
            
            # Solve the eigenvalue problem
            dlm.alleigvecs((F1.transpose()).T, eigs, Zl.T, Zr.T)

            # F(p)*u = eig*u
            # dF/dp*u + F(p)*du/dp = d(eig)/dp*u + eig*du/dp
            # v^{H}*dF/dp = d(eig)/dp

            # Determine what eigenvector to use - sort modes by
            # frequency
            k = np.argsort(abs(eigs))[kmode]

            # Print out the iteration history for impaitent people
            if i == 0:
                print '%4s %10s %15s %15s'%(
                    'Iter', 'Eig', 'Re(p)', 'Im(p)') 
            print '%4d %10.2e %15.10f %15.10f'%(
                i, abs(eigs[k]), p.real, p.imag)

            if abs(eigs[k]) < tol:
                return p

            # Extract the associated eigenvectors
            zl = Zl[k,:]
            zr = Zr[k,:]

            # Compute the derivative of the flutter matrix w.r.t. p
            F2 = self.computeFlutterMat(Uval, p + dh, qinf, Mach,
                                        m, self.Kr, self.Qm_vwash, 
                                        self.Qm_dwash, self.Qm_modes)
            dFdp = (F2 - F1)/dh
            
            # Compute the derivative of the eigenvalue
            deigdp = np.dot(zl.conjugate(), np.dot(dFdp, zr)) 

            # Apply Newton's method
            p -= eigs[k]/deigdp

        return p

    def velocitySweep(self, rho, Uvals, Mach, nmodes):
        '''
        Use the basis stored in Qm to perform a sweep of the
        velocities.
        '''

        # Allocate the eigenvalue at all iterations
        nvals = len(Uvals)
        pvals = np.zeros((nmodes, nvals), dtype=np.complex)

        # Now, evalue the flutter determinant at all iterations
        for kmode in xrange(nmodes):
            for i in xrange(nvals):
                qinf = 0.5*rho*Uvals[i]**2
    
                # Compute an estimate of p based on the lowest natural
                # frequency
                if i == 0:
                    eps = 1e-3
                    p1 = -0.1 + 1j*self.omega[kmode]
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
                    p1 = (3.0*pvals[kmode,i-1] - 
                          3.0*pvals[kmode,i-2] + pvals[kmode,i-3])
                    p2 = p1 + (eps + 1j*eps)

                # Compute the flutter determinant
                det1 = self.computeFlutterDet(Uvals[i], p1, qinf, Mach,
                                              len(self.Qm), self.Kr,
                                              self.Qm_vwash, self.Qm_dwash, 
                                              self.Qm_modes, self.omega[kmode])
                det2 = self.computeFlutterDet(Uvals[i], p2, qinf, Mach,
                                              len(self.Qm), self.Kr,
                                              self.Qm_vwash, self.Qm_dwash, 
                                              self.Qm_modes, self.omega[kmode])

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
                    det2 = self.computeFlutterDet(U[i], p2, qinf, Mach,
                                                  len(self.Qm), self.Kr,
                                                  self.Qm_vwash, self.Qm_dwash, 
                                                  self.Qm_modes, self.omega[kmode])
                    
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

