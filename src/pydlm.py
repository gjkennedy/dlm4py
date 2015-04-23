'''
This is the python-level interface for the dlm code. 

'''

import numpy as np
import dlm

class DLM:
    def __init__(self, is_symmetric=1):
        '''
        Initialize the internal mesh.
        '''

        self.Xi = None
        self.Xo = None
        self.Xr = None
        self.dXav = None
        self.is_symmetric = is_symmetric

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

        dlm.computeinputmeshsegment(n, m, x0, span, dihedral, sweep,
                                    root_chord, taper_ratio, 
                                    Xi.T, Xo.T, Xr.T, dXav)

        # Append the new mesh components
        if self.Xi is None:
            self.Xi = Xi
            self.Xo = Xo
            self.Xr = Xr
            self.dXav = dXav
        else:
            self.Xi = np.vstack(self.Xi, Xi)
            self.Xo = np.vstack(self.Xo, Xo)
            self.Xr = np.vstack(self.Xr, Xr)
            self.dXav = np.vstack(self.dXav, dXav)

        return
    
    def solve(self, U, aoa=0.0, omega=0.0, M=0.0):
        '''
        Solve the linear system (in the frequency domain)
        '''

        if self.Xi is None:
            return

        # Determine the number of panels
        npanels = self.Xi.shape[0]

        # Allocate the influence coefficient matrix
        Dtrans = np.zeros((npanels, npanels), dtype=np.complex)

        print 'compute influence'
        # Compute the influence coefficients
        dlm.computeinfluencematrix(Dtrans.T, omega, U, M,
                                   self.Xi.T, self.Xo.T, self.Xr.T, self.dXav,
                                   self.is_symmetric)
        
        # Evaluate the right-hand-side
        w = np.zeros(npanels, dtype=np.complex)
        
        # dlm.computeperiodicbc(w, aoa, omega, U, self.Xi.T, self.Xo.T)
        w[:] = -1j

        Cp = np.linalg.solve(Dtrans.T, w)

        return Cp

solver = DLM(is_symmetric=1)
nspan = 3
nchord = 3
chord = 12
span = 12
taper = 1.0
sweep = 0.0
solver.addMeshSegment(nspan, nchord, span, chord,
                      sweep=sweep, taper_ratio=taper)

U = 10.0
kreduce = 1.0 # Reduced frequency
b = 6.0 # Reference length
omega = U*kreduce/b # The actual frequency
M = 0.5 # Mach number

aoa = 0.0
Cp = solver.solve(U, aoa=aoa, omega=omega, M=M)

for i in xrange(len(Cp)):
    print 'Cpr[%d] = %e'%(i, Cp[i].real)

for i in xrange(len(Cp)):
    print 'Cpi[%d] = %e'%(i, Cp[i].imag)

fp = open('Cp.dat', 'w')
fp.write('Title = \"Solution\"\n')
fp.write('Variables = X, Y, Z, Re(Cp), Im(Cp)\n')
fp.write('Zone T=wing n=%d e=%d '%(
        (nspan+1)*(nchord+1), nspan*nchord))
fp.write('datapacking=block ')
fp.write('zonetype=fequadrilateral ')
fp.write('varlocation=([4,5]=cellcentered)\n')

u = np.linspace(0, 1, nchord+1)
y = np.linspace(0, span, nspan+1)

for i in xrange(nspan+1):
    for j in xrange(nchord+1):
        c = chord*(1.0 - (1.0 - taper)*y[i]/span)*((j + 0.75)/nchord - 0.25)
        fp.write('%e\n'%(y[i]*np.tan(sweep) + c))
for i in xrange(nspan+1):
    for j in xrange(nchord+1):
        fp.write('%e\n'%(y[i]))
for i in xrange(nspan+1):
    for j in xrange(nchord+1):
        fp.write('%e\n'%(0.0))
for i in xrange(nspan):
    for j in xrange(nchord):
        fp.write('%e\n'%(Cp[j + i*nchord].real))
for i in xrange(nspan):
    for j in xrange(nchord):
        fp.write('%e\n'%(Cp[j + i*nchord].imag))

for i in xrange(nspan):
    for j in xrange(nchord):
        fp.write('%d %d %d %d\n'%(
                j+1 + i*(nchord+1), j+2 + i*(nchord+1),
                j+2 + (i+1)*(nchord+1), j+1 + (i+1)*(nchord+1)))
fp.close()
