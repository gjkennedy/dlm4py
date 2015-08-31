'''
Perform either the Blair or Rodden test cases
'''

import numpy as np
import sys
from dlm4py import DLM

try:
    import matplotlib.pyplot as plt
except:
    pass

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
else:
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
