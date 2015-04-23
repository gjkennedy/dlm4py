'''
The kernel of the DLM method is the evaluation of integrals that are
required to determine the influence of a sending point on a receiving
point. These integrals involve non-integrable functions that are
evaluated approximately several times for each panel.  The purpose of
this code is to test the accuracy of the implementation of these
integrals and check for bugs.

The integrals are:

I0 = int_{u1}^{infty} (1 - u/sqrt(1 + u**2))*exp(-i*k1*u) du
J0 = int_{u1}^{infty} u*(1 - u/sqrt(1 + u**2))*exp(-i*k1*u) du

I1 = int_{u1}^{infty} exp(-i*k1*u)/(1 + u**2)**(3/2) du
I2 = int_{u1}^{infty} exp(-i*k1*u)/(1 + u**2)**(5/2) du
'''

import numpy as np
import dlm

# Compute an approximation of these integrals
u1 = -1.0
k1 = 4.0

# Compute the integral values using the trapezoid rule
I1 = 0.0
I2 = 0.0
I0 = 0.0
J0 = 0.0

n = 1000000
u = np.linspace(u1, 1000.0, n)
expk = np.exp(-1j*k1*u)
invsqrt = 1.0/np.sqrt(1.0 + u**2)

for j in xrange(n-1):
    uav = 0.5*(u[j+1] + u[j])    
    I1 += 0.5*(u[j+1] - u[j])*(expk[j+1]*invsqrt[j+1]**3 + expk[j]*invsqrt[j]**3)
    I2 += 0.5*(u[j+1] - u[j])*(expk[j+1]*invsqrt[j+1]**5 + expk[j]*invsqrt[j]**5)

    I0 += 0.5*(u[j+1] - u[j])*((1.0 - u[j+1]*invsqrt[j+1])*expk[j+1] + 
                               (1.0 - u[j]*invsqrt[j])*expk[j])
    J0 += 0.5*(u[j+1] - u[j])*(u[j+1]*(1.0 - u[j+1]*invsqrt[j+1])*expk[j+1] + 
                               u[j]*(1.0 - u[j]*invsqrt[j])*expk[j])

# Evaluate the integrals using the considerably faster approximation
# used in the DLM method

if u1 < 0.0:
    I0_approx, J0_approx = dlm.approxkernelintegrals(-u1, k1)

    # Compute I1(-u1, k1) and I2(-u1, k1)
    I1_neg = (1.0 + u1/np.sqrt(1.0 + u1**2))*np.exp(1j*k1*u1) - 1j*k1*I0_approx
    I2_neg = (((2.0 - 1j*k1*u1)*(1.0 + u1/np.sqrt(1.0 + u1**2)) 
               + u1/(1.0 + u1**2)**(1.5))*np.exp(1j*k1*u1)
              - 1j*k1*I0_approx + k1**2*J0_approx)

    I0_approx, J0_approx = dlm.approxkernelintegrals(0.0, k1)

    # Compute I1(-u1, k1) and I2(-u1, k1)
    I1_0 = 1.0 - 1j*k1*I0_approx
    I2_0 = 2.0 - 1j*k1*I0_approx + k1**2*J0_approx

    I1_approx = 2.0*I1_0.real - I1_neg.real + 1j*I1_neg.imag
    I2_approx = 2.0*I2_0.real - I2_neg.real + 1j*I2_neg.imag

else:
    I0_approx, J0_approx = dlm.approxkernelintegrals(u1, k1)
    
    I1_approx = (1.0 - u1/np.sqrt(1.0 + u1**2))*np.exp(-1j*k1*u1) - 1j*k1*I0_approx
    I2_approx = (((2.0 + 1j*k1*u1)*(1.0 - u1/np.sqrt(1.0 + u1**2)) 
                  - u1/(1.0 + u1**2)**(1.5))*np.exp(-1j*k1*u1)
                 - 1j*k1*I0_approx + k1**2*J0_approx)

if u1 > 0.0:
    # Note: the test only works for I0/J0 when u1 >= 0
    print 'I0        ', I0
    print 'I0_approx ', I0_approx
    print 'I0 - I0_approx ', I0 - I0_approx
    print ' '
    print 'J0        ', J0
    print 'J0_approx ', J0_approx
    print 'J0 - J0_approx ', J0 - J0_approx
    print ' '

print 'I1:        ', I1
print 'I1_approx: ', I1_approx
print 'I1 - I1_approx: ', I1 - I1_approx
print ' '
print 'I2:        ', 3*I2
print 'I2_approx: ', I2_approx
print 'I2 - I2_approx: ', 3*I2 - I2_approx
