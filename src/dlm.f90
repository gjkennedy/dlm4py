! The following code is a basic Doublet Lattice method implementation
! that is designed to be run from python. The computationally
! intensive computations are performed at the fortran level for
! efficiency. This code is intended to be used for flutter
! computations but could also be used for general subsonic oscilatory
! flows.

module precision
  ! A module to handle setting the data type for the real/complex analysis
  integer, parameter :: dtype = selected_real_kind(12)
end module precision

subroutine approxKernelIntegrals(I0, J0, u1, k1)
  ! Compute the approximate values of the integrals I0 and J0. These
  ! integrals are required for the computation of the kernel function
  ! at points along the bound vortex line. The integrals are
  ! approximated based on the following expression from Desmarais:
  !
  ! 1 - u/sqrt(1 + u^2) \approx \sum_{n} a_{n} exp(-pn*u)
  !
  ! Where pn = b*2**n. The output I0 and J0 are defined as follows:
  !
  ! I0 = int_{u1}^{infty} (1 - u/sqrt(1 + u^2)) du
  ! J0 = int_{u1}^{infty} u (1 - u/sqrt(1 + u^2)) du
  !
  ! The function input/output are defined as follows:
  ! 
  ! Input:
  ! u1:  (M*R - x0)/(beta^2*x0)
  ! k1:  omega*r1/U
  !
  ! Output:
  ! I0:  Approximate value of the integral I0
  ! J0:  Approximate value of the integral J0 
  
  use precision
  implicit none
  real(kind=dtype), intent(in) :: u1, k1
  complex(kind=dtype), intent(out) :: I0, J0
  integer :: n
  real(kind=dtype) :: pn, b, a(12)
  complex(kind=dtype) :: expn, invn, kval

  ! Set the values of the constants requried for the evaluation of the
  ! approximate integrals
  a(1) = 0.000319759140_dtype
  a(2) = -0.000055461471_dtype
  a(3) = 0.002726074362_dtype
  a(4) = 0.005749551566_dtype
  a(5) = 0.031455895072_dtype
  a(6) = 0.106031126212_dtype
  a(7) = 0.406838011567_dtype
  a(8) = 0.798112357155_dtype
  a(9) = -0.417749229098_dtype
  a(10) = 0.077480713894_dtype
  a(11) = -0.012677284771_dtype
  a(12) = 0.001787032960_dtype

  ! Set the parameter b - the expontential in the kernel integral
  b = 0.009054814793_dtype

  ! Evaluate the integral for I0 and J0
  I0 = cmplx(0.0, 0.0, kind=dtype)
  J0 = cmplx(0.0, 0.0, kind=dtype)

  do n = 1, 12
     pn = b*(2**n)
     kval = cmplx(pn, k1, kind=dtype)
     expn = exp(-kval*u1)
     invn = 1.0/kval
     I0 = I0 + a(n)*expn*invn
     J0 = J0 + a(n)*expn*(kval*u1 + cmplx(1.0, 0.0, kind=dtype))*invn*invn
  end do

end subroutine approxKernelIntegrals

subroutine evalK1K2Coeff(Kf1, Kf2, r1, u1, k1, beta, R, M)
  ! Compute the value of the K1 and K2 functions, given the values of
  ! the local panel variables. This code calls the function
  ! approxKernelIntegrals to obtain the values of I0 and J0 which are
  ! used in the evaluation of the kernel.
  !
  ! Input:
  ! u1:   (M*R - x0)/(beta^2*x0)
  ! k1:   omega*r1/U
  ! beta: sqrt(1.0 - M**2)
  ! R:    sqrt(x0**2 + (beta*r1)**2)
  ! M:    Mach number
  !
  ! Output:
  ! K1:   first kernel function
  ! K2:   second kernel function
  
  use precision
  implicit none
  real(kind=dtype), intent(in) :: r1, u1, k1, beta, R, M
  complex(kind=dtype), intent(out) :: Kf1, Kf2
  
  ! Local temporary variables
  real(kind=dtype) :: invsqrt, invR
  complex(kind=dtype) :: expk, I0, J0, I1, I2

  ! Constant definitions
  complex(kind=dtype) :: I = cmplx(0.0, 1.0, kind=dtype)
  real(kind=dtype) :: one = 1.0_dtype
  real(kind=dtype) :: two = 2.0_dtype

  ! Compute the temporary variable values that will be used below
  invsqrt = one/sqrt(one + u1*u1)
  invR = one/R
  expk = exp(-I*k1*u1)

  ! Evaluate the approximate integrals I0 and J0
  call approxKernelIntegrals(I0, J0, u1, k1)

  ! Evaluate I1
  I1 = (one - u1*invsqrt)*expk - I*k1*I0

  ! Evaluate 3*I2
  I2 = ((two + I*k1*u1)*(one - u1*invsqrt) - u1*invsqrt**3)*expk &
       - I*k1*I0 + k1**2*J0

  ! Compute the first component of the kernel function
  Kf1 = I1 + M*r1*invR*invsqrt*expk

  ! Compute the second component of the kernel function
  Kf2 = -I2 - I*k1*invsqrt*(M*r1*invR)**2 &
       - (M*r1*invR)*((one + u1**2)*(beta*r1*invR)**2 + &
       two + M*r1*u1*invR)*expk*invsqrt**3

end subroutine evalK1K2Coeff

subroutine evalKernel(Kf1, Kf2, omega, U, beta, M, &
     x0, y0, z0, cosr, sinr, coss, sins)
  ! Evaluate the kernel function 
  ! 
  ! Input:
  ! omega:      the frequency of oscillation
  ! x0, y0, z0: the distances from the current panel location
  ! coss, sins: the cos/sin of the dihedral angle of the sending panel
  ! cosr, sinr: the cos/sin of the dihedral angle of the receiving panel

  use precision
  implicit none

  complex(kind=dtype), intent(out) :: Kf1, Kf2
  real(kind=dtype), intent(in) :: omega, U, beta, M, x0, y0, z0
  real(kind=dtype), intent(in) :: coss, sins, cosr, sinr
  complex(kind=dtype) :: expk, I = cmplx(0.0, 1.0, kind=dtype)
  real(kind=dtype) :: r1, R, k1, u1, T1, T2

  ! Compute the distances
  r1 = dsqrt(y0**2 + z0**2)
  R = dsqrt(x0**2 + beta**2*(y0**2 + z0**2))

  ! Compute the k1 and u1 coefficients used elsewhere
  k1 = omega*r1/U
  u1 = (M*R - x0)/(r1*beta**2)

  ! T1 = cos(gr - gs)
  T1 = cosr*coss + sinr*sins

  ! T2 = (z0*cos(gr) - y0*sin(gr))*(z0*cos(gs) - y0*sin(gs))
  T2 = (z0*coss - y0*sins)*(z0*cosr - y0*sinr)

  call evalK1K2Coeff(Kf1, Kf2, r1, u1, k1, beta, R, M)

  expk = exp(-I*omega*x0/U)

  Kf1 = expk*Kf1*T1
  Kf2 = expk*Kf2*T2

end subroutine evalKernel

subroutine computeQuadDoubletCoeff(dinf, omega, U, beta, M, xr, xi, xo, &
     cosr, sinr, coss, sins)
  ! Evaluate the doublet kernel function 
  ! 
  ! Input:
  ! omega:      the frequency of oscillation
  ! x0, y0, z0: the distances from the current panel location
  ! coss, sins: the cos/sin of the dihedral angle of the sending panel
  ! cosr, sinr: the cos/sin of the dihedral angle of the receiving panel

  use precision
  implicit none

  ! Input/output arguments
  complex(kind=dtype), intent(out) :: dinf
  real(kind=dtype), intent(in) :: omega, U, beta, M
  real(kind=dtype), intent(in) :: xr(3), xi(3), xo(3)
  real(kind=dtype), intent(in) :: cosr, sinr, coss, sins

  ! Local real values
  real(kind=dtype) :: x0, y0, z0
  real(kind=dtype) :: eta, zeta, r1, e

  ! The kernel functions evaluate at the different points
  complex(kind=dtype) :: Ki1, Ki2, Km1, Km2, Ko1, Ko2
  complex(kind=dtype) :: A1, B1, C1, A2, B2, C2

  ! Set a constant for later useage
  real(kind=dtype) :: zero = 0.0_dtype
  real(kind=dtype) :: one = 1.0_dtype

  ! Compute the kernel function at the inboard point
  x0 = xr(1) - xi(1)
  y0 = xr(2) - xi(2)
  z0 = xr(3) - xi(2)
  call evalKernel(Ki1, Ki2, omega, U, beta, M, &
       x0, y0, z0, cosr, sinr, coss, sins)

  ! Evaluate the kernel function at the outboard point
  x0 = xr(1) - xo(1)
  y0 = xr(2) - xo(2)
  z0 = xr(3) - xo(2)
  call evalKernel(Ko1, Ko2, omega, U, beta, M, &
       x0, y0, z0, cosr, sinr, coss, sins)

  ! Evaluate the kennel function at the mid-point
  x0 = xr(1) - 0.5*(xi(1) + xo(1))
  y0 = xr(2) - 0.5*(xi(2) + xo(2))
  z0 = xr(3) - 0.5*(xi(3) + xo(3))
  call evalKernel(Km1, Km2, omega, U, beta, M, &
       x0, y0, z0, cosr, sinr, coss, sins)

  ! Compute the A, B and C coefficients for the first term
  A1 = (Ki1 - 2*Km1 + Ko1)/(2*e**2)
  B1 = (Ko1 - Ki1)/(2*e)
  C1 = Km1

  ! Compute the A, B and C coefficients for the second term
  A2 = (Ki2 - 2*Km2 + Ko2)/(2*e**2)
  B2 = (Ko2 - Ki2)/(2*e)
  C2 = Km2

  ! Compute horizontal distance in the local reference frame
  eta = (xr(2) - 0.5*(xi(2) + xo(2)))*coss &
       + (xr(3) - 0.5*(xi(3) + xo(3)))*sins

  ! Compute the normal distance in the local reference frame
  zeta = -(xr(2) - 0.5*(xi(2) + xo(2)))*sins &
       + (xr(3) - 0.5*(xi(3) + xo(3)))*coss

  if (zeta == zero) then
     dinf = ((eta**2*A1 + eta*B1 + C1)*(one/(eta - e) - one/(eta + e)) &
          + (0.5*B1 + eta*A1)*2.0*log((eta - e)/(eta + e)) &
          + 2.0*e*A1)
  else
     dinf = (((eta**2 - zeta**2)*A1 + eta*B1 + C1)* &
          atan2(2*e*abs(zeta), (r1**2 - e**2))/abs(zeta) &
          + (0.5*B1 + eta*A1)*log((r1**2 - 2*eta*e + e**2)/(r1**2 + 2*eta*e + e**2)) &
          + 2*e*A1)  
  end if

end subroutine computeQuadDoubletCoeff

subroutine computeInputMeshSegment(n, m, x0, span, dihedral, sweep, cr, tr, Xi, Xo, Xr)
  ! This routine computes parts of the input mesh for a given lifting
  ! segment. The input consists of the root location, span, dihedral,
  ! sweep, root chord and taper ratio. This function can be called
  ! repeatedly to construct a model for a wing. Note that the sweep is
  ! the 1/4-chord sweep and the wing is always constructed such that
  ! it is parallel with the x-axis as required by the DLM theory.
  ! 
  ! Input:
  ! n:        the number of span-wise panels
  ! m:        the number of chord-wise panels
  ! span:     the semi-span of the segment
  ! dihedral: the wing dihedral
  ! sweep:    the wing sweep
  ! cr:       the root chord
  ! tr:       the taper ratio (tip chord = tr*cr)
  !
  ! Output:
  ! Xi:  the inboard sending point (1/4 box chord in board)
  ! Xo:  the outboard sending point (1/4 box chord outboard)
  ! Xr:  the receiving point (3/4 box chord)

  use precision
  implicit none

  integer, intent(in) :: n, m
  integer :: i, j, counter
  real(kind=dtype), intent(in) :: x0(3), span, dihedral, sweep, cr, tr
  real(kind=dtype), intent(inout) :: Xi(3,n*m), Xo(3,n*m), Xr(3,n*m)
  real(kind=dtype) :: yp, c

  counter = 1
  do i = 1, n
     do j = 1, m
        ! Compute the inboard doublet location at the 1/4 chord Note
        ! that yp is the span-wise station and c is the chord position
        ! relative to the 1/4 chord location of this segment. The
        ! tan(sweep) takes care of the 1/4 chord sweep.
        yp = (i-1.0)*span/n
        c = cr*(1.0 - (1.0 - tr)*yp/span)*((j - 0.75)/m - 0.25)
        Xi(1, counter) = x0(1) + yp*tan(sweep) + c
        Xi(2, counter) = x0(2) + yp
        Xi(3, counter) = x0(3) + yp*tan(dihedral)

        ! Compute the outboard doublet location at the 1/4 chord
        yp = i*span/n
        c = cr*(1.0 - (1.0 - tr)*yp/span)*((j - 0.75)/m - 0.25)
        Xo(1, counter) = x0(1) + yp*tan(sweep) + c
        Xo(2, counter) = x0(2) + yp
        Xo(3, counter) = x0(3) + yp*tan(dihedral)

        ! Compute the receiving point at the 3/4 chord
        yp = (i-0.5)*span/n
        c = cr*(1.0 - (1.0 - tr)*yp/span)*((j - 0.25)/m - 0.25)
        Xr(1, counter) = x0(1) + yp*tan(sweep) + c
        Xr(2, counter) = x0(2) + yp
        Xr(3, counter) = x0(3) + yp*tan(dihedral)

        ! Update the counter location
        counter = counter + 1
     end do
  end do

end subroutine computeInputMeshSegment

subroutine computeDoublet(D, omega, U, M, np, Xi, Xo, Xr)
  ! This routine computes the complex influence coefficient
  ! matrix. The input consists of a number of post-processed
  ! connectivity and nodal locations are given and locations,
  ! determine
  ! 
  ! Input:
  ! omega: the frequency of oscillation
  ! U:     the velocity of the free-stream
  ! M:     the free-stream Mach number
  ! Xi:    inboad sending point
  ! Xo:    outboard sending point
  ! Xr:    receiving point
  !
  ! Output:
  ! A:  complex coefficient matrix
  ! np: number of panels

  use precision
  implicit none

  integer, intent(in) :: np
  complex(kind=dtype), intent(inout) :: D(np, np)
  real(kind=dtype), intent(in) :: omega, U, M
  real(kind=dtype), intent(in) :: Xi(3,np), Xo(3,np), Xr(3,np)

  integer :: r, s
  real(kind=dtype) :: beta

  ! Compute the factor beta
  beta = sqrt(1.0 - M**2)

  do s = 1, np
     do r = 1, np
        ! Compute the local panel properties
        D(s, r) = 0.0_dtype
     end do
  end do

end subroutine computeDoublet
     
