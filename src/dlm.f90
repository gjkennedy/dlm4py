! dlm.f90: A simple python code for oscillating aerodynamic analysis
!
! Copyright (c) 2015 Graeme Kennedy. All rights reserved. 
!
! The following code is a basic Doublet Lattice method implementation
! that is designed to be run from python. The computationally
! expensive computations are performed at the fortran level for
! efficiency. This code is intended to be used for flutter
! computations but could also be used for general subsonic oscilatory
! flows.

module precision
  ! A module to handle setting the data type for the real/complex analysis
  integer, parameter :: dtype = selected_real_kind(12)
end module precision

module constants
  ! A module that defines constants for later useage
  use precision
  real(kind=dtype), parameter :: PI = 3.1415926535897931_dtype
end module constants

subroutine approxKernelIntegrals(I0, J0, u1, k1)
  ! Compute the approximate values of the integrals I0 and J0. These
  ! integrals are required for the computation of the kernel function
  ! at points along the bound vortex line. The integrals are
  ! approximated based on the following expression from Desmarais:
  !
  ! 1 - u/sqrt(1 + u^2) \approx \sum_{n} a_{n} exp(-p_{n}*u)
  !
  ! Where p_{n} = b*2**n. The output I0 and J0 are defined as follows:
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
  real(kind=dtype) :: pn, a(12)
  complex(kind=dtype) :: expn, invn, kval

  ! Set the parameter b - the expontential in the kernel integral
  real(kind=dtype), parameter :: b = 0.009054814793_dtype

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

  ! Evaluate the integral for I0 and J0
  I0 = cmplx(0.0, 0.0, kind=dtype)
  J0 = cmplx(0.0, 0.0, kind=dtype)

  ! Evaluate the integral for positive values of u1
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
  ! Compute the value of the K1 and K2 functions given the values of
  ! the local panel variables. This code calls the function
  ! approxKernelIntegrals to obtain the values of I0 and J0 which are
  ! used in the evaluation of the kernel coefficients.
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
  real(kind=dtype) :: invsqrt, invR, u1pos
  complex(kind=dtype) :: expk, I0, J0, I1, I2 
  complex(kind=dtype) :: I10, I20, I11, I21

  ! Constant definitions
  complex(kind=dtype), parameter :: I = cmplx(0.0, 1.0, kind=dtype)
  real(kind=dtype), parameter :: zero = 0.0_dtype
  real(kind=dtype), parameter :: one = 1.0_dtype
  real(kind=dtype), parameter :: two = 2.0_dtype

  invR = one/R
  invsqrt = one/sqrt(one + u1**2)

  if (u1 < zero) then
     ! Use separate logic when the argument u1 is negative. This is
     ! required since the approximate integrals for I0 and J0 are not
     ! defined for negative values of u1. 
     call approxKernelIntegrals(I0, J0, zero, k1)

     ! Evaluate I1
     I10 = one - I*k1*I0
     
     ! Evaluate 3*I2
     I20 = two - I*k1*I0 + k1**2*J0

     ! Evaluate the approximate integrals I0 and J0
     u1pos = -u1
     call approxKernelIntegrals(I0, J0, u1pos, k1)

     ! Compute the temporary variable values that will be used below
     expk = exp(-I*k1*u1pos)

     ! Evaluate I1
     I11 = (one - u1pos*invsqrt)*expk - I*k1*I0
     
     ! Evaluate 3*I2
     I21 = ((two + I*k1*u1pos)*(one - u1pos*invsqrt) &
          - u1pos*invsqrt**3)*expk - I*k1*I0 + k1**2*J0

     I1 = cmplx(2.0*real(I10) - real(I11), aimag(I11), kind=dtype)
     I2 = cmplx(2.0*real(I20) - real(I21), aimag(I21), kind=dtype)

     ! Recompute expk
     expk = exp(-I*k1*u1)
  else 
     ! Compute the temporary variable values that will be used below
     expk = exp(-I*k1*u1)

     ! Evaluate the approximate integrals I0 and J0
     call approxKernelIntegrals(I0, J0, u1, k1)

     ! Evaluate I1
     I1 = (one - u1*invsqrt)*expk - I*k1*I0
     
     ! Evaluate 3*I2
     I2 = ((two + I*k1*u1)*(one - u1*invsqrt) &
          - u1*invsqrt**3)*expk - I*k1*I0 + k1**2*J0
  end if

  ! Compute the first component of the kernel function
  Kf1 = I1 + M*r1*invR*invsqrt*expk
  
  ! Compute the second component of the kernel function
  Kf2 = -I2 - I*k1*invsqrt*expk*(M*r1*invR)**2 &
       - (M*r1*invR)*((one + u1**2)*(beta*r1*invR)**2 + &
       two + M*r1*u1*invR)*expk*invsqrt**3

end subroutine evalK1K2Coeff

subroutine evalKernelNumerator(Kf1, Kf2, omega, U, beta, M, &
     x0, y0, z0, cosr, sinr, coss, sins)
  ! Evaluate the two components of the kernel function which are
  ! required for the evaluation of the influence coeffficients. These
  ! coefficients are the difference between the oscillator and
  ! zero-frequency components of the influence coefficients.
  ! 
  ! Input:
  ! omega:      the frequency of oscillation
  ! U:          the free-stream velocity
  ! beta:       sqrt(1 - M**2)
  ! M :         the free-stream Mach number
  ! x0, y0, z0: the distances from the current panel location
  ! coss, sins: the cos/sin of the dihedral angle of the sending panel
  ! cosr, sinr: the cos/sin of the dihedral angle of the receiving panel
  !
  ! Output
  ! Kf1:        the influence

  use precision
  implicit none

  complex(kind=dtype), intent(out) :: Kf1, Kf2
  real(kind=dtype), intent(in) :: omega, U, beta, M, x0, y0, z0
  real(kind=dtype), intent(in) :: coss, sins, cosr, sinr
  
  ! Local temporary variables
  complex(kind=dtype) :: expk
  real(kind=dtype) :: r1, R, k1, u1, T1, T2, Kf10, Kf20
  complex(kind=dtype) :: Kf11, Kf21

  ! Constants used in this function
  real(kind=dtype), parameter :: zero = 0.0_dtype
  complex(kind=dtype), parameter :: I = cmplx(0.0, 1.0, kind=dtype)

  ! Conmpute the distances
  r1 = sqrt(y0**2 + z0**2)
  R = sqrt(x0**2 + beta**2*(y0**2 + z0**2))

  ! Compute the k1 and u1 coefficients used elsewhere
  k1 = omega*r1/U
  if (r1 == zero) then
     if (M*R > x0) then
        u1 = 1D20
     else if (M*R < x0) then
        u1 = -1D20
     else
        u1 = 0.0
     endif
  else
     u1 = (M*R - x0)/(r1*beta**2)
  end if

  ! T1 = cos(gr - gs)
  T1 = cosr*coss + sinr*sins

  ! T2 = (z0*cos(gr) - y0*sin(gr))*(z0*cos(gs) - y0*sin(gs))
  T2 = (z0*coss - y0*sins)*(z0*cosr - y0*sinr)

  call evalK1K2Coeff(Kf11, Kf21, r1, u1, k1, beta, R, M)

  ! Compute the zero-frequency contributions from the coefficients
  Kf10 = 1.0 + x0/R
  Kf20 = -2.0 - (x0/R)*(2.0 + (beta*r1/R)**2)
  
  ! Complete the value values of the kernel function
  expk = exp(-I*omega*x0/U)
  ! Kf1 = expk*(Kf11 - Kf10)*T1
  ! Kf2 = expk*(Kf21 - Kf20)*T2

  Kf1 = expk*Kf11*T1
  Kf2 = expk*Kf21*T2

end subroutine evalKernelNumerator

subroutine computeQuadDoubletCoeff(dinf, omega, U, beta, M, &
     dxav, xr, xi, xo, e, cosr, sinr, coss, sins)
  ! Evaluate the influence coefficient between a sending panel and a
  ! recieving point using a quadratic approximation across a panel.

  use precision
  use constants
  implicit none

  ! Input/output arguments
  complex(kind=dtype), intent(out) :: dinf
  real(kind=dtype), intent(in) :: omega, U, beta, M
  real(kind=dtype), intent(in) :: dxav, xr(3), xi(3), xo(3)
  real(kind=dtype), intent(in) :: e, cosr, sinr, coss, sins

  ! Local real values
  real(kind=dtype) :: x0, y0, z0
  real(kind=dtype) :: eta, zeta, F

  ! The influence coefficients for the different terms
  complex(kind=dtype) :: dinf0, dinf1, dinf2

  ! The kernel functions evaluate at the different points
  complex(kind=dtype) :: Ki1, Ki2, Km1, Km2, Ko1, Ko2
  complex(kind=dtype) :: A1, B1, C1, A2, B2, C2

  ! The coefficients for the horseshoe vortex computation
  real(kind=dtype) :: vy, vz, a(3), b(3), anrm, bnrm, ainv, binv
  real(kind=dtype) :: fact

  ! Set a constant for later useage
  real(kind=dtype), parameter :: zero = 0.0_dtype
  real(kind=dtype), parameter :: half = 0.5_dtype
  real(kind=dtype), parameter :: one = 1.0_dtype

  fact = dxav/(8.0*PI)

  dinf1 = 0.0
  dinf2 = 0.0

  if (omega > 0.0) then
     ! Compute the kernel function at the inboard point
     x0 = xr(1) - xi(1)
     y0 = xr(2) - xi(2)
     z0 = xr(3) - xi(3)
     call evalKernelNumerator(Ki1, Ki2, omega, U, beta, M, &
          x0, y0, z0, cosr, sinr, coss, sins)
     
     ! Evaluate the kernel function at the outboard point
     x0 = xr(1) - xo(1)
     y0 = xr(2) - xo(2)
     z0 = xr(3) - xo(3)
     call evalKernelNumerator(Ko1, Ko2, omega, U, beta, M, &
          x0, y0, z0, cosr, sinr, coss, sins)
     
     ! Evaluate the kennel function at the mid-point
     x0 = xr(1) - half*(xi(1) + xo(1))
     y0 = xr(2) - half*(xi(2) + xo(2))
     z0 = xr(3) - half*(xi(3) + xo(3))
     call evalKernelNumerator(Km1, Km2, omega, U, beta, M, &
          x0, y0, z0, cosr, sinr, coss, sins)
     
     ! Compute the A, B and C coefficients for the first term
     A1 = (Ki1 - 2*Km1 + Ko1)/(2*e**2)
     B1 = (Ko1 - Ki1)/(2*e)
     C1 = Km1

     ! Compute the A, B and C coefficients for the second term
     A2 = (Ki2 - 2*Km2 + Ko2)/(2*e**2)
     B2 = (Ko2 - Ki2)/(2*e)
     C2 = Km2

     ! Print out the coefficients
     ! write(*,*) ' '
     ! write(*,*) 'x0 = ', x0, y0, z0
     ! write(*,*) 'A1 = ', A1
     ! write(*,*) 'B1 = ', B1
     ! write(*,*) 'C1 = ', C1
     
     ! Compute horizontal and vertical distances from the origin in
     ! the local ref. frame
     eta = y0*coss + z0*sins
     zeta = -y0*sins + z0*coss
     
     ! First compute the F-integral
     if (zeta == zero) then
        F = 2*e/(eta**2 - e**2)
     else 
        F = atan2(2*e*abs(zeta), eta*2 + zeta**2 - e**2)/abs(zeta)
     end if

     ! Compute the contribution from the integral of 
     ! (A1*y**2 + B1*y + C1)/((eta - y)**2 + zeta**2)
     dinf1 = (((eta**2 - zeta**2)*A1 + eta*B1 + C1)*F &
          + (0.5*B1 + eta*A1)*log(((eta - e)**2 + zeta**2)/((eta + e)**2 + zeta**2)) &
          + 2.0*e*A1)

     if (zeta == zero) then
        dinf2 = zero
     else
        ! Compute the contribution from the integral of 
        ! (A2*y**2 + B2*y + C2)/((eta - y)**2 + zeta**2)**2
        dinf2 = 0.5*(((eta**2 + zeta**2)*A2 + eta*B2 + C2)*F &
             + (((eta**2 + zeta**2)*eta + (eta**2 - zeta**2)*e)*A2 &
             + (eta**2 + zeta**2 + eta*e)*B2 + (eta + e)*C2)/((eta + e)**2 + zeta**2) &
             - (((eta**2 + zeta**2)*eta - (eta**2 - zeta**2)*e)*A2 &
             + (eta**2 + zeta**2 - eta*e)*B2 + (eta - e)*C2)/((eta - e)**2 + zeta**2))/zeta**2
     end if
  end if

  ! Compute the term dinf0 from a horseshoe vortex method. First add
  ! the contribution from the inboard and outboard vorticies
  a(1) = (xr(1) - xi(1))/beta
  a(2) = (xr(2) - xi(2))
  a(3) = (xr(3) - xi(3))
  anrm = sqrt(a(1)**2 + a(2)**2 + a(3)**2)
  ainv = one/(anrm*(anrm - a(1)))

  b(1) = (xr(1) - xo(1))/beta
  b(2) = (xr(2) - xo(2))
  b(3) = (xr(3) - xo(3))
  bnrm = sqrt(b(1)**2 + b(2)**2 + b(3)**2)
  binv = one/(bnrm*(bnrm - b(1)))

  vy =  a(3)*ainv - b(3)*binv
  vz = -a(2)*ainv + b(2)*binv
  
  ! Now, add the contribution from the bound vortex
  ainv = one/(anrm*bnrm*(anrm*bnrm + a(1)*b(1) + a(2)*b(2) + a(3)*b(3)))
  vy = vy + (a(3)*b(1) - a(1)*b(3))*(anrm + bnrm)*ainv
  vz = vz + (a(1)*b(2) - a(2)*b(1))*(anrm + bnrm)*ainv

  ! Compute the steady normalwash
  dinf0 = sinr*vy - cosr*vz

  dinf0 = 0.0

  ! Add up all the contributions to the doublet
  dinf = fact*(dinf0 + dinf1 + dinf2)

end subroutine computeQuadDoubletCoeff

subroutine computeInputMeshSegment(n, m, x0, span, dihedral, sweep, cr, tr, &
     Xi, Xo, Xr, dXav)
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
  ! Xi:   the inboard sending point (1/4 box chord in board)
  ! Xo:   the outboard sending point (1/4 box chord outboard)
  ! Xr:   the receiving point (3/4 box chord)
  ! dXav: the average panel length in the x-direction

  use precision
  implicit none

  integer, intent(in) :: n, m
  integer :: i, j, counter
  real(kind=dtype), intent(in) :: x0(3), span, dihedral, sweep, cr, tr
  real(kind=dtype), intent(inout) :: Xi(3,n*m), Xo(3,n*m), Xr(3,n*m), dXav(n*m)
  real(kind=dtype) :: yi, yo, yr, c

  counter = 1
  do i = 1, n
     do j = 1, m
        ! Compute the inboard doublet location at the 1/4 chord Note
        ! that yp is the span-wise station and c is the chord position
        ! relative to the 1/4 chord location of this segment. The
        ! tan(sweep) takes care of the 1/4 chord sweep.
        yi = (i-1.0)*span/n
        c = cr*(1.0 - (1.0 - tr)*yi/span)*((j - 0.75_dtype)/m - 0.25_dtype)
        Xi(1, counter) = x0(1) + yi*tan(sweep) + c
        Xi(2, counter) = x0(2) + yi
        Xi(3, counter) = x0(3) + yi*tan(dihedral)

        ! Compute the outboard doublet location at the 1/4 chord
        yo = i*span/n
        c = cr*(1.0 - (1.0 - tr)*yo/span)*((j - 0.75_dtype)/m - 0.25_dtype)
        Xo(1, counter) = x0(1) + yo*tan(sweep) + c
        Xo(2, counter) = x0(2) + yo
        Xo(3, counter) = x0(3) + yo*tan(dihedral)

        ! Compute the receiving point at the 3/4 chord
        yr = (i-0.5)*span/n
        c = cr*(1.0 - (1.0 - tr)*yr/span)*((j - 0.25_dtype)/m - 0.25_dtype)
        Xr(1, counter) = x0(1) + yr*tan(sweep) + c
        Xr(2, counter) = x0(2) + yr
        Xr(3, counter) = x0(3) + yr*tan(dihedral)

        ! Compute the average chord length of this panel
        dXav(counter) = 0.5*cr*((1.0 - (1.0 - tr)*yo/span) &
             + (1.0 - (1.0 - tr)*yi/span))/m

        ! Update the counter location
        counter = counter + 1
     end do
  end do

end subroutine computeInputMeshSegment

subroutine computeInfluenceMatrix(D, omega, U, M, np, &
     Xi, Xo, Xr, dXav, symmetric)
  ! This routine computes the complex influence coefficient
  ! matrix. The input consists of a number of post-processed
  ! connectivity and nodal locations are given and locations,
  ! determine
  ! 
  ! Input:
  ! omega: the frequency of oscillation
  ! U:     the velocity of the free-stream
  ! M:     the free-stream Mach number
  ! np:    number of panels
  ! Xi:    inboad sending point
  ! Xo:    outboard sending point
  ! Xr:    receiving point
  ! dXav:  average length in the x-direction of the panel
  !
  ! Output:
  ! D:  complex coefficient matrix

  use precision
  implicit none

  ! Input/output types
  integer, intent(in) :: np, symmetric
  complex(kind=dtype), intent(inout) :: D(np, np)
  real(kind=dtype), intent(in) :: omega, U, M
  real(kind=dtype), intent(in) :: Xi(3,np), Xo(3,np), Xr(3,np), dXav(np)

  ! Temporary data used internally
  integer :: r, s
  real(kind=dtype) :: beta, xrsymm(3), sinsymm
  real(kind=dtype) :: lr, pe(np), pcos(np), psin(np)
  complex(kind=dtype) :: dtmp

  ! Compute the compressibility factor
  beta = sqrt(1.0 - M**2)

  ! Pre-processing step: Compute the sin/cos and length of all the
  ! panels in the model
  do r = 1, np
     ! Compute 1/2 the bound vortex length
     pe(r) = 0.5*sqrt((Xo(2,r) - Xi(2,r))**2 + (Xo(3,r) - Xi(3,r))**2)

     ! Compute the sin and cos of the dihedral
     pcos(r) = 0.5*(Xo(2,r) - Xi(2,r))/pe(r)
     psin(r) = 0.5*(Xo(3,r) - Xi(3,r))/pe(r)
  end do

  if (symmetric == 0) then
     do s = 1, np
        do r = 1, np
           ! Compute the panel influence coefficient
           call computeQuadDoubletCoeff(D(r, s), omega, U, beta, M, &
                dXav(s), Xr(:, r), Xi(:, s), Xo(:, s), pe(s), &
                pcos(r), psin(r), pcos(s), psin(s))
        end do
     end do
  else
     do s = 1, np
        do r = 1, np
           ! Compute the panel influence coefficient
           call computeQuadDoubletCoeff(D(r, s), omega, U, beta, M, &
                dXav(s), Xr(:, r), Xi(:, s), Xo(:, s), pe(s), &
                pcos(r), psin(r), pcos(s), psin(s))

           ! Compute the influence from the same panel, but the
           ! reflected point
           xrsymm(1) =  Xr(1, r)
           xrsymm(2) = -Xr(2, r)
           xrsymm(3) =  Xr(3, r)
           sinsymm = -psin(r)

           call computeQuadDoubletCoeff(dtmp, omega, U, beta, M, &
                dXav(s), xrsymm, Xi(:, s), Xo(:, s), pe(s), &
                pcos(r), sinsymm, pcos(s), psin(s))
           D(r, s) = D(r, s) + dtmp
        end do
     end do
  end if
end subroutine computeInfluenceMatrix

subroutine computePeriodicBC(rhs, aoa, omega, U, np, Xi, Xo)
  
  use precision
  implicit none

  ! Input/output types
  integer, intent(in) :: np
  complex(kind=dtype), intent(inout) :: rhs(np)
  real(kind=dtype), intent(in) :: aoa, omega, U
  real(kind=dtype), intent(in) :: Xi(3,np), Xo(3,np)

  integer :: i
  real(kind=dtype) :: lr, cosr

  ! Set the normal wash in each panel
  do i = 1, np
     ! Compute
     lr = sqrt((Xo(2,i) - Xi(2,i))**2 + (Xo(3,i) - Xi(3,i))**2)
     cosr = (Xo(2,i) - Xi(2,i))/lr

     rhs(i) = sin(aoa)*cosr*exp(cmplx(0.0_dtype, -omega, kind=dtype))
  end do

end subroutine computePeriodicBC
