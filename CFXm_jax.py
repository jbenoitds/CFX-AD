'''
XC energy density of the CFX correlation-factor functional, and its derivatives, in JAX.

The coefficients A, B and E of the correlation factor are closed-form fits; C and D follow from
the normalization and energy conditions, using exact moments of the Becke-Roussel exchange hole.
Adapted from Přecechtělová, Bahmann, Kaupp and Ernzerhof. 2015. https://doi.org/10.1063/1.4932074

Cycle 2 below refers to the determination of D from epsilon_x^PBE and epsilon_xc^PBE (normalized
BR hole). Cycle 3 refers to the exchange hole built from epsilon_x^exact (B03 non-normalized hole).

The energy density and its 14 derivatives come from a single value_and_grad pass, so the
potential is the exact derivative of the energy that is integrated. Points of negligible density
are screened at the input, and the grid is processed in chunks to bound memory.

Two module constants select the variant. They are read when JAX first traces the functional, so
set them before the first call and use one process per setting:
    LAMBD     = 0.0 : CFX   (0.2 : CFXmix)
    EXACT_B03 = True: exact inversion of the B03 hole (False: the Arbuznikov-Kaupp fit)
'''

from jax import config, jit, vmap, value_and_grad, lax
config.update("jax_enable_x64", True)
from B03_jax import B03_fit, B03_param, _solve_x_b03
import jax.numpy as np
from jax.scipy.special import log_ndtr
import numpy as onp


### Numerical-safety constants

RHO_MIN   = 1e-10   # total density below which the CF contribution is set to zero
RHO_FLOOR = 1e-12   # per-quantity floor used inside denominators to avoid 0/0
CHUNK     = 20000   # number of grid points evaluated per vmap batch

## Placeholder ingredients for points with rho_tot < RHO_MIN: they keep every branch finite, so the
## gradient there is 0 rather than NaN. The output at those points is set to 0 (see get_exc_CF).
_DUM_RHO   =  0.1
_DUM_G     =  0.01
_DUM_LAP   =  0.1
_DUM_TAU   =  0.1
_DUM_EPSX  = -0.05
_DUM_EPSXC = -0.1
_DUM_EXX   = -0.05


### Variant selection

LAMBD     = 0.0     # fraction of the normalized BR hole curvature mixed into Q: 0.0 = CFX, 0.2 = CFXmix
EXACT_B03 = True    # True = exact inversion _solve_x_b03; False = B03_fit (Arbuznikov & Kaupp 2006)


### Fit for the coefficient A

@jit
def calc_A(rs,zeta):
        """
        Calculate A which is used to reproduce the on-top value of rho_xc of LSD
        Input:
            rs:float
                Wigner-Seitz radius
            zeta:
                spin polarisation
        Return:
            A
        """
        mu=0.193
        nu=0.525
       #Calculate A (eq.38)
        return ((1.-zeta**2.)*0.5*(1.0+mu*rs)/(1.0+nu*rs+mu*nu*rs**2.)-1.0)*(-2./(1.+zeta**2.))


### Fit for the coefficient B

@jit
def calc_B(rs,zeta):
        """
        Calculate B which is used to reproduce the cusp value of rho_xc of LSD
        Input:
            rs:float
                Wigner-Seitz radius
            zeta:
                spin polarisation
        Return:
            B
        """
        a=0.193
        b=0.525
        u=0.33825
        v=0.89679
        t=0.10134
        kappa=(4.0/(3.0*np.pi))*(9.0*np.pi/4.0)**(1.0/3.0)
        H = (1.0+u*rs)/(2.0+v*rs+t*rs**2.)
        return (-4.0/(3.0*np.pi*kappa))*rs*H*((1.0-zeta**2.)/(zeta**2.+1.0))*((1.0+a*rs)/(1.0+b*rs+a*b*rs**2.))


### Fit for the coefficient E

@jit
def calc_E_fit(rs,zeta):
        """
        Calculate E, the Gaussian damping of the correlation factor, from a fit to the
        condition that the CF applied to the LSD exchange hole reproduces epsilon_xc^LSD

        Input:
            rs: Wigner-Seitz radius
            zeta: spin polarization
        Return:
            E
        """
        a0,b0,c0,d0,e0 = 0.773708,2.717775 , 0.210202, 21.708929, 9.661078
        a1,b1,c1,d1,e1 = 0.541112, 1.926632, 0.236849, 13.024536, 8.921728

        alpha=  a0+(a1-a0)*zeta**2
        beta=   b0+(b1-b0)*zeta**2
        gamma=  c0+(c1-c0)*zeta**2
        delta = d0+(d1-d0)*zeta**2
        epsilon=e0+(e1-e0)*zeta**2

        x=1./rs
        E = (alpha+beta*x+gamma*x**2)/(1.+delta*x+epsilon*x**2)
        return E


### Exact moments of the Becke-Roussel hole for the C/D coefficients (Cycles 2 and 3)
#
# The moments  m_n = int_0^inf y^n brhole(y) e^{-E y^2} dy,  n=1..6, of the
# spherically-averaged Becke-Roussel exchange hole
#   brhole(y) = -c/(2 a^2 b y) [ (a|b-y|+1) e^{-a|b-y|} - (a(b+y)+1) e^{-a(b+y)} ]
# are evaluated in closed form.
#
# Splitting the integral at the kink y=b:
#   m_n = -c/(2 a^2 b) (P_n - Q_n),  P_n,Q_n built from
#   A_k^s(L,U) = int_L^U y^k e^{s a y - E y^2} dy,  s = +/-1,  for k=0..6.
# Each A_k is seeded at k=0 by an erfc term (completing the square) and raised in k
# by the integration-by-parts recurrence
#   A_k = (1/2E)[ s a A_{k-1} + (k-1) A_{k-2} - (bdry_U - bdry_L) ].
#
# Numerical stability:
#  * The prefactor e^{a^2/4E} overflows for small E, so it is never formed alone: it is
#    folded with the outer e^{+/-ab} factor, which turns erfc into erfcx (bounded for
#    nonnegative arguments) and every boundary term into e^{-E b^2} or e^{-ab} (both <= 1).
#  * erfcx is built from log_ndtr, which is differentiable and does not overflow:
#        erfcx(x) = exp(x^2 + ln2 + log_ndtr(-x*sqrt2))
#  * The finite-interval seed (group P1, s=+1 on [0,b]) is branched on mu_+=a/2E vs b so
#    every erfcx argument stays >= 0. Each branch is fed a safe argument ("double where")
#    so the inactive branch cannot produce a NaN in the gradient. The two branch
#    formulas agree to C1 at mu_+ = b.
#  * For large E all moments go to 0 smoothly.

_SQRT2 = np.sqrt(2.0)
_LOG2  = np.log(2.0)

@jit
def _erfcx(x):
    """exp(x^2) erfc(x) for x >= 0, via log_ndtr (overflow-free, differentiable)."""
    return np.exp(x * x + _LOG2 + log_ndtr(-x * _SQRT2))

@jit
def single_moments_JX_E(params):
    """Exact m_1..m_6 (closed form; see the block comment above) for scalar (E,a,b,c),
    mapped per grid point by the surrounding vmap."""
    E, a, b, c = params
    sqrtE = np.sqrt(E)
    inv2E = 1.0 / (2.0 * E)
    mu_p  = a * inv2E                     # mu_+ = a/(2E)   (mu_- = -mu_+)
    hs    = 0.5 * np.sqrt(np.pi / E)      # (1/2) sqrt(pi/E)
    ab    = a * b
    e_mab  = np.exp(-ab)                  # e^{-ab}    (<= 1)
    e_mEb2 = np.exp(-E * b * b)           # e^{-E b^2} (<= 1)

    # --- k=0 seeds of the scaled A-integrals (all e^{a^2/4E} folded away) ---
    # Group Q : e^{-ab} int_0^inf y^k e^{-a y - E y^2} dy       (s=-1)
    AQ0  = e_mab * hs * _erfcx(sqrtE * mu_p)
    # Group P2: e^{+ab} int_b^inf y^k e^{-a y - E y^2} dy       (s=-1)
    AP20 = e_mEb2 * hs * _erfcx(sqrtE * (b + mu_p))
    # Group P1: e^{-ab} int_0^b  y^k e^{+a y - E y^2} dy        (s=+1, finite -> branch)
    d      = b - mu_p                                   # upper shifted limit; sign -> branch
    argA   = sqrtE * np.where(d >= 0.0,  d, 0.0)        # used when mu_p <= b
    argB   = sqrtE * np.where(d <  0.0, -d, 0.0)        # used when mu_p >  b
    arg_mu = sqrtE * mu_p                                # >= 0 always
    expo_A = np.minimum(a * a * (0.25 / E) - ab, 0.0)   # <= 0 in the active (mu_p<=b) region
    termA  = 2.0 * np.exp(expo_A) - e_mEb2 * _erfcx(argA) - e_mab * _erfcx(arg_mu)
    termB  = e_mEb2 * _erfcx(argB) - e_mab * _erfcx(arg_mu)
    AP10   = hs * np.where(mu_p <= b, termA, termB)

    # --- raise k=0..6 by the recurrence. The upward recurrence loses about log10(kappa^k)
    #     digits, kappa = a^2/(2E), so for kappa >= SWITCH_KAPPA the downward (Olver)
    #     recurrence is used instead, for which this integral is the stable solution.
    #     The clamps only keep the inactive branch finite; at active points they do nothing.
    SWITCH_KAPPA = 25.0
    KMAX_DN      = 30                                    # Olver downward start index
    def _upward(A0, s, aa, betaf):
        A = [A0, inv2E * (s * aa * A0 - betaf(1))]
        for k in range(2, 7):
            A.append(inv2E * (s * aa * A[k-1] + (k-1) * A[k-2] - betaf(k)))
        return A
    def _downward(A0, s, aa, EE, betaf):                 # minimal solution via Olver
        # lax.scan rather than a Python loop keeps the differentiated graph small
        two = 2.0 * EE
        ks  = np.arange(KMAX_DN, 1, -1)                  # k = KMAX_DN, ..., 2
        init = (0.0 * a, 1.0 + 0.0 * a, 0.0 * a, 0.0 * a)  # (h_k,h_{k-1},p_k,p_{k-1}) at k=KMAX
        def step(carry, k):
            hk, hk1, pk, pk1 = carry
            km1 = k - 1.0
            h_new = (two * hk - s * aa * hk1) / km1              # h[k-2]
            p_new = (two * pk - s * aa * pk1 + betaf(k)) / km1   # p[k-2]
            return (hk1, h_new, pk1, p_new), (h_new, p_new)
        _, (hh, pp) = lax.scan(step, init, ks)          # outputs ordered k-2 = KMAX-2..0
        h6 = hh[-7:][::-1]; p6 = pp[-7:][::-1]           # -> index 0..6
        Cc = (A0 - p6[0]) / h6[0]                        # fix free constant from exact A0
        return [p6[i] + Cc * h6[i] for i in range(7)]
    kappa  = a * a * inv2E
    use_dn = kappa >= SWITCH_KAPPA
    a_up   = np.minimum(a, np.sqrt(4.0 * E * SWITCH_KAPPA))  # inactive-up: kappa_eff<=2*SWITCH
    E_dn   = np.minimum(E, 30.0)                         # inactive-dn: bounded h growth
    a_dn   = np.minimum(a, 1.0e4)
    def blend(A0, s, betaf):
        up = _upward(A0, s, a_up, betaf)
        dn = _downward(A0, s, a_dn, E_dn, betaf)
        return [np.where(use_dn, dn[k], up[k]) for k in range(7)]
    bd_b  = lambda k: b ** (k - 1) * e_mEb2              # scaled boundary at y=b
    bd_0  = lambda k: np.where(k == 1, e_mab, 0.0 * a)   # y=0 boundary (only k=1)
    betaQ  = lambda k: -bd_0(k)                          # Q : U=inf, L=0
    betaP1 = lambda k: bd_b(k) - bd_0(k)                 # P1: U=b,   L=0
    betaP2 = lambda k: -bd_b(k)                          # P2: U=inf, L=b
    AQ  = blend(AQ0,  -1.0, betaQ)                       # (0, inf)
    AP1 = blend(AP10,  1.0, betaP1)                      # (0, b)
    AP2 = blend(AP20, -1.0, betaP2)                      # (b, inf)

    pref = -c / (2.0 * a * a * b)
    m = []
    for n in range(1, 7):
        Qn = (ab + 1.0) * AQ[n-1] + a * AQ[n]
        Pn = ((ab + 1.0) * AP1[n-1] - a * AP1[n]
              + (1.0 - ab) * AP2[n-1] + a * AP2[n])
        m.append(pref * (Pn - Qn))
    return m[0], m[1], m[2], m[3], m[4], m[5]


### Exact inversion of the normalized Becke-Roussel hole
#
# The reduced exchange energy per particle of the normalized BR hole is a function of the
# shape variable x = a*b alone (Proynov 2008 Eq. 9; Precechtelova 2015 Eq. 28):
#     F(x) = (3 pi)^{2/3}/2 * e^{x/3}/x * (1 - e^{-x} - x/2 e^{-x})
# F is increasing on (0, inf), with F(0+) = (3 pi)^{2/3}/4 = 1.11546 (the alpha-limit, Eq. 29).
# x is found from a target eps_up by bisection, and dx/deps = 1/F'(x*) comes from the implicit
# function theorem through lax.custom_root, not from differentiating the iterations.
#
# Below the alpha-limit the paper pins alpha = 0.96 and attenuates the hole by
# f_X = 1/(1+delta y^4) (Eqs. 29-30). The attenuation leaves the curvature unchanged, and in this
# code the sub-limit region is reached only on the exact-exchange curvature path, so the
# prescription reduces to flooring the solve at x_min = ln(3 pi/8) + 3 ln 0.96 = 0.041435.

_C_F      = float((3.0*onp.pi)**(2.0/3.0) / 2.0)                 # F(x) prefactor
_C_RC     = float(4.0 / (3.0**(5.0/3.0) * onp.pi**(2.0/3.0)))    # rc(x) prefactor
_X_MIN    = float(onp.log(3.0*onp.pi/8.0) + 3.0*onp.log(0.96))   # alpha=0.96 pin -> x_min=0.041435
_F_XMIN   = float(_C_F*(onp.exp(_X_MIN/3.0)/_X_MIN)              # F(x_min) = 1.13066 (solve floor)
                  *(1.0 - onp.exp(-_X_MIN) - 0.5*_X_MIN*onp.exp(-_X_MIN)))
_X_HI     = 60.0        # bisection upper bracket; F(60) ~ 1.8e7, covers any realistic eps_up
_BISECT_N = 40         # bisection steps -> x* to ~ (X_HI-X_MIN)/2^40 ~ 5e-11
_EPS_TAIL_CAP = 15.0   # cap on eps_up for the curvature path only: rc(x) ~ e^{2x/3} would
                       # otherwise overflow in the density tail. Acts only where the density
                       # is vanishingly small.

@jit
def _Feps(x):
    """Normalized-BR reduced exchange energy per particle vs x=ab (Proynov 2008 Eq. 9, reduced).
    Monotone increasing on (0,inf); F(0+) = (3 pi)^{2/3}/4 (the alpha-limit)."""
    return _C_F * (np.exp(x/3.0)/x) * (1.0 - np.exp(-x) - 0.5*x*np.exp(-x))

@jit
def _rc_of_x(x):
    """Reduced curvature of the normalized BR hole vs x=ab:
    rc = 4/(3^{5/3} pi^{2/3} y(x)),  y(x)=x e^{-2x/3}/(x-2)  (Proynov Eq. 7), rewritten to
    avoid the x=2 pole.  rc is proportional to the hole curvature Q and is smooth through x=2."""
    return _C_RC * (x - 2.0) * np.exp(2.0*x/3.0) / x

@jit
def _solve_x(eps_up):
    """Solve F(x) = eps_up for x on [x_min, inf), pinning x = x_min (alpha=0.96) when
    eps_up < F(x_min).  Differentiable via the implicit function theorem (lax.custom_root):
    dx/d(eps_up) = 1/F'(x*).  Scalar-in (vmapped per grid point)."""
    eps_c = np.maximum(eps_up, _F_XMIN)                 # alpha=0.96 floor (Precechtelova Eqs. 29-30)
    f = lambda x: _Feps(x) - eps_c
    def solve(f, x0):
        lo0 = _X_MIN + 0.0*eps_c; hi0 = _X_HI + 0.0*eps_c
        def body(_, lohi):
            lo, hi = lohi
            mid = 0.5*(lo + hi)
            go_hi = f(mid) < 0.0                        # F(mid) < eps -> root is above mid
            return (np.where(go_hi, mid, lo), np.where(go_hi, hi, mid))
        lo, hi = lax.fori_loop(0, _BISECT_N, body, (lo0, hi0))
        return 0.5*(lo + hi)
    tangent_solve = lambda g, y: y / g(1.0)             # scalar IFT: z = y / F'(x*)
    return lax.custom_root(f, 0.5*(_X_MIN+_X_HI) + 0.0*eps_c, solve, tangent_solve)

@jit
def calc_rc_exact(rho_up, eps_x_up):
    """Reduced curvature of the normalized BR hole that reproduces the exchange energy per
    particle eps_x_up: solve F(x)=eps_up, then rc = rc(x). eps_up is capped at _EPS_TAIL_CAP."""
    kf_up = (3.*np.pi**2.*rho_up)**(1./3.)
    rho_safe = np.where(rho_up > RHO_FLOOR, rho_up, RHO_FLOOR)
    eps_up = -eps_x_up*kf_up**2./(rho_safe*2.*np.pi)
    eps_up = np.minimum(eps_up, _EPS_TAIL_CAP)          # tail cap (see _EPS_TAIL_CAP)
    return _rc_of_x(_solve_x(eps_up))


### Parameters a, b and c of the B03 Non-Normalized Exchange hole (Cycle 3)
#
# get_B03_params inverts upsis = G(x) = (x-2)(e^x - 1 - x/2)/x^2 for the shape x = a*b (see
# B03_jax.py). With EXACT_B03 the inversion is a bisection with the same implicit-function
# derivative as _solve_x; otherwise it uses the three-region fit of Arbuznikov & Kaupp 2006.

@jit
def get_B03_params(rhos,Qs,exxs):
         """a, b, c of the B03 hole from the spin density, the curvature Q and the exact-exchange
         energy density exxs (an energy density, not per particle)."""
         upsis = -3./(2.*np.pi) *Qs/rhos**3 * exxs
         xs = _solve_x_b03(upsis) if EXACT_B03 else B03_fit(upsis)   # resolved at trace time
         a,b,c = B03_param(xs,exxs,Qs,rhos)
         return a,b,c


### Parameters a, b and c of the BR Normalized Exchange hole (Cycle 2)

@jit
def get_BRN_params(rhos,epsxu_PBE):
         ## Solve F(x)=eps_up for x=ab (Proynov Eq. 9 / Precechtelova Eq. 28), then a,b,c
         ## follow from the on-top and normalization conditions.
         kf = (3.*np.pi**2.*rhos)**(1./3.)
         rho_safe = np.where(rhos > RHO_FLOOR, rhos, RHO_FLOOR)
         eps_up = -epsxu_PBE*kf**2./(rho_safe*2.*np.pi)
         xs = _solve_x(eps_up)
         b = xs**3*np.exp(-xs)/(8.*np.pi*rhos)
         b = b**(1./3.)
         a = xs/b
         c = a**3/(8.*np.pi)
         return a,b,c


### Determination of the GGA Moments (Cycle 2)

@jit
def get_moments_PBE(rho, zeta, epsx_PBE, E ):
         ot = 0.5*rho*(1.+zeta**2)
         a,b,c = get_BRN_params(ot,epsx_PBE)
         kf = (3.*np.pi**2*rho)**(1./3.)
         a = a/kf
         b = b*kf
         c = c/rho
         params = [E,a,b,c]
         moments = single_moments_JX_E(params)
         return moments


### Computation of the D Coefficient (Cycle 2)

@jit
def get_D_param(A,B,E,rho_up,rho_down,epsx_PBE_up, epsx_PBE_down, epsxc_PBE):

    ## Spin Polarization and Total Density
    zeta = (rho_up-rho_down)/(rho_up+rho_down)
    rho = rho_up+rho_down

    ## Moments for the Up and Down Spin and Spin-Scaling
    m_up = np.asarray(get_moments_PBE(rho_up,zeta*0.+1., epsx_PBE_up,E/((1.+zeta)*0.5)**(2./3.)))
    afu = [(0.5*(1.+zeta))**(4./3.), (0.5*(1.+zeta)), (0.5*(1.+zeta))**(2./3.), (0.5*(1.+zeta))**(1./3.), 0.*zeta+1., (0.5*(1.+zeta))**(-1./3.)]
    afu = np.asarray(afu)
    m_up = m_up*afu

    m_down = np.asarray(get_moments_PBE(rho_down,zeta*0.+1., epsx_PBE_down,E/((1.-zeta)*0.5)**(2./3.)))
    afd = [(0.5*(1.-zeta))**(4./3.), (0.5*(1.-zeta)), (0.5*(1.-zeta))**(2./3.), (0.5*(1.-zeta))**(1./3.), 0.*zeta+1., (0.5*(1.-zeta))**(-1./3.)]
    afd = np.asarray(afd)
    m_down = m_down*afd

    ## Sum of Spin Contributions
    m = m_up + m_down
    m1,m2,m3,m4,m5,m6 = m

    ## Spin-Scaled Target PBE Energy
    epsxc_PBE = (3.*np.pi**2.*rho)**(2./3.)*epsxc_PBE/(2.*np.pi*rho)

    ## Computation of the D Coefficient
    D = (A*m1*m4 - A*m2*m3 + B*m2*m4 - B*m3**2. - epsxc_PBE*m4 - 3.*np.pi*m3/4.)/(m3*m6 - m4*m5)

    return D


### Main Work to obtain the Exchange Energy via Previous Functions

@jit
def get_exc_CF(rho_up, GA, lap_up, tau_up, rho_down, GB, lap_down, tau_down, GC, epsx_PBE_up,epsx_PBE_down, epsxc_PBE, exxa, exxb):
        """
        To calculate the XC energy density per grid point for the CF functional.

        Returns an energy density (per grid point); the caller integrates exc*weights.
        The value is set to exactly 0 wherever the total density is below RHO_MIN. Because
        this same masked function is differentiated by value_and_grad, the potential is
        likewise 0 there and is exactly consistent with the energy elsewhere.
        """

        ## Input Screening
        ## np.where differentiates both branches, so an inf in the discarded branch would still
        ## give NaN in the gradient (0*inf). Below RHO_MIN every ingredient is therefore replaced
        ## by a placeholder value; above it the ingredients pass through unchanged.
        safe = (rho_up + rho_down) > RHO_MIN

        rho_up        = np.where(safe, rho_up,        _DUM_RHO)
        rho_down      = np.where(safe, rho_down,      _DUM_RHO)
        GA            = np.where(safe, GA,            _DUM_G)
        GB            = np.where(safe, GB,            _DUM_G)
        GC            = np.where(safe, GC,            _DUM_G)
        lap_up        = np.where(safe, lap_up,        _DUM_LAP)
        lap_down      = np.where(safe, lap_down,      _DUM_LAP)
        tau_up        = np.where(safe, tau_up,        _DUM_TAU)
        tau_down      = np.where(safe, tau_down,      _DUM_TAU)
        epsx_PBE_up   = np.where(safe, epsx_PBE_up,   _DUM_EPSX)
        epsx_PBE_down = np.where(safe, epsx_PBE_down, _DUM_EPSX)
        epsxc_PBE     = np.where(safe, epsxc_PBE,     _DUM_EPSXC)
        exxa          = np.where(safe, exxa,          _DUM_EXX)
        exxb          = np.where(safe, exxb,          _DUM_EXX)

        ## Per-channel floors: one spin channel can be vanishingly small even when the total
        ## density is above RHO_MIN.
        rho_up   = np.where(rho_up   >  RHO_FLOOR, rho_up,   RHO_FLOOR)
        rho_down = np.where(rho_down >  RHO_FLOOR, rho_down, RHO_FLOOR)
        exxa     = np.where(exxa     < -RHO_FLOOR, exxa,    -RHO_FLOOR)
        exxb     = np.where(exxb     < -RHO_FLOOR, exxb,    -RHO_FLOOR)

        ## Spin Quantities
        rho = rho_up + rho_down
        zeta = (rho_up-rho_down)/(rho)
        rs = (3./(4.*np.pi*rho))**(1./3.)
        kf = (3.*np.pi**2*rho)**(1./3.)

        kfu = (3.*np.pi**2*rho_up)**(1./3.)
        kfd = (3.*np.pi**2*rho_down)**(1./3.)

        ## Normalization of Energy Densities
        epsx_PBE_up = epsx_PBE_up/rho_up
        epsx_PBE_down = epsx_PBE_down/rho_down
        epsxc_PBE = epsxc_PBE/rho

        ## Retrieve A and B from Density
        A = calc_A(rs,zeta)
        B = calc_B(rs,zeta)

        ## Construct Q quantities for Both Spins (Non-Normalized: B03)
        D_up = tau_up*2.-(1./4.)*GA/rho_up
        D_down = tau_down*2-(1./4.)*GB/rho_down
        Q_up = 1./6.*(lap_up-2.*D_up)
        Q_down = 1./6.*(lap_down-2.*D_down)

        ## Mixing (CFXmix): a fraction LAMBD of the normalized BR hole curvature. At LAMBD = 0.0
        ## (CFX) this branch is skipped at trace time.
        if LAMBD != 0.0:
            QN_up = 0.5*rho_up*kfu**2*calc_rc_exact(rho_up, exxa/rho_up)
            QN_down = 0.5*rho_down*kfd**2*calc_rc_exact(rho_down, exxb/rho_down)
            Q_up = (1. - LAMBD) * Q_up + LAMBD * QN_up
            Q_down = (1. - LAMBD) * Q_down + LAMBD * QN_down

        ## Retrieve E and Solve for D Using PBE XC Data
        E = calc_E_fit(rs,zeta)
        D = get_D_param(A,B,E,rho_up,rho_down,epsx_PBE_up, epsx_PBE_down, epsxc_PBE)

        ## Get the B03 Hole Parameters for Both Spins
        au,bu,cu = get_B03_params(rho_up,Q_up,exxa)
        ad,bd,cd = get_B03_params(rho_down,Q_down,exxb)
        au, bu, cu = au/kfu,bu*kfu, cu/rho_up
        ad, bd, cd = ad/kfd,bd*kfd, cd/rho_down

        ## Compute Exchange Hole Moments
        params_m_up = E/((1.+zeta)*0.5)**(2./3.),au,bu,cu
        m_up = np.asarray(single_moments_JX_E(params_m_up))
        params_m_down = E/((1.-zeta)*0.5)**(2./3.),ad,bd,cd
        m_down = np.asarray(single_moments_JX_E(params_m_down))

        ## Spin-Scaling
        afu = [(0.5*(1.+zeta))**(4./3.), (0.5*(1.+zeta)), (0.5*(1.+zeta))**(2./3.), (0.5*(1.+zeta))**(1./3.), 0.*zeta+1., (0.5*(1.+zeta))**(-1./3.)]
        afu = np.asarray(afu)
        m_up = m_up*afu

        afd = [(0.5*(1.-zeta))**(4./3.), (0.5*(1.-zeta)), (0.5*(1.-zeta))**(2./3.), (0.5*(1.-zeta))**(1./3.), 0.*zeta+1., (0.5*(1.-zeta))**(-1./3.)]
        afd = np.asarray(afd)
        m_down = m_down*afd

        ## Combine Moments
        m = m_up+ m_down
        m1,m2,m3,m4,m5,m6 = m

        ## Solve for C
        C = -(3.*np.pi/4.+m2*A+m3*B+m6*D)/m4
        eps_xc_calc = 2.*np.pi*rho**2/kf**2*(A*m1+B*m2+C*m3+D*m5)

        ## Screen: exactly zero (energy and, by differentiation, potential) in negligible density
        return np.where(safe, eps_xc_calc, 0.0)


### Single-point value + gradient (one reverse-mode pass gives the energy and all 14 derivatives)

_ARGNUMS = tuple(range(14))
_val_and_grad_point = value_and_grad(get_exc_CF, argnums=_ARGNUMS)
_val_and_grad_batch = jit(vmap(_val_and_grad_point))


def U_energy_and_grad(*args):
    """
    Evaluate the XC energy density and its gradient w.r.t. all 14 ingredients over the whole
    grid, in a single value_and_grad pass, processed in chunks to bound peak memory.

    Returns:
        exc      : ndarray (N,)         -- XC energy density per grid point
        grad_exc : tuple of 14 ndarrays -- d(exc)/d(ingredient), same order as get_exc_CF args
    """
    args = [onp.asarray(a) for a in args]
    n = args[0].shape[0]

    e_chunks = []
    g_chunks = None
    for start in range(0, n, CHUNK):
        sl = slice(start, min(start + CHUNK, n))
        chunk = [a[sl] for a in args]
        val, grd = _val_and_grad_batch(*chunk)
        e_chunks.append(onp.asarray(val))
        if g_chunks is None:
            g_chunks = [[onp.asarray(gi)] for gi in grd]
        else:
            for k, gi in enumerate(grd):
                g_chunks[k].append(onp.asarray(gi))

    exc = onp.concatenate(e_chunks)
    grad_exc = tuple(onp.concatenate(gk) for gk in g_chunks)
    return exc, grad_exc
