'''
This file includes the routines for the B03 non-normalized exchange hole: the shape variable
x = a*b, either from the exact inversion _solve_x_b03 or from the fit B03_fit of Arbuznikov and
Kaupp (doi:10.1016/j.theochem.2005.08.039), and the hole parameters a, b, c from x.

Both take as input y = -3/(2 pi) * Q/rho^3 * eps_x^{exact}, where eps_x^{exact} is the exact
exchange energy density (not per particle). The paper's eq. 10 writes -3/(4 pi) * Q/rho^2 * eps_X;
the form used here is the one consistent with the forward map G below.

The function B03_param then computes the BR parameters a,b,c from x, Q (the second order X hole) and rho.
'''

from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as np
import numpy as onp
from jax import jit, lax

# === B03 Fit Functions Based on Arbuznikov & Kaupp (2006) ===

@jit
def B03_f1(y):
    """
    Region I: y <= -0.640362
    Eq. 25 of the paper
    """
    y1 = -0.640362
    x1 = 1.470896
    B = 0.937781
    A = (2.0 * x1) / np.pi
    return A * np.arctan(B * (y - y1)) + x1

@jit
def B03_f2(y):
    """
    Region II: -0.640362 < y <= 8.0
    Eq. 29 - Padé approximant
    """
    x0 = 2.604471
    y0 = 1.0
    p1, p2, p3, p4 = 0.488513, 0.403272, 0.108382, 0.00688645
    q1, q2, q3, q4 = 1.017201, 0.375899, 0.0513617, 0.00158852
    z = y - y0
    numerator = p1 * z + p2 * z**2 + p3 * z**3 + p4 * z**4
    denominator = 1.0 + q1 * z + q2 * z**2 + q3 * z**3 + q4 * z**4
    return x0 + numerator / denominator

@jit
def B03_f3(y):
    """
    Region III: y > 8.0
    Eq. 30 - Logarithmic + polynomial terms
    """
    a, b, c, d, e = 1.23767, 9.37587, -19.4777, 13.6816, -0.078677
    z = np.log(y)
    term = z ** (1.0 + 1.0 / z)
    correction = a / z + b / z**2 + c / z**3 + d / z**4 + e
    return term + correction

@jit
def B03_fit(y):
    """
    Piecewise analytic inverse fit: x = f^(-1)(y)
    Matches Eq. (14) from Arbuznikov & Kaupp (2006)
    """
    y1 = -0.640362
    y2 = 8.0

    def f1(y): return B03_f1(y)
    def f2(y): return B03_f2(y)
    def f3(y): return B03_f3(y)

    return lax.cond(y <= y1, f1,
                    lambda y: lax.cond(y <= y2, f2, f3, y),
                    y)

### Exact inversion of the B03 non-normalized hole
#
# Eliminating a^2 between the two conditions used in B03_param,
#     energy    : a^2 = 6 rho^2 (e^x - 1 - x/2) / (x W),   W = -3 eps_x/(2 pi)
#     curvature : a^2 = 6 Q x / (rho (x - 2))
# gives the map that B03_fit approximates, in closed form:
#     upsis = W Q / rho^3 = G(x) = (x - 2)(e^x - 1 - x/2) / x^2
# G is strictly increasing on (0, inf), with G(0+) = -inf, G(2) = 0 and G(x) ~ e^x/x at large x,
# so the root is unique. The fit's error in x is up to 2.5e-2, and since Q is linear in (x - 2)
# its small absolute error near y = 0 gives a wrong (even sign-reversed) curvature for |y| small.
# The bisection below resolves x - 2 to ~6e-11.

def _G_b03_np(x):
    """_G_b03 in numpy, evaluated once at import for the two bracket clips."""
    return (x - 2.0)*(onp.exp(x) - 1.0 - 0.5*x)/x**2

_B03_X_LO     = 1e-12        # bracket floor;   G(1e-12) = -1e12
_B03_X_HI     = 60.0         # bracket ceiling; G(60) = 1.84e24
_B03_BISECT_N = 40           # log-space: (log X_HI - log X_LO)/2^40 -> rel. 2.9e-11 in x, uniformly
_B03_Y_LO     = float(_G_b03_np(_B03_X_LO))   # = -1.000e+12
_B03_Y_HI     = float(_G_b03_np(_B03_X_HI))   # =  1.840e+24


@jit
def _G_b03(x):
    """Forward map of the B03 non-normalized hole: upsis = -3/(2 pi) Q eps_x / rho^3.
    Strictly increasing on (0, inf); G(2) = 0 exactly (the Q -> 0 point)."""
    return (x - 2.0) * (np.exp(x) - 1.0 - 0.5*x) / x**2


@jit
def _solve_x_b03(upsi):
    """Exact inverse of _G_b03 (replaces B03_fit).

    Bisection in log x, since G ~ -1/x as x -> 0 and log x gives uniform relative precision.
    Differentiated by the implicit function theorem via lax.custom_root: dx/dupsi = 1/G'(x*).
    Scalar-in (vmapped per grid point), as _solve_x in CFXm_jax.

    The clip only keeps upsi inside the bracket [G(X_LO), G(X_HI)]; it is not a physical limit."""
    upsi_c = np.clip(upsi, _B03_Y_LO, _B03_Y_HI)     # keeps the bracket valid
    f = lambda x: _G_b03(x) - upsi_c
    def solve(f, x0):
        lo0 = onp.log(_B03_X_LO) + 0.0*upsi_c
        hi0 = onp.log(_B03_X_HI) + 0.0*upsi_c
        def body(_, lohi):
            lo, hi = lohi
            mid = 0.5*(lo + hi)
            go_hi = f(np.exp(mid)) < 0.0                 # G(mid) < upsi -> root is above mid
            return (np.where(go_hi, mid, lo), np.where(go_hi, hi, mid))
        lo, hi = lax.fori_loop(0, _B03_BISECT_N, body, (lo0, hi0))
        return np.exp(0.5*(lo + hi))
    tangent_solve = lambda g, y: y / g(1.0)              # scalar IFT: z = y / G'(x*)
    return lax.custom_root(f, 2.0 + 0.0*upsi_c, solve, tangent_solve)


# === Param Function from Inverted BR89 ===

@jit
def B03_param(x, ex, Q, rho):
    """
    Calculate parameters a, b, c from x, ex, Q, and rho.
    a is taken from the energy condition, which is finite for every x > 0 (ex < 0),
    including x = 2 where the curvature form a = 6 Q x / (rho (x-2)) is 0/0.
    """
    def from_ex(x, ex, Q, rho):
        a = (1.0 / x) * (np.exp(x) - 1.0 - x / 2.0) * rho**2 / (-3.0 / (2.0 * np.pi) * ex)
        a *= 6.0
        a = np.sqrt(a)
        b = x / a
        c = rho * np.exp(x)
        return a, b, c

    return from_ex(x, ex, Q, rho)
