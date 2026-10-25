'''
This script creates the grids, evaluates the ingredients (rho, gradient, Laplacian, tau, the PBE
energy densities and the exact-exchange energy density), then feeds them to the CFXm_jax script
and the fock script to get Exc and Vxc.

It is separated in the Unrestricted (U_get_veff) and Restricted (R_get_veff) parts. Either one
replaces the get_veff method of a PySCF UKS/RKS object, e.g. mf.get_veff = MethodType(U_get_veff, mf).
'''

from pyscf import lib, dft
from pyscf.dft import numint
from CFXm_jax import U_energy_and_grad
from fock import calc_Fock
import numpy as np
import numpy
import hfx_density as hfx


### U_get_veff Creates the Grids and Import Ingredients (LDA, GGA, mGGA, and Exact) for Unrestricted

def U_get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    '''Coulomb + CF XC potential for UKS (same signature as pyscf.dft.uks.get_veff).

    Returns (Vxc_a + J, Vxc_b + J) as a lib.tag_array carrying ecoul, exc and vj.
    dm_last and vhf_last are accepted for compatibility and not used.
    '''
    if mol is None: mol = ks.mol
    if dm is None: dm = ks.make_rdm1()
    if not isinstance(dm, numpy.ndarray):
        dm = numpy.asarray(dm)
    if dm.ndim == 2:
        dm = numpy.asarray((dm*.5,dm*.5))

    ## Creation of the Grid
    ks.initialize_grids(mol, dm)
    ground_state = (dm.ndim == 3 and dm.shape[0] == 2)
    coords, weights = ks.grids.coords, ks.grids.weights


    ## AO Values (up to second derivatives, for the Laplacian)
    ao_values = numint.eval_ao(mol, coords, deriv=2)

    ## mGGA: Ingredients
    rho_up, dx_up, dy_up, dz_up, lap_up, tau_up = numint.eval_rho(mol, ao_values, dm[0], xctype='Meta-GGA', with_lapl=True)
    rho_down, dx_down, dy_down, dz_down, lap_down, tau_down = numint.eval_rho(mol, ao_values, dm[1], xctype='Meta-GGA', with_lapl=True)

    GA = dx_up**2 + dy_up**2 + dz_up**2
    GB = dx_down**2 + dy_down**2 + dz_down**2
    GC = dx_up * dx_down + dy_up * dy_down + dz_up * dz_down

    zeros = numpy.zeros_like(rho_up)

    ## GGA: Ingredients
    gga_up = [rho_up,dx_up,dy_up,dz_up]
    gga_down = [rho_down,dx_down,dy_down,dz_down]
    eps_x_PBE_up,vx_PBE_up = dft.libxc.eval_xc("PBE,", [gga_up,[zeros,zeros,zeros,zeros]],spin=1)[:2]
    eps_x_PBE_down,vx_PBE_down = dft.libxc.eval_xc("PBE,", [[zeros,zeros,zeros,zeros],gga_down],spin=1)[:2]
    eps_xc_PBE,vxc_PBE = dft.libxc.eval_xc("PBE,PBE",[gga_up,gga_down],spin=1)[:2]

    eps_x_PBE_up *= rho_up
    eps_x_PBE_down *= rho_down
    eps_xc_PBE *= (rho_up + rho_down)

    ## Exact: Ingredients
    hfx_result = hfx.get_hf_density(mol, dm, deriv=1, coords=coords, weights=weights )
    exxa = hfx_result.exx[:][0]
    exxb = hfx_result.exx[:][1]
    fxxa = hfx_result.fxx[0]
    fxxb = hfx_result.fxx[1]
   

    ## Calculation of XC Energy Density per Grid Point and Gradients (CFXm_jax Script)
    ## One value_and_grad pass, so the potential is the derivative of the integrated energy.
    exc, grad_exc = U_energy_and_grad(rho_up, GA, lap_up, tau_up, rho_down, GB, lap_down,
        tau_down, GC, eps_x_PBE_up, eps_x_PBE_down, eps_xc_PBE, exxa, exxb)

    (v_rho_u, v_G_u, v_lap_u, v_tau_u, v_rho_d, v_G_d, v_lap_d,
        v_tau_d, v_G_c, vaxu, vaxd, vaxc, vxxa, vxxb) = grad_exc


    ## Chain Rule Propagation of Derivation
    v_rho_u += vx_PBE_up[0][:,0] * vaxu + vxc_PBE[0][:,0] * vaxc
    v_rho_d += vx_PBE_down[0][:,1] * vaxd + vxc_PBE[0][:,1] * vaxc
    v_G_u += vx_PBE_up[1][:,0] * vaxu + vxc_PBE[1][:,0] * vaxc
    v_G_d += vx_PBE_down[1][:,2] * vaxd + vxc_PBE[1][:,2] * vaxc
    v_G_c += vxc_PBE[1][:,1] * vaxc

    
    ## Calculation of Final XC Potential Matrices (fock Script)
    vxc_up = calc_Fock(ao_values,weights, v_rho_u, [dx_up,dy_up,dz_up], v_G_u, v_GC=v_G_c, 
        grad_B=[dx_down,dy_down,dz_down], v_lap = v_lap_u, v_tau = v_tau_u, fxx = fxxa, v_xx = vxxa)
    
    vxc_down = calc_Fock(ao_values,weights, v_rho_d, [dx_down,dy_down,dz_down], v_G_d, v_GC=v_G_c, 
        grad_B=[dx_up,dy_up,dz_up], v_lap = v_lap_d, v_tau = v_tau_d, fxx=fxxb, v_xx = vxxb)


    ## Coulomb Matrix
    vj = ks.get_j(mol, dm[0]+dm[1], hermi)


    ## Total Effective Potential (Vxc + J)
    vxc = vxc_up +vj , vxc_down+vj


    ## Total XC Energy
    exc = np.sum(exc*weights)


    ## Compute the Coulomb Energy
    if ground_state:
        ecoul = numpy.einsum('ij,ji', dm[0]+dm[1], vj).real * .5
    else:
        ecoul = None


    ## Create the Vxc Array (including Vxc, Ecoul, Exc and Vj)
    vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=None)
    
    return vxc


### R_get_veff Creates the Grids and Import Ingredients (LDA, GGA, mGGA, and Exact) for Restricted

def R_get_veff(ks, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
    '''Coulomb + CF XC potential for RKS (same signature as pyscf.dft.rks.get_veff).

    The closed-shell density is split into two equal spin densities and passed to the same
    spin-resolved functional. ks.xc is not used; ks.grids is initialized here.

    Args:
        ks : an instance of :class:`RKS`
        dm : ndarray, the density matrix
        dm_last, vhf_last : accepted for compatibility and not used
        hermi : int, passed to ks.get_j

    Returns:
        Veff = J + Vxc as a lib.tag_array carrying ecoul, exc and vj.
    '''

    if mol is None: mol = ks.mol
    if dm is None: dm = ks.make_rdm1()
    
    ## Creation of the Grid
    ks.initialize_grids(mol, dm)
    coords, weights = ks.grids.coords, ks.grids.weights


    ## AO Values (up to second derivatives, for the Laplacian)
    ao_values = numint.eval_ao(mol, coords, deriv=2)

    ## mGGA: Ingredients
    rho, dx, dy, dz, lap, tau = numint.eval_rho(mol, ao_values, dm, xctype='Meta-GGA', with_lapl=True)

    G = dx**2 + dy**2 + dz**2

    zeros = numpy.zeros_like(rho)

    ## GGA: Ingredients
    gga_up = [rho/2,dx/2,dy/2,dz/2]
    gga_down = [rho/2,dx/2,dy/2,dz/2]
    eps_x_PBE_up,vx_PBE_up = dft.libxc.eval_xc("PBE,", [gga_up,[zeros,zeros,zeros,zeros]],spin=1)[:2]
    eps_x_PBE_down,vx_PBE_down = dft.libxc.eval_xc("PBE,", [[zeros,zeros,zeros,zeros],gga_down],spin=1)[:2]
    eps_xc_PBE,vxc_PBE = dft.libxc.eval_xc("PBE,PBE",[gga_up,gga_down],spin=1)[:2]

    eps_x_PBE_up *= rho/2
    eps_x_PBE_down *= rho/2
    eps_x_PBE = 2 * eps_x_PBE_up
    eps_xc_PBE *= (rho)

    ## Exact: Ingredients
    hfx_result = hfx.get_hf_density(mol, dm, deriv=1, coords=coords, weights=weights )
    exxa = hfx_result.exx[:][0]
    exxb = hfx_result.exx[:][1]
    fxxa = hfx_result.fxx[0]
    fxxb = hfx_result.fxx[1]

    rho_up = rho_down = rho/2.
    lap_up = lap_down = lap/2.
    tau_up = tau_down = tau/2.
    GA = GB = GC = G/4.


    ## Calculation of XC Energy Density per Grid Point and Gradients (CFXm_jax Script)
    exc, grad_exc = U_energy_and_grad(rho_up, GA, lap_up, tau_up, rho_down, GB, lap_down, tau_down, GC, eps_x_PBE_up, eps_x_PBE_down, eps_xc_PBE, exxa, exxb)

    v_rho_u, v_G_u, v_lap_u, v_tau_u, v_rho_d, v_G_d, v_lap_d, v_tau_d, v_G_c, vaxu, vaxd, vaxc, vxxa, vxxb = grad_exc


    ## Chain Rule Propagation of Derivation
    v_rho_u += vx_PBE_up[0][:,0] * vaxu + vxc_PBE[0][:,0] * vaxc
    v_rho_d += vx_PBE_down[0][:,1] * vaxd + vxc_PBE[0][:,1] * vaxc
    v_G_u += vx_PBE_up[1][:,0] * vaxu + vxc_PBE[1][:,0] * vaxc
    v_G_d += vx_PBE_down[1][:,2] * vaxd + vxc_PBE[1][:,2] * vaxc
    v_G_c += vxc_PBE[1][:,1] * vaxc

    v_rho = np.nan_to_num(v_rho_u +v_rho_d)/2.
    v_G = (v_G_u+v_G_d + v_G_c)/4.
    v_lap = (v_lap_u +v_lap_d)/2.
    v_tau = np.nan_to_num(v_tau_u +v_tau_d)/2.


    ## Calculation of Final XC Potential Matrices (fock Script)
    vxc = calc_Fock(ao_values,weights, v_rho, grad = [dx,dy,dz], v_G = v_G, 
        v_lap = v_lap, v_tau = v_tau, fxx= (fxxa+fxxb)/2., v_xx = (vxxa+vxxb)/2.)


    ## Coulomb Matrix
    vj = ks.get_j(mol, dm, hermi)


    ## Total Effective Potential (Vxc + J)
    vxc += vj


    ## Total XC Energy
    exc = np.sum(exc*weights)
    ground_state = (isinstance(dm, numpy.ndarray) and dm.ndim == 2)


    ## Compute the Coulomb Energy
    if ground_state:
        ecoul = numpy.einsum('ij,ji', dm, vj).real * .5
    else:
        ecoul = None


    ## Create the Vxc Array (including Vxc, Ecoul, Exc and Vj)
    vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=None)

    return vxc