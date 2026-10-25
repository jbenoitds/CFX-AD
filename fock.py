'''
This script builds the XC Fock matrix from the derivatives of the energy density with respect to
each ingredient, rung by rung (LDA, GGA, meta-GGA, exact exchange), following Lehtola, Blockhuys
and Van Alsenoy. 2020. doi:10.3390/molecules25051218 (their eq 56)
'''

import numpy as np

def calc_Fock(ao_values, weights, v_rho, grad=None, v_G=None, v_GC=None, grad_B=None,
    v_lap=None, v_tau=None, fxx=None, v_xx = None):
    '''
    XC matrix of one spin channel on the AO basis.

    ao_values : AO values and derivatives from numint.eval_ao(deriv=2)
    weights   : grid weights
    v_rho     : dexc/drho
    grad, v_G : this spin's density gradient and dexc/d|grad rho|^2
    grad_B, v_GC : the other spin's gradient and dexc/d(grad rho_a . grad rho_b)
    v_lap, v_tau : dexc/dlap and dexc/dtau
    fxx, v_xx : the exact-exchange intermediate (hfx_density) and dexc/dexx
    '''

    aog, dx_ao, dy_ao, dz_ao, dxx, _, _, dyy, _, dzz = ao_values

    ## LDA
    vw = np.nan_to_num(v_rho * weights)
    vxc = np.einsum('r,ri,rj->ij',vw, aog, aog)

    ## GGA
    if v_G is not None:
       dx,dy,dz = grad
       gao = (np.einsum('r,ri->ri', dx, dx_ao) + np.einsum('r,ri->ri', 
            dy, dy_ao) + np.einsum('r,ri->ri', dz ,dz_ao))
       v_gao = np.einsum('ri,r,r->ri', gao, v_G, weights)
       vxc_GGA = 2 * np.einsum('ri,rj->ij', v_gao, aog)
       vxc_GGA += vxc_GGA.T
       vxc += vxc_GGA

    if v_GC is not None:
       dx,dy,dz = grad_B
       gao = (np.einsum('r,ri->ri', dx, dx_ao) + np.einsum('r,ri->ri', 
        dy, dy_ao) + np.einsum('r,ri->ri', dz ,dz_ao))
       v_gao = np.einsum('ri,r,r->ri', gao, v_GC, weights)
       vxc_GGA =  np.einsum('ri,rj->ij', v_gao, aog)
       vxc_GGA += vxc_GGA.T
       vxc += vxc_GGA

    ## Meta-GGA
    if v_tau is not None:
       fw = 0.5 * v_tau + 2 * v_lap
       fw *= weights
       fw = np.nan_to_num(fw)

       dxao_fw = np.einsum('ri,r->ri', dx_ao, fw)
       vxc_mGGA = np.einsum('ri,rj->ij', dxao_fw, dx_ao) 

       dyao_fw = np.einsum('ri,r->ri', dy_ao, fw)
       vxc_mGGA += np.einsum('ri,rj->ij', dyao_fw, dy_ao)

       dzao_fw = np.einsum('ri,r->ri', dz_ao, fw)
       vxc_mGGA += np.einsum('ri,rj->ij', dzao_fw, dz_ao)

       flwao = np.einsum('r,r,ri->ri', v_lap, weights, aog)
       lapao = dxx + dyy + dzz

       vxc_lap = np.einsum('ri,rj->ij', flwao, lapao)
       vxc_lap += vxc_lap.T

       vxc_mGGA += vxc_lap
       vxc += 0.5 * (vxc_mGGA + vxc_mGGA.T)

    ## Exact Exchange Energy Density
    if v_xx is not None:
       lmfw = weights*v_xx
       vxx = -np.einsum('r, ri,rj -> ij', lmfw, aog, fxx)
       vxx += vxx.transpose()
       vxx = vxx/2.
       vxc += vxx

    return vxc
