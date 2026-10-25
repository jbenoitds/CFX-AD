"""
Figure 1 of the paper: the CO molecule along the bond axis, in the layout of Figure 1 of
Roy, Cuierrier and Ernzerhof 2023 (J. Phys. Chem. A 127, 2026).

  Upper panel : all-electron CCSD density along the bond axis
  Middle panel: |rho_CCSD - rho_model| / rho_CCSD for CFX (lambda = 0.0), PBE and SCAN
  Lower panel : the same for CFXmix (lambda = 0.2)

  * Basis aug-cc-pVTZ; the CCSD reference is all-electron, since the figure is about the
    core region.
  * All densities are evaluated on the same 1-D line, z = linspace(-2, 2, 2001) Bohr, with
    mol.eval_gto("GTOval", coords), not on the DFT quadrature grid, so the ratios are exact.
  * Frame: bond midpoint at the origin, C at z = -1.0658, O at z = +1.0658 Bohr
    (R = 1.128 A, as in bench_density.py).

Three steps; the reference and the CF densities are computed in separate processes:
  python -u fig1_CO.py --ref               # PySCF only: PBE, SCAN, CCSD -> data/fig1_CO_ref_<basis>.npz
  CFLAMBD=0.0 python -u fig1_CO.py --dft   # CF density at one lambda   -> fig1_CO_cf_<basis>_lambd<L>.npz
  CFLAMBD=0.2 python -u fig1_CO.py --dft
  python -u fig1_CO.py --plot              # figure1_CO.png/.pdf + fig1_CO_<basis>.csv
The reference npz is shipped in data/, so --ref is only needed to regenerate it.
"""
import sys, os, time, types
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

BASIS = os.environ.get("CFBASIS", "aug-cc-pVTZ")
LAMBD = float(os.environ.get("CFLAMBD", "0.0"))
_basistag = BASIS.lower().replace("-", "").replace(".", "")
REF_NPZ = f"{ROOT}/data/fig1_CO_ref_{_basistag}.npz"
CF_NPZ  = lambda lam: f"{ROOT}/fig1_CO_cf_{_basistag}_lambd{lam}.npz"

# --- frame: bond midpoint at the origin ------------------------------------
R_ANG  = 1.128                      # CO bond length, as in bench_density.py
ANG2BOHR = 1.8897261254578281
R_BOHR = R_ANG * ANG2BOHR           # 2.13161 Bohr
HALF   = R_BOHR / 2.0               # 1.06580
ATOM   = [["C", (0.0, 0.0, -HALF)], ["O", (0.0, 0.0, +HALF)]]

MAXMEM = float(os.environ.get("CFMAXMEM", "2200"))   # PySCF max_memory (MB)

import numpy as np
ZLINE = np.linspace(-2.0, 2.0, 2001)
COORDS = np.stack([np.zeros_like(ZLINE), np.zeros_like(ZLINE), ZLINE], axis=1)


def build_mol(verbose=3):
    from pyscf import gto
    return gto.M(atom=ATOM, unit="Bohr", basis=BASIS, spin=0, charge=0, verbose=verbose)


def rho_on_line(mol, dm):
    """Total density on the 1-D line grid. Same grid for every density -> exact ratios."""
    dm = np.asarray(dm)
    if dm.ndim == 3:
        dm = dm[0] + dm[1]
    ao = mol.eval_gto("GTOval", COORDS)          # (npts, nao)
    return np.einsum('pi,ij,pj->p', ao, dm, ao)


mode = sys.argv[1] if len(sys.argv) > 1 else "--plot"

# ===================== reference and PBE/SCAN curves (PySCF only) =====================
if mode == "--ref":
    from pyscf import dft, cc
    t0 = time.time()
    mol = build_mol()
    out = {"z": ZLINE}

    for xc in ("pbe", "scan"):
        m = dft.RKS(mol, xc=xc); m.conv_tol = 1e-10; m.max_memory = MAXMEM; m.kernel()
        out[f"rho_{xc}"] = rho_on_line(mol, m.make_rdm1())
        print(f"[fig1] {xc}: E={m.e_tot:.8f} conv={m.converged}", flush=True)

    mf = mol.RHF(); mf.conv_tol = 1e-10; mf.max_memory = MAXMEM; mf.run()
    out["rho_hf"] = rho_on_line(mol, mf.make_rdm1())

    # all-electron CCSD, relaxed density
    mycc = cc.CCSD(mf); mycc.max_memory = MAXMEM; mycc.run(); mycc.solve_lambda()
    out["rho_ccsd"] = rho_on_line(mol, np.asarray(mycc.make_rdm1(ao_repr=True)))
    print(f"[fig1] CCSD(all-electron): E={mycc.e_tot:.8f} conv={mycc.converged}", flush=True)

    np.savez(REF_NPZ, **out)
    print(f"[fig1] refs done {time.time()-t0:.1f}s -> {REF_NPZ}", flush=True)
    sys.exit(0)

# ===================== the CF functional (JAX) =====================
if mode == "--dft":
    from jax import config
    config.update("jax_enable_x64", True)
    import CFXm_jax
    CFXm_jax.LAMBD = LAMBD          # must be set before the first traced call
    from pyscf.dft import RKS
    from get_veff import R_get_veff

    t0 = time.time()
    mol = build_mol()
    m = RKS(mol); m.get_veff = types.MethodType(R_get_veff, m)
    m.max_cycle = 200; m.conv_tol = 1e-9
    m.kernel()
    if not m.converged:
        dm = m.make_rdm1()
        m2 = RKS(mol); m2.get_veff = types.MethodType(R_get_veff, m2)
        m2 = m2.newton(); m2.max_cycle = 100; m2.kernel(dm0=dm); m = m2
    np.savez(CF_NPZ(LAMBD), z=ZLINE, rho_cf=rho_on_line(mol, m.make_rdm1()),
             conv=bool(m.converged), e_tot=m.e_tot)
    print(f"[fig1] CF(LAMBD={LAMBD}): E={m.e_tot:.8f} conv={m.converged} "
          f"{time.time()-t0:.1f}s -> {CF_NPZ(LAMBD)}", flush=True)
    sys.exit(0)

# ===================== assemble the figure =====================
if mode != "--plot":
    sys.exit(__doc__)

ref = np.load(REF_NPZ)
z, rho_ref = ref["z"], ref["rho_ccsd"]
relerr = lambda rho: np.abs(rho_ref - rho) / rho_ref
err = {"PBE": relerr(ref["rho_pbe"]), "SCAN": relerr(ref["rho_scan"])}
for lam, name in ((0.0, "CFX"), (0.2, "CFXmix")):
    cf = np.load(CF_NPZ(lam))
    if not (np.array_equal(cf["z"], z) and bool(cf["conv"])):
        sys.exit(f"{CF_NPZ(lam)}: different line grid or unconverged SCF")
    err[name] = relerr(cf["rho_cf"])

# ---- mean relative error by region (the averages quoted in the text)
CORE, BOND = 0.35, 0.6              # Bohr: within CORE of a nucleus; |z| <= BOND
core_m = (np.abs(z + HALF) < CORE) | (np.abs(z - HALF) < CORE)
bond_m = np.abs(z) < BOND
print(f"  {'model':7s} {'core':>9s} {'bond':>9s} {'whole':>9s} {'max':>9s}")
for k in ("CFX", "CFXmix", "PBE", "SCAN"):
    e = err[k]
    print(f"  {k:7s} {e[core_m].mean():9.5f} {e[bond_m].mean():9.5f} {e.mean():9.5f} {e.max():9.5f}")

# ---- CSV
csv = f"{ROOT}/fig1_CO_{_basistag}.csv"
with open(csv, "w") as f:
    f.write(f"# Figure 1 | CO along the bond axis | basis={BASIS} | ref=CCSD relaxed, all-electron\n")
    f.write(f"# frame: bond midpoint at origin, C at z={-HALF:.5f}, O at z={+HALF:.5f} Bohr "
            f"(R={R_ANG} A = {R_BOHR:.5f} Bohr); rho in e/Bohr^3\n")
    f.write("z_bohr,rho_ccsd,relerr_cfx,relerr_cfxmix,relerr_pbe,relerr_scan\n")
    for i in range(len(z)):
        f.write(f"{z[i]:.6f},{rho_ref[i]:.10e},{err['CFX'][i]:.6e},{err['CFXmix'][i]:.6e},"
                f"{err['PBE'][i]:.6e},{err['SCAN'][i]:.6e}\n")

# ---- figure: single column (3.45 in), 8 pt, 600 dpi PNG + vector PDF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

COL = {"CFX": "#0b0b0b", "CFXmix": "#0b0b0b", "PBE": "#eb6834", "SCAN": "#2a78d6"}
INK, MUTED = "#0b0b0b", "#6b6a66"
BAND_CORE, BAND_BOND = "#ececea", "#e6edf5"
YMAX_ERR = 0.018

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.7,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 3.5, "ytick.major.size": 3.5, "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6, "legend.fontsize": 7,
})


def shade(ax):
    for x0 in (-HALF, HALF):
        ax.axvspan(x0 - CORE, x0 + CORE, color=BAND_CORE, lw=0, zorder=0)
    ax.axvspan(-BOND, BOND, color=BAND_BOND, lw=0, zorder=0)


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(3.45, 6.6), sharex=True,
                                    gridspec_kw={"height_ratios": [1, 1.25, 1.25], "hspace": 0.07})

shade(ax1)
ax1.plot(z, rho_ref, color=INK, lw=1.0, zorder=3)
ax1.set_ylim(0, 340)
ax1.yaxis.set_major_locator(MultipleLocator(100))
ax1.yaxis.set_minor_locator(MultipleLocator(50))
ax1.set_ylabel(r"$\rho_{\rm CCSD}(z)$  (e Bohr$^{-3}$)")
for x0, sym in ((-HALF, "C"), (HALF, "O")):
    ax1.text(x0 + 0.07, rho_ref[np.argmin(np.abs(z - x0))], sym, ha="left", va="center",
             fontsize=8.5, color=INK)
for xc, name in ((-HALF, "core"), (0.0, "bond"), (HALF, "core")):
    ax1.text(xc, 1.015, name, transform=ax1.get_xaxis_transform(),
             ha="center", va="bottom", fontsize=6.5, color=MUTED)

for ax, name, label in ((ax2, "CFX", r"CFX ($\lambda = 0.0$)"),
                        (ax3, "CFXmix", r"CFXmix ($\lambda = 0.2$)")):
    shade(ax)
    ax.plot(z, err[name], color=COL[name], lw=1.35, label=label, zorder=4)
    ax.plot(z, err["PBE"], color=COL["PBE"], lw=0.9, label="PBE", zorder=3)
    ax.plot(z, err["SCAN"], color=COL["SCAN"], lw=0.9, label="SCAN", zorder=3)
    ax.set_ylim(0, YMAX_ERR)
    ax.yaxis.set_major_locator(MultipleLocator(0.005))
    ax.yaxis.set_minor_locator(MultipleLocator(0.001))
    ax.set_ylabel(r"$|\rho_{\rm CCSD}-\rho_{\rm model}|\,/\,\rho_{\rm CCSD}$")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.97, 1.0), handlelength=1.6,
              handletextpad=0.5, labelspacing=0.3, borderaxespad=0.3)
ax3.set_xlabel(r"$z$  (Bohr)")
ax3.set_xlim(-2, 2)
ax3.xaxis.set_major_locator(MultipleLocator(1.0))
ax3.xaxis.set_minor_locator(MultipleLocator(0.25))
fig.align_ylabels((ax1, ax2, ax3))

for ext in ("png", "pdf"):
    fig.savefig(f"{ROOT}/figure1_CO.{ext}", dpi=600, bbox_inches="tight", facecolor="white")
print(f"\n[written] {csv}\n[written] {ROOT}/figure1_CO.png / .pdf", flush=True)
