"""
Density accuracy on the 28 diatomics of Roy, Cuierrier and Ernzerhof 2023 (J. Phys. Chem. A
127, 2026), Table 3, following their protocol:
  - rho_DFT : self-consistent density of the CF functional (get_veff hook);
  - rho_CCSD: CCSD relaxed density (solve_lambda + make_rdm1, ao_repr);
  - metric (their Eq. 15):  error_i = (1/Ne_i) * INT |rho_DFT - rho_CCSD| d3r,
    on the DFT integration grid; the set MAE = mean_i error_i.

Config via environment:
  CFBASIS  basis set          (default aug-cc-pVTZ)
  CFLAMBD  functional variant (0.0 = CFX (default), 0.2 = CFXmix)
  CFFROZEN CCSD core          ("core" frozen-core (default) | "0" all-electron)
Output: density_results_<basistag>_lambd<L>.csv  (one row per molecule, resumable)

Two stages per molecule, in separate processes, so that JAX and the CCSD never hold memory
at the same time: the DFT stage writes its density matrix to an npz and exits, and the CCSD
stage never imports JAX.

Usage (run_density.sh runs these two per molecule, then --mae once):
  python -u bench_density.py --dft <MOL>   # stage 1: CF SCF density -> stage_dft_<tag>_<MOL>.npz
  python -u bench_density.py --cc  <MOL>   # stage 2: CCSD + Eq. 15  -> append CSV row
  python -u bench_density.py --mae          # set MAE over the CSV
"""
import sys, os, time, types, re
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
BASIS  = os.environ.get("CFBASIS", "aug-cc-pVTZ")
FROZEN = os.environ.get("CFFROZEN", "core")           # "0"=all-electron CCSD, "core"=frozen-core
LAMBD  = float(os.environ.get("CFLAMBD", "0.0"))      # 0.0=CFX, 0.2=CFXmix
_variant = "CFX" if LAMBD == 0.0 else ("CFXmix" if LAMBD == 0.2 else f"lambda={LAMBD}")
_tag   = BASIS.lower().replace("-", "").replace(".", "") + f"_lambd{LAMBD}"
OUT    = f"{ROOT}/density_results_{_tag}.csv"

# 28-molecule diatomic set (Roy 2023): name -> (spin=2S, bond length / Angstrom)
DIATOMICS = {
    'LiH':(0,1.595), 'BH':(0,1.232), 'CN':(1,1.172), 'OH':(1,0.969), 'LiO':(1,1.695),
    'CO':(0,1.128),  'NO':(1,1.151), 'HF':(0,0.917), 'LiF':(0,1.564), 'BF':(0,1.263),
    'CF':(1,1.272),  'FO':(1,1.354), 'NaF':(0,1.926),'AlF':(0,1.654), 'SiO':(0,1.509),
    'PN':(0,1.491),  'PO':(1,1.476), 'HS':(1,1.341), 'SiS':(0,1.929), 'CS':(0,1.535),
    'NS':(1,1.494),  'SO':(2,1.481), 'SF':(1,1.601), 'HCl':(0,1.275), 'NaCl':(0,2.361),
    'LiCl':(0,2.021),'ClO':(1,1.570), 'ClF':(0,1.628),   # ClF: the 28th, from the SI of Roy 2023
}
ORDER = list(DIATOMICS.keys())


def _stage_path(MOL):
    return f"{ROOT}/stage_dft_{_tag}_{MOL}.npz"


def _ensure_header():
    if not os.path.exists(OUT):
        with open(OUT, "a") as f:
            f.write(f"# DENSITIES | basis={BASIS} | LAMBD={LAMBD} ({_variant}) | "
                    f"ref=CCSD relaxed (solve_lambda, frozen={FROZEN}) | metric=Eq.15 (1/Ne)*INT|drho| on DFT grid\n")
            f.write("molecule,spin,nelec,density_error,dft_converged,cc_converged,secs\n")


def _done_set():
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for ln in f:
                if not ln.strip() or ln.startswith(("#", "molecule", "MAE")):
                    continue
                done.add(ln.split(",")[0])
    return done


def compute_mae():
    import numpy as np
    errs = []
    with open(OUT) as f:
        for ln in f:
            if not ln.strip() or ln.startswith(("#", "molecule", "MAE")):
                continue
            p = ln.strip().split(",")
            if p[3] not in ("nan", "") and p[4] == "True" and p[5] == "True":
                errs.append(float(p[3]))
    val = float(np.mean(errs)) if errs else float("nan")
    with open(OUT, "a") as f:
        f.write(f"MAE_over_{len(errs)}_converged,,,{val:.12f},,,\n")
    print(f"[MAE] over {len(errs)} fully-converged molecules = {val:.6f}", flush=True)


def build_mol(MOL):
    from pyscf import gto
    spin, R = DIATOMICS[MOL]
    positions = [[0.0, 0.0, 0.0], [0.0, 0.0, R]]
    atom_list = []
    for el, count in re.findall(r'([A-Z][a-z]*)(\d*)', MOL):
        atom_list.extend([el] * (int(count) if count else 1))
    molecule = [[a, tuple(positions[i])] for i, a in enumerate(atom_list)]
    return gto.M(atom=molecule, spin=spin, basis=BASIS, verbose=3), spin


# ----------------------------- arg parsing -----------------------------
mode = sys.argv[1]
if mode == "--mae":
    compute_mae(); sys.exit(0)
if mode not in ("--dft", "--cc"):
    sys.exit("usage: bench_density.py [--dft|--cc] <MOL> | --mae   (two-stage: run --dft then --cc)")
MOL = sys.argv[2]
if MOL not in DIATOMICS:
    sys.exit(f"unknown molecule {MOL!r}; choose from {ORDER}")

import numpy as np

# ============================ STAGE 1: DFT (JAX) ============================
if mode == "--dft":
    _ensure_header()
    if MOL in _done_set():
        print(f"[{MOL}] already in CSV; skip DFT stage.", flush=True); sys.exit(0)
    from jax import config
    config.update("jax_enable_x64", True)
    import CFXm_jax
    CFXm_jax.LAMBD = LAMBD          # must be set before the first traced call
    from pyscf.dft import UKS, RKS
    from get_veff import U_get_veff, R_get_veff

    t0 = time.time()
    mol, spin = build_mol(MOL)

    def make_dft():
        if spin > 0:
            m = UKS(mol); m.get_veff = types.MethodType(U_get_veff, m)
        else:
            m = RKS(mol); m.get_veff = types.MethodType(R_get_veff, m)
        m.max_cycle = 200; m.conv_tol = 1e-8
        return m

    def converge(mf):
        mf.level_shift = 0.0; mf.damp = 0.0; mf.kernel()
        if mf.converged:
            return mf, "clean"
        dm = mf.make_rdm1(); mf.level_shift = 0.2; mf.damp = 0.4; mf.kernel(dm)
        dm = mf.make_rdm1(); mf.level_shift = 0.0; mf.damp = 0.0; mf.kernel(dm)
        if mf.converged:
            return mf, "stab+polish"
        dm = mf.make_rdm1(); m2 = make_dft().newton(); m2.max_cycle = 100
        m2.kernel(dm0=dm)
        return m2, "newton"

    dft_conv = False; how = "-"; dm = np.zeros((1,))
    try:
        mf, how = converge(make_dft())
        dft_conv = bool(mf.converged)
        dm = np.asarray(mf.make_rdm1())
    except Exception as e:
        print(f"[{MOL}] DFT EXCEPTION: {type(e).__name__}: {e}", flush=True)
    np.savez(_stage_path(MOL), dm=dm, conv=dft_conv, spin=spin)
    print(f"[{MOL}] DFT done: conv={dft_conv} how={how} {time.time()-t0:.1f}s", flush=True)
    sys.exit(0)

# ========================= STAGE 2: CCSD (PySCF, no JAX) =========================
if mode == "--cc":
    _ensure_header()
    if MOL in _done_set():
        print(f"[{MOL}] already in CSV; skip.", flush=True); sys.exit(0)
    from pyscf import cc, dft
    from pyscf.dft import numint

    t0 = time.time()
    mol, spin = build_mol(MOL)

    # load the DFT density from stage 1
    sp = _stage_path(MOL)
    dft_conv = False; dm_dft = None
    if os.path.exists(sp):
        z = np.load(sp, allow_pickle=True)
        dm_dft = z["dm"]; dft_conv = bool(z["conv"])

    cc_conv = False; error = float("nan")
    try:
        if dm_dft is None or dm_dft.ndim == 1:
            raise RuntimeError("no valid DFT density from stage 1")
        # CCSD relaxed-density reference (max_memory in MB; lower it on a small machine)
        mf_hf = mol.UHF() if spin > 0 else mol.RHF()
        mf_hf.conv_tol = 1e-9; mf_hf.max_memory = 2600; mf_hf.run()
        ncore = 0
        if FROZEN == "core":
            from pyscf.data.elements import chemcore
            ncore = int(chemcore(mol))
        mycc = cc.CCSD(mf_hf, frozen=ncore) if ncore else cc.CCSD(mf_hf)
        mycc.max_memory = 2400; mycc.run()
        mycc.solve_lambda()
        cc_conv = bool(getattr(mycc, "converged", True))
        dm_ccsd = np.asarray(mycc.make_rdm1(ao_repr=True))

        # Eq. 15 on the same default DFT grid, rebuilt here without JAX
        g = dft.UKS(mol) if spin > 0 else dft.RKS(mol)
        g.grids.build()
        coords, weights = g.grids.coords, g.grids.weights
        ao = numint.eval_ao(mol, coords)

        def trho(dm):
            if dm.ndim == 3:
                return numint.eval_rho(mol, ao, dm[0]) + numint.eval_rho(mol, ao, dm[1])
            return numint.eval_rho(mol, ao, dm)

        error = float(np.sum(np.abs(trho(dm_ccsd) - trho(dm_dft)) * weights) / mol.nelectron)
    except Exception as e:
        print(f"[{MOL}] CC EXCEPTION: {type(e).__name__}: {e}", flush=True)

    secs = time.time() - t0
    with open(OUT, "a") as f:
        f.write(f"{MOL},{spin},{mol.nelectron},{error:.12f},{dft_conv},{cc_conv},{secs:.1f}\n")
    print(f"[{MOL}] error={error:.6f} dft_conv={dft_conv} cc_conv={cc_conv} {secs:.1f}s", flush=True)
    if os.path.exists(sp):
        os.remove(sp)
    sys.exit(0)
