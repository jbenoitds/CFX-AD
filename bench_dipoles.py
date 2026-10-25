"""
Dipole moments of the 28 diatomics of Roy, Cuierrier and Ernzerhof 2023 (J. Phys. Chem. A
127, 2026), Table 3.

  mu_DFT  : from the self-consistent density matrix of the CF functional
  mu_CCSD : from the CCSD relaxed 1-RDM (solve_lambda + make_rdm1(ao_repr=True)),
            the same object bench_density.py --cc builds
  check   : PBE and SCAN dipoles from plain PySCF, to compare with the PBE and SCAN
            entries of Table 3 of Roy 2023 (0.185 / 0.109 D)
  MAE     : mean_i | |mu_DFT| - |mu_CCSD| |   (the signed error is also reported)

Four modes. The CCSD reference and the PBE/SCAN check do not depend on the functional and are
shipped in data/; the CF stream runs in its own process, so that JAX and the CCSD never hold
memory at the same time:

  python -u bench_dipoles.py --gate <MOL|all>  # PBE + SCAN           -> data/dipoles_gate_<tag>.csv
  python -u bench_dipoles.py --cc   <MOL|all>  # CCSD reference       -> data/dipoles_cc_<tag>.csv
  python -u bench_dipoles.py --cf   <MOL|all>  # CF functional        -> dipoles_cf_<tag>_lambd<L>.csv
  python -u bench_dipoles.py --merge           # join + report        -> dipoles_<tag>_lambd<L>.csv + .md

Each stream is resumable. A list of molecules runs in one process, so the JAX compilation is
done once.

Config via environment:
  CFBASIS  basis           (default aug-cc-pVTZ)
  CFLAMBD  variant         (0.0 = CFX (default), 0.2 = CFXmix)
  CFFROZEN "core" (default here) | "0" all-electron  -- CCSD core treatment
  CFMAXMEM PySCF max_memory in MB for the --gate and --cc streams (default 1800)
"""
import sys, os, time, types, re
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

BASIS  = os.environ.get("CFBASIS", "aug-cc-pVTZ")
FROZEN = os.environ.get("CFFROZEN", "core")           # "0"=all-electron CCSD, "core"=frozen-core
LAMBD  = float(os.environ.get("CFLAMBD", "0.0"))
_variant  = "CFX" if LAMBD == 0.0 else ("CFXmix" if LAMBD == 0.2 else f"lambda={LAMBD}")
_basistag = BASIS.lower().replace("-", "").replace(".", "")
_ftag     = "" if FROZEN == "core" else f"_frz{FROZEN}"      # default (core) keeps the plain name
GATE_OUT  = f"{ROOT}/data/dipoles_gate_{_basistag}.csv"      # PBE + SCAN  (no lambda dependence)
CC_OUT    = f"{ROOT}/data/dipoles_cc_{_basistag}{_ftag}.csv" # CCSD relaxed reference
CF_OUT    = f"{ROOT}/dipoles_cf_{_basistag}_lambd{LAMBD}.csv"
MERGED    = f"{ROOT}/dipoles_{_basistag}_lambd{LAMBD}{_ftag}.csv"
REPORT    = f"{ROOT}/dipole_report_{_basistag}_lambd{LAMBD}{_ftag}.md"

# 28-molecule diatomic set, the same as in bench_density.py (which cannot be imported: it
# parses sys.argv at module level).  name -> (spin=2S, bond length / Angstrom).
DIATOMICS = {
    'LiH':(0,1.595), 'BH':(0,1.232), 'CN':(1,1.172), 'OH':(1,0.969), 'LiO':(1,1.695),
    'CO':(0,1.128),  'NO':(1,1.151), 'HF':(0,0.917), 'LiF':(0,1.564), 'BF':(0,1.263),
    'CF':(1,1.272),  'FO':(1,1.354), 'NaF':(0,1.926),'AlF':(0,1.654), 'SiO':(0,1.509),
    'PN':(0,1.491),  'PO':(1,1.476), 'HS':(1,1.341), 'SiS':(0,1.929), 'CS':(0,1.535),
    'NS':(1,1.494),  'SO':(2,1.481), 'SF':(1,1.601), 'HCl':(0,1.275), 'NaCl':(0,2.361),
    'LiCl':(0,2.021),'ClO':(1,1.570), 'ClF':(0,1.628),   # ClF: the 28th, from the SI of Roy 2023
}
ORDER = list(DIATOMICS.keys())

import numpy as np


def build_mol(MOL, verbose=3):
    """atom1 at the origin, atom2 at (0,0,R), as in bench_density.build_mol."""
    from pyscf import gto
    spin, R = DIATOMICS[MOL]
    positions = [[0.0, 0.0, 0.0], [0.0, 0.0, R]]
    atom_list = []
    for el, count in re.findall(r'([A-Z][a-z]*)(\d*)', MOL):
        atom_list.extend([el] * (int(count) if count else 1))
    molecule = [[a, tuple(positions[i])] for i, a in enumerate(atom_list)]
    return gto.M(atom=molecule, spin=spin, basis=BASIS, verbose=verbose), spin


def dipole_debye(mol, dm):
    """Total dipole (Debye) from an AO density matrix, origin at (0,0,0) = atom 1.
    All 28 species are neutral, so the value does not depend on the origin.
    Returns the full vector."""
    from pyscf.data import nist
    dm = np.asarray(dm)
    if dm.ndim == 3:                                   # unrestricted: (2, n, n)
        dm = dm[0] + dm[1]
    with mol.with_common_orig((0.0, 0.0, 0.0)):
        ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    el   = -np.einsum('xij,ji->x', ao_dip, dm)         # electronic (electron charge = -1)
    nucl = np.einsum('i,ix->x', mol.atom_charges(), mol.atom_coords())
    return (el + nucl) * nist.AU2DEBYE


def _ensure(path, header_comment, columns):
    if not os.path.exists(path):
        with open(path, "a") as f:
            f.write(header_comment + "\n")
            f.write(columns + "\n")


def _done(path):
    d = set()
    if os.path.exists(path):
        for ln in open(path):
            if ln.startswith(("#", "molecule", "MAE")) or "," not in ln:
                continue
            d.add(ln.split(",")[0])
    return d


# ============================== MERGE / REPORT ==============================
def merge():
    def load(path, keycols):
        rows = {}
        if not os.path.exists(path):
            return rows
        for ln in open(path):
            if ln.startswith(("#", "molecule", "MAE")) or "," not in ln:
                continue
            p = ln.strip().split(",")
            rows[p[0]] = dict(zip(keycols, p))
        return rows

    gate = load(GATE_OUT, ["molecule", "spin", "nelec", "mu_pbe", "mu_scan",
                           "pbe_converged", "scan_converged", "secs"])
    ccr  = load(CC_OUT,   ["molecule", "spin", "nelec", "mu_ccsd", "cc_converged",
                           "ncore", "secs"])
    ref = {m: {**ccr.get(m, {}), **gate.get(m, {})} for m in set(ccr) | set(gate)}
    cf  = load(CF_OUT,  ["molecule", "spin", "nelec", "mu_cf", "cf_converged", "how", "secs"])

    L, A = [], None
    L.append(f"# Dipole benchmark -- basis={BASIS}  LAMBD={LAMBD} ({_variant})  CCSD core={FROZEN}")
    L.append("")
    L.append("Reference = CCSD relaxed 1-RDM (solve_lambda -> make_rdm1(ao_repr=True)).")
    L.append("Signed convention: mu_z with atom 1 at the origin and atom 2 at +z "
             "(so mu_z > 0 means the negative end sits on atom 1).")
    L.append("")
    L.append("| molecule | spin | mu_CCSD | mu_CF | mu_PBE | mu_SCAN | signed err CF | |err| CF |")
    L.append("|---|---|---|---|---|---|---|---|")

    errs = {k: [] for k in ("cf", "pbe", "scan")}
    signed = []
    nrow = 0
    for m in ORDER:
        r, c = ref.get(m), cf.get(m)
        if not r or r.get("cc_converged") != "True":
            L.append(f"| {m} | {DIATOMICS[m][0]} | (no CCSD ref) | | | | | |")
            continue
        mu_ref = float(r["mu_ccsd"])
        cells = [f"{mu_ref:+.4f}"]
        vals = {}
        for tag, src, key, conv in (("cf", c, "mu_cf", "cf_converged"),
                                    ("pbe", r, "mu_pbe", "pbe_converged"),
                                    ("scan", r, "mu_scan", "scan_converged")):
            if src and src.get(conv) == "True" and src[key] not in ("nan", ""):
                v = float(src[key]); vals[tag] = v
                errs[tag].append(abs(abs(v) - abs(mu_ref)))
            else:
                vals[tag] = None
        for tag in ("cf", "pbe", "scan"):
            cells.append(f"{vals[tag]:+.4f}" if vals[tag] is not None else "-")
        if vals["cf"] is not None:
            s = abs(vals["cf"]) - abs(mu_ref)
            signed.append((m, s))
            cells.append(f"{s:+.4f}"); cells.append(f"{abs(s):.4f}")
        else:
            cells.append("-"); cells.append("-")
        L.append(f"| {m} | {DIATOMICS[m][0]} | " + " | ".join(cells) + " |")
        nrow += 1

    L.append("")
    for tag, name in (("cf", f"CF (LAMBD={LAMBD}, {_variant})"), ("pbe", "PBE"), ("scan", "SCAN")):
        e = errs[tag]
        mae = float(np.mean(e)) if e else float("nan")
        L.append(f"- **MAE {name} = {mae:.4f} D** over {len(e)}/28 molecules")
    L.append("")
    if signed:
        over  = [m for m, s in signed if s > 0]
        under = [m for m, s in signed if s < 0]
        L.append(f"- Signed: CF over-polarizes {len(over)}/{len(signed)} "
                 f"({', '.join(over)}), under-polarizes {len(under)} ({', '.join(under)}).")
        L.append(f"- Mean signed error (|mu_CF| - |mu_CCSD|) = "
                 f"{np.mean([s for _, s in signed]):+.4f} D")
    txt = "\n".join(L)
    open(REPORT, "w").write(txt + "\n")

    with open(MERGED, "w") as f:
        f.write(f"# DIPOLES | basis={BASIS} | LAMBD={LAMBD} ({_variant}) | "
                f"ref=CCSD relaxed (solve_lambda, frozen={FROZEN}) | Debye | origin=(0,0,0)=atom1\n")
        f.write("molecule,spin,mu_ccsd,mu_cf,mu_pbe,mu_scan,signed_err_cf\n")
        for m in ORDER:
            r, c = ref.get(m), cf.get(m)
            if not r:
                continue
            g = lambda src, k: (src[k] if src and src.get(k) not in (None, "") else "nan")
            mu_ref = g(r, "mu_ccsd"); mu_cf = g(c, "mu_cf")
            try:
                se = f"{abs(float(mu_cf)) - abs(float(mu_ref)):+.6f}"
            except (ValueError, TypeError):
                se = "nan"
            f.write(f"{m},{DIATOMICS[m][0]},{mu_ref},{mu_cf},{g(r,'mu_pbe')},{g(r,'mu_scan')},{se}\n")
    print(txt)
    print(f"\n[written] {MERGED}\n[written] {REPORT}", flush=True)
    return


mode = sys.argv[1] if len(sys.argv) > 1 else "--merge"
if mode == "--merge":
    merge(); sys.exit(0)
if mode not in ("--gate", "--cc", "--cf"):
    sys.exit("usage: bench_dipoles.py [--gate|--cc|--cf] <MOL>|all [...] | --merge")
_args = sys.argv[2:] or ["all"]
MOLS = ORDER if _args == ["all"] else _args
for _m in MOLS:
    if _m not in DIATOMICS:
        sys.exit(f"unknown molecule {_m!r}; choose from {ORDER}")


# ==================== PBE/SCAN check and CCSD reference (PySCF only) ====================
# Kept as separate streams: PySCF may use up to max_memory in each stage, so stacking PBE,
# SCAN and CCSD in one process can exceed the memory of a small machine.
MAXMEM = float(os.environ.get("CFMAXMEM", "1800"))


def run_gate(MOL):
    from pyscf import dft
    t0 = time.time()
    mol, spin = build_mol(MOL)
    mu_pbe = mu_scan = float("nan")
    pbe_conv = scan_conv = False
    for xc in ("pbe", "scan"):
        try:
            m = dft.UKS(mol, xc=xc) if spin > 0 else dft.RKS(mol, xc=xc)
            m.conv_tol = 1e-9; m.max_cycle = 200; m.max_memory = MAXMEM
            m.kernel()
            if not m.converged:                       # SOSCF fallback
                dm = m.make_rdm1()
                m2 = (dft.UKS(mol, xc=xc) if spin > 0 else dft.RKS(mol, xc=xc)).newton()
                m2.max_cycle = 100; m2.max_memory = MAXMEM; m2.kernel(dm0=dm); m = m2
            v = dipole_debye(mol, m.make_rdm1())[2]
            if xc == "pbe":
                mu_pbe, pbe_conv = v, bool(m.converged)
            else:
                mu_scan, scan_conv = v, bool(m.converged)
            print(f"[{MOL}] {xc}: mu_z={v:+.4f} D conv={m.converged}", flush=True)
        except Exception as e:
            print(f"[{MOL}] {xc} EXCEPTION: {type(e).__name__}: {e}", flush=True)
    with open(GATE_OUT, "a") as f:
        f.write(f"{MOL},{spin},{mol.nelectron},{mu_pbe:.6f},{mu_scan:.6f},"
                f"{pbe_conv},{scan_conv},{time.time()-t0:.1f}\n")
    print(f"[{MOL}] gate done {time.time()-t0:.1f}s", flush=True)


def run_cc(MOL):
    from pyscf import cc
    t0 = time.time()
    mol, spin = build_mol(MOL)
    mu_cc = float("nan"); cc_conv = False; ncore = 0
    try:
        mf_hf = mol.UHF() if spin > 0 else mol.RHF()
        mf_hf.conv_tol = 1e-9; mf_hf.max_memory = MAXMEM; mf_hf.run()
        if FROZEN == "core":
            from pyscf.data.elements import chemcore
            ncore = int(chemcore(mol))
        mycc = cc.CCSD(mf_hf, frozen=ncore) if ncore else cc.CCSD(mf_hf)
        mycc.max_memory = MAXMEM; mycc.run()
        mycc.solve_lambda()
        cc_conv = bool(getattr(mycc, "converged", True))
        mu_cc = dipole_debye(mol, np.asarray(mycc.make_rdm1(ao_repr=True)))[2]
        print(f"[{MOL}] CCSD(relaxed): mu_z={mu_cc:+.4f} D conv={cc_conv} ncore={ncore}", flush=True)
    except Exception as e:
        print(f"[{MOL}] CCSD EXCEPTION: {type(e).__name__}: {e}", flush=True)
    with open(CC_OUT, "a") as f:
        f.write(f"{MOL},{spin},{mol.nelectron},{mu_cc:.6f},{cc_conv},{ncore},"
                f"{time.time()-t0:.1f}\n")
    print(f"[{MOL}] cc done {time.time()-t0:.1f}s", flush=True)


if mode == "--gate":
    _ensure(GATE_OUT,
            f"# dipole check (plain PySCF PBE/SCAN) | basis={BASIS} "
            f"| mu_z in Debye | origin=(0,0,0)=atom1",
            "molecule,spin,nelec,mu_pbe,mu_scan,pbe_converged,scan_converged,secs")
    for _mol in MOLS:
        if _mol in _done(GATE_OUT):
            print(f"[{_mol}] already in gate CSV; skip.", flush=True); continue
        run_gate(_mol)
    sys.exit(0)

if mode == "--cc":
    _ensure(CC_OUT,
            f"# dipole CCSD reference | basis={BASIS} | relaxed 1-RDM "
            f"(solve_lambda, frozen={FROZEN}) | mu_z in Debye | origin=(0,0,0)=atom1",
            "molecule,spin,nelec,mu_ccsd,cc_converged,ncore,secs")
    for _mol in MOLS:
        if _mol in _done(CC_OUT):
            print(f"[{_mol}] already in cc CSV; skip.", flush=True); continue
        run_cc(_mol)
    sys.exit(0)


# ==================== CF functional (JAX) ====================
if mode == "--cf":
    _ensure(CF_OUT,
            f"# dipole, CF functional | basis={BASIS} | LAMBD={LAMBD} ({_variant}) "
            f"| mu_z in Debye | origin=(0,0,0)=atom1",
            "molecule,spin,nelec,mu_cf,cf_converged,how,secs")
    from jax import config
    config.update("jax_enable_x64", True)
    import CFXm_jax
    CFXm_jax.LAMBD = LAMBD                     # must be set before the first traced call
    from pyscf.dft import UKS, RKS
    from get_veff import U_get_veff, R_get_veff


def run_cf(MOL):
    t0 = time.time()
    mol, spin = build_mol(MOL)

    def make_dft():
        if spin > 0:
            m = UKS(mol); m.get_veff = types.MethodType(U_get_veff, m)
        else:
            m = RKS(mol); m.get_veff = types.MethodType(R_get_veff, m)
        m.max_cycle = 200; m.conv_tol = 1e-8
        return m

    def converge():
        """Open shells try quasi-second-order SCF first (DIIS tends to oscillate for these
        radicals); then a clean DIIS run, a level-shift + polish, and SOSCF as the last resort."""
        if spin > 0:
            m = make_dft().newton(); m.max_cycle = 100; m.kernel()
            if m.converged:
                return m, "newton"
        mf = make_dft()
        mf.level_shift = 0.0; mf.damp = 0.0; mf.kernel()
        if mf.converged:
            return mf, "clean"
        dm = mf.make_rdm1(); mf.level_shift = 0.2; mf.damp = 0.4; mf.kernel(dm)
        dm = mf.make_rdm1(); mf.level_shift = 0.0; mf.damp = 0.0; mf.kernel(dm)
        if mf.converged:
            return mf, "stab+polish"
        dm = mf.make_rdm1(); m2 = make_dft().newton(); m2.max_cycle = 100
        m2.kernel(dm0=dm)
        return m2, "newton(fallback)"

    mu = float("nan"); conv = False; how = "-"
    try:
        mf, how = converge()
        conv = bool(mf.converged)
        mu = dipole_debye(mol, np.asarray(mf.make_rdm1()))[2]
    except Exception as e:
        how = f"EXC:{type(e).__name__}"
        print(f"[{MOL}] CF EXCEPTION: {type(e).__name__}: {e}", flush=True)

    secs = time.time() - t0
    with open(CF_OUT, "a") as f:
        f.write(f"{MOL},{spin},{mol.nelectron},{mu:.6f},{conv},{how},{secs:.1f}\n")
    print(f"[{MOL}] CF mu_z={mu:+.4f} D conv={conv} how={how} {secs:.1f}s", flush=True)


if mode == "--cf":
    for _mol in MOLS:
        if _mol in _done(CF_OUT):
            print(f"[{_mol}] already in cf CSV; skip.", flush=True); continue
        run_cf(_mol)
    sys.exit(0)
