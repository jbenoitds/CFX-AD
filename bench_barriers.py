"""
Barrier heights of BH76 (GMTKN55), as in Table 3 of Roy, Cuierrier and Ernzerhof 2023
(J. Phys. Chem. A 127, 2026).

  76 barriers over 79 unique species, single points on the GMTKN55 geometries.
  barrier_j = sum_k coef_k * E(sys_k),  627.5095 kcal/mol per Hartree.
  Reference = W2-F12 (GMTKN55).  MAD = plain mean over all 76 rows (duplicates retained).

Method selected by CFMETHOD:
  CFMETHOD=cf            -> the CF functional (variant from CFLAMBD)
  CFMETHOD=pbe | scan |… -> plain PySCF DFT, to check the set-up against GMTKN55's PBE values

Config via environment:
  CFBASIS   basis        (default cc-pVTZ)
  CFLAMBD   CF variant   (0.0 = CFX (default), 0.2 = CFXmix); ignored unless CFMETHOD=cf
  CFGRID    grid level   (unset = PySCF default 3)
  CFMETHOD  see above    (default cf)
  CFMAXMEM  PySCF max_memory in MB (default 2200)
  CFDF      1 = RI-J density fitting, plain-PySCF methods only (default 0)
  CFCHUNK   grid points per JAX batch (default 20000); lower it to reduce peak memory,
            results are unchanged
  CFSPECSEC wall-clock limit per species in s (default 5400; 0 = none)
  CFMAXSPEC species per process before a clean exit (default 0 = no limit)
  CFDM0     npz from CFSAVEDM: start the SCF from this density (e.g. a converged CFXmix
            density as the starting point of a CFX run)
  CFSAVEDM  write the converged density to this npz
  CFRESCUE  "pbe" = after every other strategy has failed, one more SOSCF run from a converged
            PBE density (default off). Rows from it are tagged pbe-rescue in the CSV.
Output: barriers_<basistag>_<method>.csv  -- one row per species, resumable.

Usage:
  python -u bench_barriers.py --all             # every species (resumable)
  python -u bench_barriers.py --sys <NAME> ...  # one or more species (names are case-sensitive)
  python -u bench_barriers.py --list            # print the species list, smallest first
  python -u bench_barriers.py --report          # barriers + MAD -> barrier_report_<tag>.md
"""
import sys, os, time, types
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import bh76_lib as B

BASIS  = os.environ.get("CFBASIS", "cc-pVTZ")
LAMBD  = float(os.environ.get("CFLAMBD", "0.0"))
GRID   = os.environ.get("CFGRID", "")
METHOD = os.environ.get("CFMETHOD", "cf").lower()
MAXMEM = float(os.environ.get("CFMAXMEM", "2200"))   # PySCF max_memory (MB)
# Density fitting of the Coulomb term, for the plain-PySCF check only. DF and non-DF energies
# go to separate CSVs and must not be mixed in one barrier set.
USEDF  = os.environ.get("CFDF", "0") not in ("0", "", "false", "False")
DM0IN  = os.environ.get("CFDM0", "")        # npz written by CFSAVEDM: start the SCF here
DMOUT  = os.environ.get("CFSAVEDM", "")     # write the converged density to this npz
RESCUE = os.environ.get("CFRESCUE", "").lower()  # "pbe" = last SOSCF attempt from a PBE density
if USEDF and METHOD == "cf":
    sys.exit("CFDF is for the plain-PySCF methods only; the CF path builds its own Fock matrix.")
_variant = ("CFX" if LAMBD == 0.0 else ("CFXmix" if LAMBD == 0.2 else f"lambda={LAMBD}")) \
           if METHOD == "cf" else METHOD.upper()
_basistag = BASIS.lower().replace("-", "").replace(".", "")
_mtag = f"cf_lambd{LAMBD}" if METHOD == "cf" else METHOD
if USEDF:
    _mtag += "_df"
_gtag = f"_grid{GRID}" if GRID else ""
OUT    = f"{ROOT}/barriers_{_basistag}_{_mtag}{_gtag}.csv"
REPORT = f"{ROOT}/barrier_report_{_basistag}_{_mtag}{_gtag}.md"

SPECIES = B.load_species()
RXNS    = B.load_reactions()
UNIQ    = B.unique_species(RXNS)


def _ensure():
    if not os.path.exists(OUT):
        with open(OUT, "a") as f:
            f.write(f"# BH76 barrier heights | basis={BASIS} | method={METHOD} ({_variant}) | "
                    f"grid={GRID or 'default(3)'} | geometries=GMTKN55 (Bohr) | "
                    f"E in Hartree | {B.HARTREE2KCAL} kcal/mol per Ha\n")
            f.write("species,charge,mult,natom,nbas,E_tot_Ha,converged,how,secs\n")


def _done():
    d = {}
    if os.path.exists(OUT):
        for ln in open(OUT):
            if ln.startswith(("#", "species")) or "," not in ln:
                continue
            p = ln.strip().split(",")
            d[p[0]] = (float(p[5]), p[6] == "True")
    return d


# ------------------------------------------------------------------ report
def report():
    import numpy as np
    E = _done()
    L, A = [], None
    A = L.append
    A(f"# BH76 barrier heights -- basis={BASIS}  method={METHOD} ({_variant})  "
      f"grid={GRID or 'default(3)'}")
    A("")
    A("Reference = W2-F12 (GMTKN55). Single points on the GMTKN55 geometries; no optimization.")
    A("Rows 15/16, 19/20, 47/48, 75/76 are duplicates by construction and are retained: "
      "the MAD is a plain mean over all 76.")
    A("")
    A("| # | reaction | ref (W2-F12) | this work | error |")
    A("|---|----------|--------------|-----------|-------|")
    errs, nht, ht, missing = [], [], [], set()
    per = []
    for idx, terms, ref, pbe_dev in RXNS:
        miss = [n for n, _ in terms if n not in E or not E[n][1]]
        label = " ".join(f"{'+' if c > 0 else '-'}{n}" for n, c in terms)
        if miss:
            missing.update(miss)
            A(f"| {idx} | {label} | {ref:.1f} | (missing {','.join(miss)}) | - |")
            per.append((idx, ref, None, None, pbe_dev))
            continue
        val = sum(c * E[n][0] for n, c in terms) * B.HARTREE2KCAL
        err = val - ref
        errs.append(abs(err))
        (nht if idx <= 38 else ht).append(abs(err))
        per.append((idx, ref, val, err, pbe_dev))
        A(f"| {idx} | {label} | {ref:.1f} | {val:.2f} | {err:+.2f} |")
    A("")
    n = len(errs)
    mad = float(np.mean(errs)) if errs else float("nan")
    signed = [p[3] for p in per if p[3] is not None]
    A(f"- **MAD over {n}/76 barriers = {mad:.2f} kcal/mol**"
      + ("" if n == 76 else "  (subset: not comparable to a 76-barrier MAD)"))
    A(f"- MD (signed) = {np.mean(signed):+.2f} · RMSD = {np.sqrt(np.mean(np.square(signed))):.2f}")
    A(f"- NHTBH38 (rows 1-38, heavy-atom/SN2/unimolecular) MAD = "
      f"{np.mean(nht):.2f} over {len(nht)}/38")
    A(f"- HTBH38 (rows 39-76, hydrogen transfer) MAD = {np.mean(ht):.2f} over {len(ht)}/38")
    if missing:
        A("")
        A(f"**Species missing / unconverged ({len(missing)}):** {' '.join(sorted(missing))}")

    # --- PBE check: per-barrier agreement with GMTKN55's own PBE deviations
    if METHOD == "pbe":
        A("")
        A("## PBE check -- per-barrier vs GMTKN55's own PBE deviations")
        A("")
        A("| # | GMTKN55 dev | our dev | discrepancy |")
        A("|---|-------------|---------|-------------|")
        disc = []
        for idx, ref, val, err, pbe_dev in per:
            if err is None:
                continue
            d = err - pbe_dev
            disc.append(abs(d))
            A(f"| {idx} | {pbe_dev:+.2f} | {err:+.2f} | {d:+.2f} |")
        if disc:
            A("")
            A(f"- **max |discrepancy| = {max(disc):.2f}**, mean = {np.mean(disc):.2f} kcal/mol "
              f"over {len(disc)} barriers")
            A(f"- GMTKN55's own totals: MD -9.11, MAD **9.15**, RMSD 10.39")
    txt = "\n".join(L)
    open(REPORT, "w").write(txt + "\n")
    print(txt)
    print(f"\n[report written] {REPORT}", flush=True)


mode = sys.argv[1] if len(sys.argv) > 1 else "--all"
if mode == "--report":
    report(); sys.exit(0)
if mode == "--list":
    # smallest first: atoms, then by atom count
    for s in sorted(UNIQ, key=lambda s: (len(B.parse_coord(s)), s)):
        c, m = SPECIES[s]
        print(f"{s}\t{len(B.parse_coord(s))}\tq={c}\tmult={m}")
    sys.exit(0)

# ------------------------------------------------------------------ compute
_ensure()
import numpy as onp

if METHOD == "cf":
    from jax import config
    config.update("jax_enable_x64", True)
    import CFXm_jax
    CFXm_jax.LAMBD = LAMBD           # must be set before the first traced call
    # CHUNK only sets how many grid points are evaluated at once; results are unchanged.
    if os.environ.get("CFCHUNK"):
        CFXm_jax.CHUNK = int(os.environ["CFCHUNK"])
        print(f"[bench_barriers] CFXm_jax.CHUNK = {CFXm_jax.CHUNK}", flush=True)
    from get_veff import U_get_veff, R_get_veff
from pyscf.dft import UKS, RKS


def make_mf(mol, spin):
    if spin > 0:
        m = UKS(mol)
        if METHOD == "cf":
            m.get_veff = types.MethodType(U_get_veff, m)
        else:
            m.xc = METHOD
    else:
        m = RKS(mol)
        if METHOD == "cf":
            m.get_veff = types.MethodType(R_get_veff, m)
        else:
            m.xc = METHOD
    m.max_cycle = 100; m.conv_tol = 1e-9; m.diis_space = 8
    m.max_memory = MAXMEM
    if GRID:
        m.grids.level = int(GRID)
    if USEDF:
        m = m.density_fit()          # RI-J; auxbasis auto-selected to match the orbital basis
        m.max_memory = MAXMEM
    return m


def _save_dm(mf):
    """Write the converged density matrix so that another process can start from it.
    LAMBD is fixed when JAX first traces the functional, so a run at another lambda has to be
    a separate process reading this file."""
    import numpy as np
    np.savez(DMOUT, dm=np.asarray(mf.make_rdm1()), lambd=float(LAMBD), basis=str(BASIS))
    print("[bench_barriers] saved converged density -> %s (lambda=%s, basis=%s)"
          % (DMOUT, LAMBD, BASIS), flush=True)


def _load_dm0():
    """Load a density saved by _save_dm; refuse it if it was computed in another basis."""
    import numpy as np
    z = np.load(DM0IN, allow_pickle=False)
    got = str(z["basis"])
    if got != str(BASIS):
        sys.exit("REFUSING stored density %s: basis %r, this run is %r"
                 % (DM0IN, got, str(BASIS)))
    lam = float(z["lambd"])
    print("[bench_barriers] starting from stored density %s (converged at lambda=%g)"
          % (DM0IN, lam), flush=True)
    return z["dm"], lam


def _ret(mol, mf, conv, how):
    """Single exit point for run_species, so CFSAVEDM has the converged mf to save."""
    if DMOUT and conv:
        _save_dm(mf)
    return mol, mf.e_tot, conv, how


def pbe_guess_dm(mol, spin):
    """A converged PBE density matrix, used as a starting point by the CFRESCUE step.
    For this functional a PBE start can lead to a different SCF solution, so it is only tried
    after every other strategy has failed, and the resulting energy should be checked."""
    m = UKS(mol) if spin > 0 else RKS(mol)
    m.xc = "pbe"
    m.max_cycle = 50
    m.conv_tol = 1e-6          # a starting guess needs no more than this
    m.max_memory = MAXMEM
    if GRID:
        m.grids.level = int(GRID)
    m.kernel()
    return m.make_rdm1()


def run_species(name):
    """SCF strategy: open shells try quasi-second-order SCF (newton) first, since DIIS tends to
    oscillate for near-degenerate open shells; then a clean DIIS run, then a level shift with
    damping followed by an unshifted polish, then SOSCF as the last resort."""
    mol = B.build_mol(name, BASIS, verbose=0, species=SPECIES)
    spin = mol.spin
    if DM0IN:
        dm0, _lam = _load_dm0()
        tag = "+dm%g" % _lam
    else:
        dm0, tag = None, ""
    if spin > 0:
        mf = make_mf(mol, spin).newton()
        mf.max_cycle = 100
        mf.kernel(dm0=dm0)
        if mf.converged:
            return _ret(mol, mf, True, "newton" + tag)
        mf = make_mf(mol, spin)
    else:
        mf = make_mf(mol, spin)
    mf.level_shift = 0.0; mf.damp = 0.0
    mf.kernel(dm0=dm0)
    if mf.converged:
        return _ret(mol, mf, True, "clean" + tag)
    dm = mf.make_rdm1(); mf.level_shift = 0.2; mf.damp = 0.4; mf.kernel(dm)
    dm = mf.make_rdm1(); mf.level_shift = 0.0; mf.damp = 0.0; mf.kernel(dm)
    if mf.converged:
        return _ret(mol, mf, True, "stab+polish" + tag)
    dm = mf.make_rdm1()
    m2 = make_mf(mol, spin).newton(); m2.max_cycle = 100
    m2.kernel(dm0=dm)                                  # the newton() object takes the guess as dm0=
    if m2.converged or RESCUE != "pbe":
        return _ret(mol, m2, bool(m2.converged), "newton(fallback)" + tag)

    # ---- CFRESCUE=pbe: SOSCF from a converged PBE density, reached only after every step above
    # has failed. Converging is not proof of the right state: check a pbe-rescue energy (for
    # example against the same species at the other lambda) before using it.
    m3 = make_mf(mol, spin).newton(); m3.max_cycle = 100
    m3.kernel(dm0=pbe_guess_dm(mol, spin))
    if m3.converged:
        return _ret(mol, m3, True, "pbe-rescue" + tag)
    return _ret(mol, m2, False, "newton(fallback)+pbefail" + tag)


if mode == "--all":
    # smallest first: atom count, then basis functions
    def _size(s):
        try:
            m = B.build_mol(s, BASIS, verbose=0, species=SPECIES)
            return (len(B.parse_coord(s)), m.nao_nr(), s)
        except Exception:
            return (len(B.parse_coord(s)), 10 ** 6, s)
    targets = sorted(UNIQ, key=_size)
elif mode == "--sys":
    # a list of species shares one JAX compilation
    targets = sys.argv[2:]
    bad = [t for t in targets if t not in UNIQ]
    if not targets or bad:
        sys.exit(f"unknown species {bad!r} (names are case-sensitive; "
                 f"note C2H5 != c2h5). Run --list.")
else:
    sys.exit(__doc__)

done = _done()
print(f"[bench_barriers] basis={BASIS} method={METHOD} ({_variant}) grid={GRID or 'default'} "
      f"-> {OUT}\n  {len(done)} already done, {len(targets)} targets", flush=True)

# Per-species wall-clock limit (SIGALRM): a species that exceeds it is recorded as a TIMEOUT
# failure and the batch moves on.
SPECSEC = int(os.environ.get("CFSPECSEC", "5400"))
INFLIGHT = OUT + ".inflight"


class _SpeciesTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise _SpeciesTimeout()


if SPECSEC > 0:
    import signal
    signal.signal(signal.SIGALRM, _alarm)

# Species per process. JAX keeps one compiled executable per grid shape and every species has
# its own grid, so memory grows over a long process; after CFMAXSPEC species the process exits
# and the driver (run_bh76.sh) starts a fresh one. 0 = no cap.
MAXSPEC = int(os.environ.get("CFMAXSPEC", "0"))
_ndone_this_pass = 0

for name in targets:
    if name in done:
        print(f"[{name}] already done ({done[name][0]:.6f}); skip", flush=True)
        continue
    if MAXSPEC > 0 and _ndone_this_pass >= MAXSPEC:
        print(f"[bench_barriers] reached CFMAXSPEC={MAXSPEC} for this process; "
              f"exiting so the driver can restart with a fresh JAX cache", flush=True)
        sys.exit(0)
    t0 = time.time()
    charge, mult = SPECIES[name]
    nbas = -1
    # Name of the species in progress. If the process is killed (out of memory), run_bh76.sh
    # reads it to record which species died.
    with open(INFLIGHT, "w") as f:
        f.write(name + "\n")
    try:
        if SPECSEC > 0:
            signal.alarm(SPECSEC)
        mol, e, conv, how = run_species(name)
        nbas = mol.nao_nr()
    except _SpeciesTimeout:
        e, conv, how = float("nan"), False, f"TIMEOUT:{SPECSEC}s"
        print(f"[{name}] TIMEOUT after {SPECSEC}s", flush=True)
    except Exception as ex:
        e, conv, how = float("nan"), False, f"EXC:{type(ex).__name__}"
        print(f"[{name}] EXCEPTION: {type(ex).__name__}: {ex}", flush=True)
    finally:
        if SPECSEC > 0:
            signal.alarm(0)
    secs = time.time() - t0
    with open(OUT, "a") as f:
        f.write(f"{name},{charge},{mult},{len(B.parse_coord(name))},{nbas},"
                f"{e:.10f},{conv},{how},{secs:.1f}\n")
    print(f"[{name}] E={e:.8f} conv={conv} how={how} nbas={nbas} {secs:.1f}s", flush=True)
    if os.path.exists(INFLIGHT):
        os.remove(INFLIGHT)          # species accounted for (converged or recorded as failed)
    _ndone_this_pass += 1

print("[bench_barriers] ALL DONE", flush=True)
