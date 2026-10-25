"""
Atomic energies and atomization energies, as in Roy, Cuierrier and Ernzerhof 2023
(J. Phys. Chem. A 127, 2026), Tables 2 and 3.

  * Atomic energies (Hartree) -- 18 atoms H..Ar, MAE vs exact non-relativistic
    total energies.
  * Atomization energies (kcal/mol) -- the 19 molecules of Table 2,
    AE = (sum_atoms E_atom - E_mol) * 627.5, MAE vs experiment, reported as
    single-bonded MAE / multiple-bonded MAE / TMAE.

Config via environment:
  CFBASIS  basis set          (default cc-pVTZ)
  CFLAMBD  functional variant (0.0 = CFX (default), 0.2 = CFXmix)
  CFGRID   grid level         (unset = PySCF default 3)
Output: energies_<basistag>_lambd<L>.csv  (one row per system, resumable)

Usage:
  python -u bench_energies.py --all            # run every atom + molecule (resumable)
  python -u bench_energies.py --sys <NAME>     # run one system
  python -u bench_energies.py --report         # build the MAE report from the CSV
"""
import sys, os, time, types, re
from collections import Counter
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

BASIS  = os.environ.get("CFBASIS", "cc-pVTZ")
LAMBD  = float(os.environ.get("CFLAMBD", "0.0"))
GRID   = os.environ.get("CFGRID", "")
_variant  = "CFX" if LAMBD == 0.0 else ("CFXmix" if LAMBD == 0.2 else f"lambda={LAMBD}")
_basistag = BASIS.lower().replace("-", "").replace(".", "")
_gtag     = f"_grid{GRID}" if GRID else ""
OUT    = f"{ROOT}/energies_{_basistag}_lambd{LAMBD}{_gtag}.csv"
HARTREE2KCAL = 627.5

# ------------------------------------------------------------------ systems
# 18 atoms H..Ar with ground-state spin (2S)
ATOMS = {"H":1,"He":0,"Li":1,"Be":0,"B":1,"C":2,"N":3,"O":2,"F":1,"Ne":0,
         "Na":1,"Mg":0,"Al":1,"Si":2,"P":3,"S":2,"Cl":1,"Ar":0}

# Molecules: name -> [spin, coordinates in Angstrom]; the set of Table 2 of Roy 2023
MOLECULES = {
 'HH':     [0, [[0,0,-0.0259603084],[0,0,0.7259603084]]],
 'LiH':    [0, [[0,0,0.3962543501],[0,0,2.0037456499]]],
 'CHHHH':  [0, [[-1.629e-06,0.0,7.8502e-06],[-2.2937e-06,0.0,1.0970267936],
                [1.0342803963,0.0,-0.3656807611],[-0.5171382367,-0.8957112847,-0.3656769413],
                [-0.5171382368,0.8957112847,-0.3656769413]]],
 'NHHH':   [0, [[-0.7080703847,0.5736644371,-0.2056610779],[0.314047869,0.6090902876,-0.2439925162],
                [-1.0241861213,0.328070168,-1.147576524],[-1.024191363,1.5321151073,-0.0355598819]]],
 'OHH':    [0, [[-0.7435290312,-0.0862560218,-0.2491318075],[0.2269625234,-0.0687025898,-0.2099668601],
                [-1.0265534922,0.2938386117,0.5988786675]]],
 'FH':     [0, [[0,0,-0.0161104113],[0,0,0.9161104113]]],
 'LiLi':   [0, [[0,0,-0.0155360351],[0,0,2.7155360351]]],
 'LiF':    [0, [[0,0,0.0],[0,0,1.564]]],
 'BeBe':   [0, [[0,0,0.0085515554],[0,0,2.4414484446]]],
 'CHCH':   [0, [[-7.5637480678,-4.08536579,0.0],[-8.6353642657,-4.08536579,0.0],
                [-6.3570037122,-4.08536579,0.0],[-5.2853875143,-4.08536579,0.0]]],
 'CCHHHH': [0, [[-4.5194036917,0.9995360751,-2.41325e-05],[-3.1861963083,0.9995360751,-2.41325e-05],
                [-5.0929778983,0.1325558381,-0.3377273553],[-5.0929780326,1.866487909,0.3377519084],
                [-2.6126221017,0.1325558381,-0.3377273553],[-2.6126219674,1.866487909,0.3377519084]]],
 'HCN':    [0, [[-2.1652707291,0.99953,0.0],[-3.242302537,0.99953,0.0],[-4.4007967339,0.99953,0.0]]],
 'CO':     [0, [[0,0,-0.0185570711],[0,0,1.1185570711]]],
 'NN':     [0, [[0,0,-0.0017036831],[0,0,1.1017036831]]],
 'NO':     [1, [[0,0,-0.0797720915],[0,0,1.0797720915]]],
 'OO':     [2, [[0,0,-0.0114390797],[0,0,1.2114390797]]],
 'FF':     [0, [[0,0,-0.0083068123],[0,0,1.4083068123]]],
 'PP':     [0, [[0,0,-0.0063578484],[0,0,1.9063578484]]],
 'ClCl':   [0, [[0,0,-0.0645570711],[0,0,1.9645570711]]],
}

# Experimental atomization energies (kcal/mol) of Table 2 of Roy 2023, and its two groups
# (single-bonded / multiple-bonded)
EXPT_AE = {'HH':110,'LiH':58,'CHHHH':419,'NHHH':297,'OHH':232,'FH':141,'LiLi':24,'LiF':139,'BeBe':3,
           'CHCH':405,'CCHHHH':563,'HCN':312,'CO':259,'NN':229,'NO':153,'OO':121,'FF':39,'PP':117,'ClCl':58}
GROUP1 = ['HH','LiH','CHHHH','NHHH','OHH','FH','LiLi','LiF','BeBe']
GROUP2 = ['CHCH','CCHHHH','HCN','CO','NN','NO','OO','FF','PP','ClCl']
PRETTY = {'HH':'H2','CHHHH':'CH4','NHHH':'NH3','OHH':'H2O','LiLi':'Li2','BeBe':'Be2','CHCH':'C2H2',
          'CCHHHH':'C2H4','NN':'N2','OO':'O2','FF':'F2','PP':'P2','ClCl':'Cl2'}

# Exact non-relativistic total atomic energies (Hartree): Pekeris 1962 and Schwartz 2006 for
# H and He; Chakravorty, Gwaltney, Davidson, Parpia and Fischer, Phys. Rev. A 47, 3649 (1993)
# for Li-Ar.
EXACT_ATOM = {"H":-0.500000,"He":-2.903724,"Li":-7.478060,"Be":-14.667356,"B":-24.653866,
              "C":-37.845000,"N":-54.589200,"O":-75.067300,"F":-99.733900,"Ne":-128.937600,
              "Na":-162.254900,"Mg":-200.053000,"Al":-242.346000,"Si":-289.359000,
              "P":-341.259000,"S":-398.110000,"Cl":-460.148000,"Ar":-527.540000}


def _ensure_header():
    if not os.path.exists(OUT):
        with open(OUT, "a") as f:
            f.write(f"# ENERGIES | basis={BASIS} | LAMBD={LAMBD} ({_variant}) | "
                    f"grid={GRID or 'default(3)'} | AE=(sum E_atoms - E_mol)*{HARTREE2KCAL}\n")
            f.write("system,kind,spin,E_tot_Ha,converged,how,secs\n")


def _done():
    d = {}
    if os.path.exists(OUT):
        for ln in open(OUT):
            if ln.startswith(("#", "system")) or "," not in ln:
                continue
            p = ln.strip().split(",")
            d[p[0]] = (float(p[3]), p[4] == "True")
    return d


# ------------------------------------------------------------------ report
def report():
    E = _done()
    lines = []
    A = lines.append
    A(f"# Energy benchmark -- basis={BASIS}  LAMBD={LAMBD} ({_variant})  grid={GRID or 'default(3)'}")
    A("")
    A("## Atomic total energies (Hartree)")
    A("| atom | E_tot (this work) | exact non-rel. ref | error |")
    A("|------|-------------------|--------------------|-------|")
    errs = []
    for a in ATOMS:
        if a in E and E[a][1]:
            e = E[a][0]; ref = EXACT_ATOM[a]; d = e - ref
            errs.append(abs(d))
            A(f"| {a} | {e:.6f} | {ref:.6f} | {d:+.6f} |")
        else:
            A(f"| {a} | (not converged) | {EXACT_ATOM[a]:.6f} | - |")
    mae_at = sum(errs) / len(errs) if errs else float("nan")
    A("")
    A(f"**Atomic-energy MAE over {len(errs)}/18 converged atoms = {mae_at:.4f} Ha**")
    A("")
    A("## Atomization energies (kcal/mol)")
    A("| molecule | AE (this work) | Expt | error |")
    A("|----------|----------------|------|-------|")
    ae, aerr = {}, {}
    for m in MOLECULES:
        if m not in E or not E[m][1]:
            A(f"| {PRETTY.get(m,m)} | (not converged) | {EXPT_AE[m]} | - |")
            continue
        counts = Counter()
        for el, c in re.findall(r'([A-Z][a-z]*)(\d*)', m):
            if el:
                counts[el] += int(c) if c else 1
        miss = [a for a in counts if a not in E or not E[a][1]]
        if miss:
            A(f"| {PRETTY.get(m,m)} | (missing atom {','.join(miss)}) | {EXPT_AE[m]} | - |")
            continue
        e_at = sum(E[a][0] * n for a, n in counts.items())
        v = (e_at - E[m][0]) * HARTREE2KCAL
        ae[m] = v; aerr[m] = v - EXPT_AE[m]
        A(f"| {PRETTY.get(m,m)} | {v:.1f} | {EXPT_AE[m]} | {aerr[m]:+.1f} |")
    A("")

    def mae(keys):
        v = [abs(aerr[k]) for k in keys if k in aerr]
        return (sum(v) / len(v) if v else float("nan")), len(v)

    m1, n1 = mae(GROUP1); m2, n2 = mae(GROUP2); mt, nt = mae(list(EXPT_AE))
    A(f"- **MAE single-bonded group ({n1}/9)  = {m1:.1f} kcal/mol**")
    A(f"- **MAE multiple-bonded group ({n2}/10) = {m2:.1f} kcal/mol**")
    A(f"- **TMAE all ({nt}/19)                = {mt:.1f} kcal/mol**")
    txt = "\n".join(lines)
    path = f"{ROOT}/energy_report_{_basistag}_lambd{LAMBD}{_gtag}.md"
    open(path, "w").write(txt + "\n")
    print(txt)
    print(f"\n[report written] {path}", flush=True)
    return path


mode = sys.argv[1] if len(sys.argv) > 1 else "--all"
if mode == "--report":
    report(); sys.exit(0)

# ------------------------------------------------------------------ compute
_ensure_header()
import numpy as onp
from jax import config
config.update("jax_enable_x64", True)
import CFXm_jax
CFXm_jax.LAMBD = LAMBD          # must be set before the first traced call
from pyscf import gto
from pyscf.dft import UKS, RKS
from get_veff import U_get_veff, R_get_veff


def make_mf(mol, spin):
    if spin > 0:
        m = UKS(mol); m.get_veff = types.MethodType(U_get_veff, m)
    else:
        m = RKS(mol); m.get_veff = types.MethodType(R_get_veff, m)
    m.max_cycle = 100; m.conv_tol = 1e-9; m.diis_space = 8
    if GRID:
        m.grids.level = int(GRID)
    return m


def run_system(name, spin, atomspec):
    """SCF strategy: open shells try quasi-second-order SCF (mf.newton()) first, since DIIS
    tends to oscillate for near-degenerate open shells; then a clean DIIS run, then a level
    shift with damping followed by an unshifted polish, then SOSCF as the last resort."""
    mol = gto.M(atom=atomspec, spin=spin, basis=BASIS, verbose=0)
    if spin > 0:
        mf = make_mf(mol, spin).newton()
        mf.max_cycle = 100
        mf.kernel()
        if mf.converged:
            return mf.e_tot, True, "newton"
        mf = make_mf(mol, spin)                     # fall back to the DIIS ladder
    else:
        mf = make_mf(mol, spin)
    mf.level_shift = 0.0; mf.damp = 0.0
    mf.kernel()
    if mf.converged:
        return mf.e_tot, True, "clean"
    dm = mf.make_rdm1(); mf.level_shift = 0.2; mf.damp = 0.4; mf.kernel(dm)
    dm = mf.make_rdm1(); mf.level_shift = 0.0; mf.damp = 0.0; mf.kernel(dm)
    if mf.converged:
        return mf.e_tot, True, "stab+polish"
    dm = mf.make_rdm1()
    m2 = make_mf(mol, spin).newton(); m2.max_cycle = 100
    m2.kernel(dm0=dm)                                # the newton() object takes the guess as dm0=
    return m2.e_tot, bool(m2.converged), "newton(fallback)"


ALL = [(a, "atom", ATOMS[a], a) for a in ATOMS]
for m, (sp, pos) in MOLECULES.items():
    atom_list = []
    for el, c in re.findall(r'([A-Z][a-z]*)(\d*)', m):
        if el:
            atom_list.extend([el] * (int(c) if c else 1))
    ALL.append((m, "molecule", sp, [[el, tuple(pos[i])] for i, el in enumerate(atom_list)]))

targets = ALL if mode == "--all" else [t for t in ALL if t[0] == sys.argv[2]]
if not targets:
    sys.exit(f"unknown system {sys.argv[2] if len(sys.argv)>2 else ''}")

done = _done()
print(f"[bench_energies] basis={BASIS} LAMBD={LAMBD} ({_variant}) grid={GRID or 'default'} "
      f"-> {OUT}\n  {len(done)} already done, {len(targets)} targets", flush=True)

for name, kind, spin, spec in targets:
    if name in done:
        print(f"[{name}] already done ({done[name][0]:.6f}); skip", flush=True)
        continue
    t0 = time.time()
    try:
        e, conv, how = run_system(name, spin, spec)
    except Exception as ex:
        e, conv, how = float("nan"), False, f"EXC:{type(ex).__name__}"
        print(f"[{name}] EXCEPTION: {type(ex).__name__}: {ex}", flush=True)
    secs = time.time() - t0
    with open(OUT, "a") as f:
        f.write(f"{name},{kind},{spin},{e:.10f},{conv},{how},{secs:.1f}\n")
    print(f"[{name}] E={e:.8f} conv={conv} how={how} {secs:.1f}s", flush=True)

print("[bench_energies] ALL DONE", flush=True)
