# CFX: self-consistent correlation-factor functional in JAX + PySCF

Code for the paper

> J.-B. Deville-Stoetzel, P.-O. Roy and M. Ernzerhof, *Automatic Differentiation for
> Self-Consistent Calculations with Complex Exchange–Correlation Functionals: The Correlation
> Factor Model as a Test Case* (submitted to J. Chem. Theory Comput.).

The exchange–correlation energy density of the correlation factor (CF) model is written in JAX as
one function of 14 ingredients per grid point. Its derivatives, obtained with `jax.value_and_grad`,
give the XC potential, which is passed to PySCF as a custom `get_veff`. Two variants:

| variant | `CFLAMBD` | |
|---|---|---|
| **CFX** | `0.0` (default) | B03 exchange hole in the correlation factor |
| **CFXmix** | `0.2` | 20 % of the normalized Becke–Roussel hole curvature mixed into Q |

## Requirements

Python 3.10, PySCF 2.14, JAX 0.6.2 (CPU, double precision), NumPy, attrs, matplotlib. Linux or
WSL (the BH76 driver uses `SIGALRM`). The run scripts call `python`; set `PY=/path/to/python` to
use another interpreter.

## Files

**Engine**

| file | content |
|---|---|
| `CFXm_jax.py` | the functional: CF coefficients, exact moments of the Becke–Roussel hole, the exact inversions, `U_energy_and_grad` |
| `B03_jax.py` | the B03 non-normalized hole: exact inversion `_solve_x_b03` (used) and the Arbuznikov–Kaupp fit `B03_fit` |
| `get_veff.py` | `U_get_veff` / `R_get_veff`: ingredients on the grid, J + Vxc for PySCF |
| `fock.py` | XC matrix from the derivatives with respect to each ingredient |
| `hfx_density.py` | exact-exchange energy density on the grid, from DeepMind's DM21 code (Apache 2.0) |

`bash verify_engine.sh` checks these five files against `ENGINE_MD5.txt`.

**Benchmarks** (each driver documents its options in its header)

| script | driver | paper |
|---|---|---|
| `run_energies.sh` | `bench_energies.py` | atomic energies (18 atoms) and atomization energies (19 molecules), cc-pVTZ |
| `run_density.sh` | `bench_density.py` | density errors of 28 diatomics, aug-cc-pVTZ, CCSD reference |
| `run_dipoles.sh` | `bench_dipoles.py` | dipole moments of the same 28 diatomics |
| `run_bh76.sh` | `bench_barriers.py`, `bh76_lib.py` | BH76 barrier heights, cc-pVTZ |
| `run_fig1_co.sh` | `fig1_CO.py` | Figure 1, the CO density along the bond axis |

**Data** (`data/`): BH76 geometries, references and charges/multiplicities from GMTKN55; the
CCSD and PBE/SCAN dipole references of the 28 diatomics; the PBE/SCAN/CCSD densities of CO along
the bond axis. The BH76 geometries are kept in a tarball because two species names differ only
by case (`C2H5`, `c2h5`); `bh76_lib.py` reads them from it directly.

## Use

```bash
bash run_energies.sh                 # CFX
CFLAMBD=0.2 bash run_energies.sh     # CFXmix
```

Every driver is resumable: rows already in its CSV are skipped, so a run that stops can simply be
started again. Results are written next to the scripts.

The functional on its own, in a PySCF calculation:

```python
import types
from pyscf import gto, dft
import CFXm_jax
CFXm_jax.LAMBD = 0.0                  # set before the first call; one value per process
from get_veff import R_get_veff, U_get_veff

mol = gto.M(atom="Ne", basis="cc-pVTZ")
mf = dft.RKS(mol)
mf.get_veff = types.MethodType(R_get_veff, mf)
mf.conv_tol = 1e-9
mf.kernel()
```

Use `UKS` with `U_get_veff` for open shells.

## Notes

- `LAMBD` and `EXACT_B03` are read when JAX first traces the functional. Changing them after the
  first call has no effect, so each variant needs its own process.
- JAX and the CCSD reference calculations are run in separate processes to keep the peak memory
  low; the benchmarks ran on a laptop with about 3 GB of memory available to them.
- The BH76 set takes about 1.5 days per variant on such a machine.

## Licence

Apache License 2.0 (see `LICENSE`). Copyright 2026 Jean-Benoît Deville-Stoetzel, Pierre-Olivier
Roy and Matthias Ernzerhof. `hfx_density.py` comes from DeepMind's DM21 code (Apache 2.0) and
the BH76 data from GMTKN55; see `NOTICE`.
