#!/bin/bash
# Density accuracy on the 28 diatomics of Table 3 of Roy 2023, basis aug-cc-pVTZ.
#
#   error_i = (1/Ne_i) INT |rho_DFT - rho_CCSD| d3r      (their Eq. 15)
#   set MAE = mean_i error_i;  rho_CCSD is the CCSD relaxed 1-RDM (solve_lambda, frozen core).
#
# Two processes per molecule: the --dft stage writes stage_dft_<tag>_<MOL>.npz and exits, then
# --cc loads it and runs the CCSD without importing JAX. The CCSD reference is recomputed on the
# DFT grid of each run. Resumable: molecules already in the CSV are skipped.
#
# Usage: bash run_density.sh [MOL ...]     # default: all 28, then the set MAE
set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 1
. ./cell.env
PY=${PY:-python}
export CFBASIS=${CFBASIS:-aug-cc-pVTZ}
export CFFROZEN=${CFFROZEN:-core}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
mkdir -p logs

MOLS=${*:-"LiH BH CN OH LiO CO NO HF LiF BF CF FO NaF AlF SiO PN PO HS SiS CS NS SO SF HCl NaCl LiCl ClO ClF"}

echo "[run_density] $CELL_NAME | basis=$CFBASIS frozen=$CFFROZEN"
for M in $MOLS; do
  echo "[run_density] --- $M (dft) --- $(date +%H:%M:%S)"
  $PY -u bench_density.py --dft "$M" 2>&1 | grep --line-buffered -Ev "WARNING|xla_bridge"
  echo "[run_density] --- $M (cc)  --- $(date +%H:%M:%S)"
  $PY -u bench_density.py --cc  "$M" 2>&1 | grep --line-buffered -Ev "WARNING|xla_bridge"
done
$PY -u bench_density.py --mae 2>&1 | grep --line-buffered -Ev "WARNING|xla_bridge"
echo "[run_density] DONE $(date +%H:%M:%S)"
