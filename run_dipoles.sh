#!/bin/bash
# Dipole moments of the same 28 diatomics, Table 3 of Roy 2023, basis aug-cc-pVTZ.
# Reference = CCSD relaxed 1-RDM (frozen core); PBE and SCAN from plain PySCF as a check.
#
# Four streams; only --cf depends on the functional:
#   --cc    CCSD reference    -> data/dipoles_cc_augccpvtz.csv
#   --gate  PBE + SCAN        -> data/dipoles_gate_augccpvtz.csv
#   --cf    CF functional     -> dipoles_cf_<basis>_lambd<L>.csv
#   --merge join + report     -> dipole_report_<...>.md
#
# The two reference files are shipped in data/ (28/28), so the default run does only --cf and
# --merge. Pass `refs` to regenerate them (several hours; the UCCSD of CN alone needs ~3 GB).
#
# The --cf stream runs all 28 molecules in one process, so JAX compiles once; it is resumable.
#
# Usage: bash run_dipoles.sh          # --cf (all 28) then --merge
#        bash run_dipoles.sh refs     # regenerate the CCSD + PBE/SCAN references first
set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 1
. ./cell.env
PY=${PY:-python}
export CFBASIS=${CFBASIS:-aug-cc-pVTZ}
export CFFROZEN=${CFFROZEN:-core}
export CFMAXMEM=${CFMAXMEM:-1800}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
mkdir -p logs

MOLS="LiH BH CN OH LiO CO NO HF LiF BF CF FO NaF AlF SiO PN PO HS SiS CS NS SO SF HCl NaCl LiCl ClO ClF"

echo "[run_dipoles] $CELL_NAME | basis=$CFBASIS frozen=$CFFROZEN"

if [ "${1:-}" = "refs" ]; then
  # one molecule per process, one stage at a time, so each CCSD gets the memory to itself
  for ST in cc gate; do
    for M in $MOLS; do
      echo "[run_dipoles] ===== $ST $M $(date +%H:%M:%S) ====="
      timeout 3000 $PY -u bench_dipoles.py --$ST "$M" || echo "[run_dipoles] $ST $M FAIL rc=$?"
    done
  done
  echo "[run_dipoles] reference streams complete $(date +%H:%M:%S)"
fi

echo "[run_dipoles] ===== CF stream (one process, resumable) $(date +%H:%M:%S) ====="
timeout 36000 $PY -u bench_dipoles.py --cf all 2>&1 | grep --line-buffered -Ev "WARNING|xla_bridge"
echo "[run_dipoles] ===== merge + report ====="
$PY -u bench_dipoles.py --merge
echo "[run_dipoles] DONE $(date +%H:%M:%S)"
