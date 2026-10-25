#!/bin/bash
# Figure 1: the CO molecule along the bond axis (CCSD density; relative density error of CFX and
# of CFXmix, each with PBE and SCAN). See fig1_CO.py for the grid and the frame.
#
# data/fig1_CO_ref_augccpvtz.npz (PBE, SCAN and all-electron CCSD on the line) is shipped, so the
# default run computes only the two CF densities and draws the figure.
#
# Usage: bash run_fig1_co.sh          # CF densities at lambda 0.0 and 0.2, then the figure
#        bash run_fig1_co.sh ref      # regenerate the PySCF reference npz first
set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 1
PY=${PY:-python}
export CFBASIS=${CFBASIS:-aug-cc-pVTZ}
export CFMAXMEM=${CFMAXMEM:-2200}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

echo "[run_fig1_co] basis=$CFBASIS"

if [ "${1:-}" = "ref" ]; then
  echo "[run_fig1_co] ===== reference (PySCF only: PBE, SCAN, CCSD) ====="
  $PY -u fig1_CO.py --ref
fi

for L in 0.0 0.2; do      # one process per lambda
  echo "[run_fig1_co] ===== CF density, lambda=$L ====="
  CFLAMBD=$L $PY -u fig1_CO.py --dft 2>&1 | grep --line-buffered -Ev "WARNING|xla_bridge"
done
echo "[run_fig1_co] ===== figure + csv ====="
$PY -u fig1_CO.py --plot
echo "[run_fig1_co] DONE $(date +%H:%M:%S)"
