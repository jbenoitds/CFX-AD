#!/bin/bash
# Atomic energies (18 atoms, H..Ar) and atomization energies (19 molecules), basis cc-pVTZ.
#
# bench_energies.py --all keeps one compiled JAX executable per grid shape, so its memory grows
# over the 37 systems. It is resumable (rows already in the CSV are skipped), so whenever it stops
# early this script starts it again in a fresh process; a restart repeats at most one system.
#
# Usage:  bash run_energies.sh            # run/resume the whole set, then print the report
#         bash run_energies.sh --report   # just rebuild the report from the CSV
#
# Override the defaults on the command line, e.g.
#         CFLAMBD=0.2 CFBASIS=aug-cc-pVTZ bash run_energies.sh
set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 1
. ./cell.env
PY=${PY:-python}
export CFBASIS=${CFBASIS:-cc-pVTZ}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

_basistag=$(echo "$CFBASIS" | tr 'A-Z' 'a-z' | tr -d '.-')
CSV="energies_${_basistag}_lambd${CFLAMBD}.csv"
LOG="logs/$(basename "${CSV%.csv}").log"
mkdir -p logs

if [ "${1:-}" = "--report" ]; then
  $PY -u bench_energies.py --report
  exit $?
fi

NTOT=37            # 18 atoms + 19 molecules
MAXPASS=12
done_count() { [ -f "$CSV" ] && grep -cE '^[A-Za-z]+,(atom|molecule),' "$CSV" || echo 0; }

echo "[run_energies] $CELL_NAME | basis=$CFBASIS -> $CSV  (log $LOG)" | tee -a "$LOG"
for pass in $(seq 1 $MAXPASS); do
  N=$(done_count)
  if [ "$N" -ge "$NTOT" ]; then
    echo "[run_energies] $N/$NTOT systems present -- complete after $((pass-1)) pass(es)" | tee -a "$LOG"
    $PY -u bench_energies.py --report | tee -a "$LOG"
    exit 0
  fi
  echo "[run_energies] pass $pass: $N/$NTOT done, launching a fresh process" | tee -a "$LOG"
  $PY -u bench_energies.py --all 2>&1 | grep --line-buffered -Ev "WARNING|xla_bridge" | tee -a "$LOG"
  AFTER=$(done_count)
  if [ "$AFTER" -le "$N" ]; then
    echo "[run_energies] pass $pass made no progress ($N -> $AFTER); stopping." | tee -a "$LOG"
    exit 1
  fi
done
echo "[run_energies] hit MAXPASS=$MAXPASS with $(done_count)/$NTOT done" | tee -a "$LOG"
exit 1
