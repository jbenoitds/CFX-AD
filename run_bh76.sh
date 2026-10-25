#!/bin/bash
# BH76 barrier heights (GMTKN55), 76 barriers over 79 species, basis cc-pVTZ.
# The most expensive set here: about 1.5 days per variant on a 4-core laptop.
#
# Species run smallest first (atoms, then basis functions). Resumable throughout.
#
# JAX keeps one compiled executable per grid shape and every species has its own grid, so a long
# process keeps growing. Each process therefore stops after CFMAXSPEC species and this script
# starts a fresh one. A species that was running when a process was killed (out of memory) is
# simply retried; it is recorded as failed only if it dies a second time in a row at the start of
# a fresh process.
#
# A species that exceeds CFSPECSEC is recorded as failed (TIMEOUT); to retry it, delete its row
# from the CSV and run again. The largest species took 2-5 h at cc-pVTZ on the laptop used for the
# paper, so the default limit is 6 h.
#
# Usage: bash run_bh76.sh              # run/resume the set
#        bash run_bh76.sh --report     # build the barrier report from the CSV
#        bash run_bh76.sh --gate       # plain-PySCF PBE instead of the CF functional
set -u
cd "$(dirname "$(readlink -f "$0")")" || exit 1
. ./cell.env
PY=${PY:-python}
export CFBASIS=${CFBASIS:-cc-pVTZ}
export CFMETHOD=${CFMETHOD:-cf}
export CFSPECSEC=${CFSPECSEC:-21600}       # wall-clock limit per species (s)
export CFMAXSPEC=${CFMAXSPEC:-12}          # species per process before a clean restart
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
mkdir -p logs

case "${1:-}" in
  --report) $PY -u bench_barriers.py --report; exit $? ;;
  --gate)   export CFMETHOD=pbe; CELL_NAME="plain-PySCF PBE" ;;
esac

_basistag=$(echo "$CFBASIS" | tr 'A-Z' 'a-z' | tr -d '.-')
if [ "$CFMETHOD" = "cf" ]; then _mtag="cf_lambd${CFLAMBD}"; else _mtag="$CFMETHOD"; fi
CSV="barriers_${_basistag}_${_mtag}.csv"
INFLIGHT="$CSV.inflight"
PREV_VICTIM=""

echo "===== BH76 start $(date '+%F %T') ====="
echo "      $CELL_NAME | basis=$CFBASIS | limit=${CFSPECSEC}s/species | restart every $CFMAXSPEC"
echo "      -> $CSV"

for PASS in $(seq 1 400); do
  BEFORE=$(grep -vc '^#\|^species' "$CSV" 2>/dev/null || echo 0)
  rm -f "$INFLIGHT"
  $PY -u bench_barriers.py --all
  RC=$?
  AFTER=$(grep -vc '^#\|^species' "$CSV" 2>/dev/null || echo 0)

  if [ "$RC" = "0" ]; then
    if [ "$AFTER" = "$BEFORE" ]; then
      echo "===== all species accounted for: $AFTER/79 $(date '+%T') ====="
      break
    fi
    echo "----- pass $PASS ok, $BEFORE->$AFTER species $(date '+%T') -----"
    PREV_VICTIM=""
    continue
  fi

  # abnormal exit -- identify the species that was running
  SP=""
  [ -s "$INFLIGHT" ] && SP=$(head -1 "$INFLIGHT")
  rm -f "$INFLIGHT"
  if [ -z "$SP" ]; then
    echo "----- pass $PASS: rc=$RC, no in-flight marker, $BEFORE->$AFTER $(date '+%T') -----"
    [ "$AFTER" = "$BEFORE" ] && { echo "no progress and no in-flight species -- stopping"; break; }
    continue
  fi

  if [ "$RC" = "137" ] || [ "$RC" = "9" ]; then REASON="OOM(SIGKILL)"; else REASON="DIED(rc=$RC)"; fi

  if [ "$SP" = "$PREV_VICTIM" ] && [ "$BEFORE" = "$AFTER" ]; then
    # died again, and this pass completed nothing before dying -> it died in a fresh process
    NAT=$($PY -c "import sys;sys.path.insert(0,'.');import bh76_lib as B;print(len(B.parse_coord('$SP')))" 2>/dev/null || echo -1)
    Q=$($PY -c "import sys;sys.path.insert(0,'.');import bh76_lib as B;s=B.load_species()['$SP'];print(f'{s[0]},{s[1]}')" 2>/dev/null || echo "0,0")
    echo "$SP,$Q,$NAT,-1,nan,False,$REASON,0.0" >> "$CSV"
    echo "===== $SP failed twice (second time in a fresh process) -> recorded as $REASON $(date '+%T') ====="
    PREV_VICTIM=""
  else
    echo "----- pass $PASS: $SP died ($REASON) after $BEFORE->$AFTER; retrying in a fresh process $(date '+%T') -----"
    PREV_VICTIM="$SP"
  fi
  sleep 10
done

echo "===== BH76 ended $(date '+%F %T'), $(grep -vc '^#\|^species' "$CSV" 2>/dev/null) rows ====="
$PY -u bench_barriers.py --report
