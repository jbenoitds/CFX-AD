#!/bin/bash
# Check the five engine files (CFXm_jax.py, B03_jax.py, get_veff.py, fock.py, hfx_density.py)
# against the checksums in ENGINE_MD5.txt.
#
# Usage: bash verify_engine.sh
cd "$(dirname "$(readlink -f "$0")")" || exit 1
if md5sum -c ENGINE_MD5.txt; then
  echo "[verify_engine] PASS -- engine files identical to the published version"
  exit 0
else
  echo "[verify_engine] FAIL -- an engine file differs from the published version"
  exit 1
fi
