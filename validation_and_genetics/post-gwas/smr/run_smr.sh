#!/bin/bash
# SMR v1.03 with HEIDI (defaults): retina expression, GTEx expression and splicing, 15 traits each.
set -euo pipefail
W=$OUT/smr; mkdir -p $W/out_retina $W/out_expr $W/out_sqtl
run() { $SMR --bfile $W/ld/ld_eur --gwas-summary $W/ma/$3.ma --beqtl-summary $1 --out $2 --thread-num 2; }
for i in $(seq 1 15); do
  run $W/retina/retina_eyegex $W/out_retina/Retina.fcm_z$i fcm_z$i
  for T in $(cat $FORGE/gene/tissues.txt); do
    run $W/gtex/$T.expr $W/out_expr/$T.fcm_z$i fcm_z$i
    run $W/gtex/$T.sqtl $W/out_sqtl/$T.fcm_z$i fcm_z$i
  done
done
