#!/bin/bash
# Per-trait LD clumping with in-sample LD (r2 < 0.1, +/-500 kb, P1 = 5e-8, P2 = 1e-5).
# usage: clump.sh fcm|wovf
set -euo pipefail
P=${1:-fcm}
N=15; [ "$P" = wovf ] && N=20
C=$OUT/clump/$P; mkdir -p $C/per_trait $C/ldref

if [ ! -s $C/ldref/sub.bed ]; then
  $PLINK2 --bfile $UKB_IMPUTED --keep $UKB_KEEP --extract $C/union_p1e5.ids \
    --make-pgen --sort-vars --threads $THREADS --out $C/ldref/sorted
  $PLINK2 --pfile $C/ldref/sorted --make-bed --threads $THREADS --out $C/ldref/sub
fi

for i in $(seq 1 $N); do
  t=${P}_z$i
  $PLINK2 --bfile $C/ldref/sub --clump $C/ss_f/$t.tsv \
    --clump-log10 input-only --clump-id-field ID --clump-p-field LOG10P \
    --clump-log10-p1 7.30103 --clump-log10-p2 5 --clump-r2 0.1 --clump-kb 500 --clump-unphased \
    --threads $THREADS --out $C/per_trait/$t || true
done
for i in $(seq 1 $N); do grep -vc '^#' $C/per_trait/${P}_z$i.clumps 2>/dev/null || echo 0; done \
  | awk '{s+=$1} END{print "per-trait index variants:", s}'
