#!/bin/bash
# LDSC: munge against HapMap 3, observed-scale h2 per trait, genetic correlation for all pairs.
# usage: ldsc.sh fcm|wovf
set -euo pipefail
P=${1:-fcm}
N=15; SS=$SUMSTATS
[ "$P" = wovf ] && { N=20; SS=$SUMSTATS_NOVF; }
W=$OUT/ldsc; mkdir -p $W/munged $W/h2 $W/rg

for i in $(seq 1 $N); do
  t=${P}_z$i
  $LDSC_PY $LDSC_DIR/munge_sumstats.py --sumstats $SS/$t.tsv.gz \
    --snp rsid --a1 effect_allele --a2 other_allele --N-col n --p p_value --frq effect_allele_frequency \
    --signed-sumstats beta,0 --merge-alleles $HM3_SNPLIST --maf-min 0.01 --chunksize 500000 \
    --out $W/munged/$t
  $LDSC_PY $LDSC_DIR/ldsc.py --h2 $W/munged/$t.sumstats.gz \
    --ref-ld-chr $EUR_W_LD --w-ld-chr $EUR_W_LD --out $W/h2/$t
done

[ "$P" = fcm ] || exit 0
for i in $(seq 1 $((N - 1))); do
  rest=$(for j in $(seq $((i + 1)) $N); do printf "%s," $W/munged/${P}_z$j.sumstats.gz; done)
  $LDSC_PY $LDSC_DIR/ldsc.py --rg $W/munged/${P}_z$i.sumstats.gz,${rest%,} \
    --ref-ld-chr $EUR_W_LD --w-ld-chr $EUR_W_LD --out $W/rg/${P}_z$i
done
