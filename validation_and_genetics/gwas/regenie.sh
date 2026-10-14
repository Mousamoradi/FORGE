#!/bin/bash
# REGENIE v4.1.2. usage: regenie.sh fcm|wovf
set -euo pipefail
P=${1:-fcm}
PHENO=$UKB_PHENO; [ "$P" = wovf ] && PHENO=$UKB_PHENO_NOVF
W=$OUT/regenie/$P; mkdir -p $W

$PLINK2 --bfile $UKB_ARRAY --keep $UKB_KEEP --autosome --maf 0.01 --mac 100 --geno 0.1 --hwe 1e-15 \
  --write-snplist --out $OUT/regenie/array_qc

COV="--covarFile $UKB_COVAR --covarColList PC{1:10},age,SE --catCovarList sex,array"

$REGENIE --step 1 --bed $UKB_ARRAY --extract $OUT/regenie/array_qc.snplist --keep $UKB_KEEP \
  --phenoFile $PHENO $COV --qt --apply-rint --bsize 1000 --lowmem --lowmem-prefix $W/tmp \
  --threads $THREADS --out $W/step1

$REGENIE --step 2 --bed $UKB_IMPUTED --keep $UKB_KEEP --phenoFile $PHENO $COV \
  --qt --apply-rint --pred $W/step1_pred.list --bsize 400 --minMAC 20 --gz \
  --threads $THREADS --out $W/assoc
