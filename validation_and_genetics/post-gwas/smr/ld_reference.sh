#!/bin/bash
# 1000 Genomes GRCh38 European founders (no parents in the pedigree), restricted to the .ma variants.
set -euo pipefail
W=$OUT/smr/ld; mkdir -p $W
awk -F'\t' 'NR==1{for(i=1;i<=NF;i++) h[$i]=i; next} $h["SuperPop"]=="EUR" && $h["PAT"]=="0" && $h["MAT"]=="0"{print $1}' \
  $KG38_PSAM > $W/eur_founders.iid
awk 'NR>1{print $1}' $OUT/smr/ma/fcm_z1.ma | sort -u > $W/ma_snps.txt
$PLINK2 --pfile $KG38 vzs --keep $W/eur_founders.iid --extract $W/ma_snps.txt \
  --snps-only just-acgt --max-alleles 2 --rm-dup exclude-all --make-bed --threads $THREADS --out $W/ld_eur
