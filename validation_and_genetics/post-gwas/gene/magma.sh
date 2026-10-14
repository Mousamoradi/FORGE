#!/bin/bash
# MAGMA v1.10: 35 kb / 10 kb window, NCBI 37.3 genes, 1000 Genomes European panel, SNP-wise mean.
set -euo pipefail
W=$OUT/magma; mkdir -p $W/pval $W/out

python - <<'EOF'
import os
from common import FCM, read_sumstats
W = os.environ["OUT"] + "/magma"
snps = set()
for t in FCM:
    d = read_sumstats(t, usecols=["SNP", "P"])
    d.to_csv(f"{W}/pval/{t}.pval", sep="\t", index=False)
    snps |= set(d.SNP)
open(f"{W}/gwas_snps.txt", "w").write("\n".join(sorted(snps)) + "\n")
EOF

$PLINK2 --bfile $G1000_EUR --extract $W/gwas_snps.txt --make-bed --out $W/g1000_sub
$MAGMA --annotate window=35,10 --snp-loc $W/g1000_sub.bim --gene-loc $NCBI37_GENELOC --out $W/annot
for i in $(seq 1 15); do
  $MAGMA --bfile $W/g1000_sub --pval $W/pval/fcm_z$i.pval use=SNP,P N=39381 \
    --gene-annot $W/annot.genes.annot --genes-only --out $W/out/fcm_z$i
done
