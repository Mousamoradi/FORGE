#!/bin/bash
# MGBB: KING unrelated set (kinship < 0.0884), 10 in-sample PCs, PLINK 2 --glm on the 15 RINT axes.
set -euo pipefail
W=$OUT/mgbb; mkdir -p $W/glm
$PLINK2 --bfile $MGBB_BFILE --keep $W/sample_eur.keep --make-pgen --threads $THREADS --out $W/mgbb_eur
$PLINK2 --pfile $W/mgbb_eur --king-cutoff 0.0884 --out $W/unrelated
$PLINK2 --pfile $W/mgbb_eur --maf 0.01 --geno 0.02 --hwe 1e-6 --autosome --indep-pairwise 200 50 0.2 --out $W/prune
$PLINK2 --pfile $W/mgbb_eur --extract $W/prune.prune.in --pca 10 --out $W/pca

python - <<'EOF'
import os, pandas as pd
W = os.environ["OUT"] + "/mgbb"
pc = pd.read_csv(f"{W}/pca.eigenvec", sep="\t").rename(columns={"#FID": "FID", "#IID": "IID"})
pc["IID"] = pc.IID.astype(str)
c = pd.read_csv(f"{W}/covar_base.tsv", sep="\t", dtype={"FID": str, "IID": str})
c.merge(pc[["IID"] + [f"PC{i}" for i in range(1, 11)]], on="IID").to_csv(f"{W}/covar_full.tsv", sep="\t", index=False)
EOF

$PLINK2 --pfile $W/mgbb_eur --keep $W/unrelated.king.cutoff.in.id --pheno $W/pheno_rint.tsv --covar $W/covar_full.tsv \
  --glm hide-covar omit-ref cols=+a1freq,+beta,+se,+tz,+p,+nobs --covar-variance-standardize \
  --autosome --maf 0.01 --geno 0.05 --hwe 1e-12 midp --threads $THREADS --out $W/glm/mgbb
