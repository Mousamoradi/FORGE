#!/bin/bash
# GCTA-GREML (v1.94.1) on the 629,669 array variants, relatedness cutoff 0.05.
set -euo pipefail
W=$OUT/greml; mkdir -p $W

$GCTA --bfile $UKB_ARRAY --extract $OUT/regenie/array_qc.snplist --keep $UKB_KEEP --autosome \
  --make-grm --thread-num $THREADS --out $W/grm
$GCTA --grm $W/grm --grm-cutoff 0.05 --make-grm --out $W/grm05

python - <<'EOF'
import os, pandas as pd
from scipy.stats import norm
W = os.environ["OUT"] + "/greml"
ids = pd.read_csv(f"{W}/grm05.grm.id", sep=r"\s+", header=None, names=["FID", "IID"])
p = pd.read_csv(os.environ["UKB_PHENO"], sep="\t").merge(ids, on=["FID", "IID"])
for c in [c for c in p.columns if c.startswith("fcm_z")]:
    r = p[c].rank(method="average")
    p[c] = norm.ppf((r - 0.5) / p[c].notna().sum())
p.to_csv(f"{W}/pheno_rint.txt", sep=" ", header=False, index=False, na_rep="NA")
c = pd.read_csv(os.environ["UKB_COVAR"], sep="\t").merge(ids, on=["FID", "IID"])
c[["FID", "IID"] + [f"PC{i}" for i in range(1, 11)] + ["age", "SE"]].to_csv(f"{W}/qcovar.txt", sep=" ", header=False, index=False)
c[["FID", "IID", "sex", "array"]].to_csv(f"{W}/covar.txt", sep=" ", header=False, index=False)
EOF

for i in $(seq 1 15); do
  $GCTA --reml --grm $W/grm05 --pheno $W/pheno_rint.txt --mpheno $i \
    --qcovar $W/qcovar.txt --covar $W/covar.txt --thread-num $THREADS --out $W/fcm_z$i
done
grep -H "^V(G)/Vp" $W/fcm_z*.hsq
