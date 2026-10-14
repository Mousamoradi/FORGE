#!/bin/bash
# UK Biobank sample QC and genetic ancestry.
set -euo pipefail
W=$OUT/qc; mkdir -p $W

# withdrawn, absent from the imputed release, or without sample QC data
awk 'NR==FNR{w[$1]=1; next} FNR>1 && !($1 in w){print $1, $1}' $UKB_WITHDRAWN $UKB_SQC > $W/keep0.txt
awk 'NR==FNR{k[$1]=1; next} ($2 in k){print $1, $2}' $W/keep0.txt $UKB_IMPUTED.fam > $W/keep1.txt

# ambiguous sex: submitted and inferred sex disagree, or sex-chromosome aneuploidy
python - <<'EOF'
import os, pandas as pd
s = pd.read_csv(os.environ["UKB_SQC"], sep="\t")
bad = (s["Submitted.Gender"] != s["Inferred.Gender"]) | (s["putative.sex.chromosome.aneuploidy"] == 1)
s.loc[bad, ["eid"]].assign(iid=s.eid).to_csv(os.environ["OUT"] + "/qc/sex_fail.txt", sep=" ", header=False, index=False)
EOF

$PLINK2 --bfile $UKB_ARRAY --keep $W/keep1.txt --remove $W/sex_fail.txt --autosome --maf 0.05 \
  --indep-pairwise 200kb 1 0.2 --threads $THREADS --out $W/prune
$PLINK2 --bfile $UKB_ARRAY --keep $W/keep1.txt --remove $W/sex_fail.txt --extract $W/prune.prune.in \
  --make-bed --threads $THREADS --out $W/pruned

python $FORGE/gwas/ancestry.py $W/pruned $W/ancestry.tsv

$PLINK --bfile $W/pruned --het --out $W/het
python - <<'EOF'
import os, pandas as pd
W = os.environ["OUT"] + "/qc"
h = pd.read_csv(f"{W}/het.het", sep=r"\s+")
a = pd.read_csv(f"{W}/ancestry.tsv", sep="\t")
h = h.merge(a, on="IID")
z = h.groupby("ancestry").F.transform(lambda f: (f - f.mean()) / f.std())
h.loc[z.abs() > 5, ["FID", "IID"]].to_csv(f"{W}/het_fail.txt", sep=" ", header=False, index=False)
EOF

$PLINK --bfile $W/pruned --remove $W/het_fail.txt --genome --min 0.1875 --threads $THREADS --out $W/rel
python - <<'EOF'
import os, pandas as pd
W = os.environ["OUT"] + "/qc"
g = pd.read_csv(f"{W}/rel.genome", sep=r"\s+")
edges = list(zip(g.IID1.astype(str), g.IID2.astype(str)))
drop = set()
while True:
    e = [(a, b) for a, b in edges if a not in drop and b not in drop]
    if not e:
        break
    deg = pd.Series([x for ab in e for x in ab]).value_counts()
    drop.add(deg.index[0])
pd.DataFrame({"f": sorted(drop), "i": sorted(drop)}).to_csv(f"{W}/rel_fail.txt", sep=" ", header=False, index=False)
EOF

cat $W/het_fail.txt $W/rel_fail.txt $W/sex_fail.txt > $W/remove.txt
awk 'NR==FNR{r[$2]=1; next} !($2 in r)' $W/remove.txt $W/keep1.txt > $W/keep_qc.txt
awk -F'\t' 'NR==FNR{if($2=="EUR") e[$1]=1; next} ($2 in e)' $W/ancestry.tsv $W/keep_qc.txt > $W/keep_eur.txt
wc -l $W/keep_qc.txt $W/keep_eur.txt
