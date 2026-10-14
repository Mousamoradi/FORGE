"""Project UK Biobank onto 1000 Genomes Phase 3 PCs and assign ancestry by k-nearest neighbours.
usage: ancestry.py <ukb_pruned_bfile> <out.tsv>"""
import os, subprocess, sys
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

ukb, out = sys.argv[1], sys.argv[2]
E = os.environ
W = os.path.dirname(out)
kg = f"{W}/kg_ref"

subprocess.run([E["PLINK2"], "--pfile", E["KG38"], "vzs", "--extract", f"{ukb}.bim",
                "--make-bed", "--out", kg], check=True)
subprocess.run([E["PLINK2"], "--bfile", kg, "--freq", "counts", "--pca", "5", "allele-wts",
                "--out", f"{W}/kg_pca"], check=True)
for src, tag in ((kg, "kg"), (ukb, "ukb")):
    subprocess.run([E["PLINK2"], "--bfile", src, "--read-freq", f"{W}/kg_pca.acount",
                    "--score", f"{W}/kg_pca.eigenvec.allele", "2", "5", "header-read",
                    "no-mean-imputation", "variance-standardize",
                    "--score-col-nums", "6-10", "--out", f"{W}/{tag}_proj"], check=True)

ref = pd.read_csv(f"{W}/kg_proj.sscore", sep="\t").rename(columns={"#IID": "IID"})
ukb_s = pd.read_csv(f"{W}/ukb_proj.sscore", sep="\t").rename(columns={"#FID": "FID"})
lab = pd.read_csv(E["KG38_PSAM"], sep="\t").rename(columns={"#IID": "IID"})
ref = ref.merge(lab[["IID", "SuperPop"]], on="IID")
pcs = [c for c in ref.columns if c.endswith("_AVG")]
knn = KNeighborsClassifier(n_neighbors=10).fit(ref[pcs], ref.SuperPop)
ukb_s["ancestry"] = knn.predict(ukb_s[pcs])
ukb_s[["IID", "ancestry"]].to_csv(out, sep="\t", index=False)
print(ukb_s.ancestry.value_counts().to_string())
