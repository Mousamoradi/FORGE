"""Genetic (LDSC) versus phenotypic correlation across the 105 endophenotype pairs.
Phenotypic r is computed in the GWAS set (UKB_PHENO restricted to UKB_KEEP)."""
import glob, itertools, os
import numpy as np, pandas as pd
from common import E, FCM, OUT

rows = []
for f in glob.glob(f"{OUT}/ldsc/rg/*.log"):
    lines = open(f).read().split("Summary of Genetic Correlation Results")[1].strip().splitlines()
    rows.append(pd.DataFrame([l.split() for l in lines[1:] if l.strip() and not l.startswith("Analysis")],
                             columns=lines[0].split()))
rg = pd.concat(rows)
name = lambda p: os.path.basename(p).replace(".sumstats.gz", "")
rg["a"], rg["b"] = rg.p1.map(name), rg.p2.map(name)
for c in ("rg", "se", "z", "p"):
    rg[c] = pd.to_numeric(rg[c], errors="coerce")
assert len(rg) == 105

ph = pd.read_csv(E["UKB_PHENO"], sep="\t")
keep = pd.read_csv(E["UKB_KEEP"], sep=r"\s+", header=None, names=["FID", "IID"])
rp = ph.merge(keep, on=["FID", "IID"])[FCM].corr()
rg["rp"] = [rp.loc[a, b] for a, b in zip(rg.a, rg.b)]
bonf = 0.05 / len(rg)
rg["resolved"] = rg.p < bonf
rg["same_sign"] = np.sign(rg.rg) == np.sign(rg.rp)
uncentred = (rg.rg * rg.rp).sum() / np.sqrt((rg.rg ** 2).sum() * (rg.rp ** 2).sum())
rg.to_csv(f"{OUT}/ldsc/rg_vs_rp.tsv", sep="\t", index=False)
print(f"median |rg| {rg.rg.abs().median():.2f}  median |rp| {rg.rp.abs().median():.2f}")
print(f"uncentred correlation {uncentred:.3f}  same sign {rg.same_sign.sum()}/105")
print(f"discordant pairs passing Bonferroni {(~rg.same_sign & rg.resolved).sum()}/{(~rg.same_sign).sum()}; "
      f"resolved pairs agreeing in sign {(rg.same_sign & rg.resolved).sum()}/{rg.resolved.sum()}")
