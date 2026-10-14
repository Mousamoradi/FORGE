"""Per-region LD from the UK Biobank windows. Variants are joined on rsID and unordered allele pair;
r is re-signed to the GWAS effect allele."""
import os
import numpy as np, pandas as pd
from common import E, OUT, out

W = pd.read_csv(f"{OUT}/ld/windows.tsv", sep="\t")
RAW = E["ALKES_LD"]
for win, grp in W.groupby("win", sort=False):
    sc = pd.read_csv(f"{RAW}/{win}.gz", sep="\t").reset_index().rename(columns={"index": "widx"})
    sc["rsid"] = sc.rsid.astype(str)
    z = np.load(f"{RAW}/{win}.npz", allow_pickle=True)
    row, col, data = z["row"], z["col"], z["data"]
    for reg in grp.region:
        d = pd.read_csv(f"{OUT}/finemap/regions/{reg}.tsv", sep="\t").drop_duplicates("SNP")
        m = d.merge(sc, left_on="SNP", right_on="rsid")
        same = [{a.upper(), b.upper()} == {c.upper(), e.upper()}
                for a, b, c, e in zip(m.A1, m.A2, m.allele1, m.allele2)]
        m = m[same].drop_duplicates("SNP").sort_values("widx").reset_index(drop=True)
        if len(m) < 200:
            print(reg, "too few matched variants:", len(m))
            continue
        loc = np.full(int(sc.widx.max()) + 1, -1)
        loc[m.widx.values] = np.arange(len(m))
        keep = (loc[row] >= 0) & (loc[col] >= 0)
        R = np.zeros((len(m), len(m)))
        R[loc[row[keep]], loc[col[keep]]] = data[keep]
        R = np.where(R != 0, R, R.T)
        D = np.where(m.A1.str.upper().values != m.allele1.str.upper().values, -1.0, 1.0)
        R = R * D[:, None] * D[None, :]
        np.fill_diagonal(R, 1.0)
        np.savez_compressed(out("ld", "ukb", f"{reg}.npz"), R=R.astype(np.float32),
                            snp=m.SNP.values.astype(str), pos=m.POS.values, a1=m.A1.values.astype(str),
                            a2=m.A2.values.astype(str), beta=m.BETA.values, se=m.SE.values,
                            freq=m.FREQ.values, n=m.N.values, p=m.P.values)
        print(f"{reg}: {len(m)}/{len(d)} variants", flush=True)
