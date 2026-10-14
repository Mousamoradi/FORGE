"""Assign each region to a 3 Mb UK Biobank LD window (GRCh37). The GRCh38-to-GRCh37 offset is the
modal position difference over the region's rsIDs; the window is the one containing the shifted
region whose centre is closest to the region centre."""
import pandas as pd
from common import E, OUT, out

R = pd.read_csv(f"{OUT}/loci/regions.tsv", sep="\t")
bim = pd.read_csv(E["KG37_EUR_BIM"], sep="\t", header=None, usecols=[1, 3], names=["rs", "p37"])
bim = bim[bim.rs.str.startswith("rs")].drop_duplicates("rs").set_index("rs").p37

rows = []
for r in R.itertuples():
    d = pd.read_csv(f"{OUT}/finemap/regions/{r.region_id}.tsv", sep="\t", usecols=["SNP", "POS"])
    d = d[d.SNP.isin(bim.index)]
    off = int(pd.Series(d.POS.values - bim.loc[d.SNP].values).value_counts().index[0])
    s37, e37 = r.region_start - off, r.region_end - off
    cand = [(abs(s + 1.5e6 - (s37 + e37) / 2), s) for s in range(1, 300_000_001, 1_000_000)
            if s <= s37 and s + 3_000_000 >= e37]
    s = min(cand)[1]
    rows.append(dict(region=r.region_id, chr=r.CHROM, b37_start=s37, b37_end=e37,
                     win=f"chr{r.CHROM}_{s}_{s + 3_000_000}"))
W = pd.DataFrame(rows)
W.to_csv(out("ld", "windows.tsv"), sep="\t", index=False)
open(out("ld", "alkes_order.txt"), "w").write("\n".join(W.win.unique()) + "\n")
print(len(W), "regions in", W.win.nunique(), "windows")
