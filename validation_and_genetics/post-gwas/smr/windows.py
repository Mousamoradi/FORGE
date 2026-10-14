"""Windows around the regions (GRCh38): probes within +/-1 Mb, GWAS variants within +/-3 Mb,
eQTL Catalogue fetch within +/-2 Mb (merged)."""
import pandas as pd
from common import OUT, out

R = pd.read_csv(f"{OUT}/loci/regions.tsv", sep="\t")
W = pd.DataFrame([dict(region=r.region_id, kind=k, chr=int(r.CHROM), start=max(1, int(r.region_start) - pad),
                       end=int(r.region_end) + pad)
                  for r in R.itertuples() for k, pad in (("probe", 1_000_000), ("variant", 3_000_000), ("fetch", 2_000_000))])
W.to_csv(out("smr", "windows.csv"), index=False)
merged = []
for c, s, e in W[W.kind == "fetch"].sort_values(["chr", "start"])[["chr", "start", "end"]].itertuples(index=False):
    if merged and merged[-1][0] == c and s <= merged[-1][2]:
        merged[-1][2] = max(merged[-1][2], e)
    else:
        merged.append([c, s, e])
pd.DataFrame(merged).to_csv(out("smr", "fetch_regions.tsv"), sep="\t", index=False, header=False)
