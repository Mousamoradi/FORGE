"""Regional summary statistics for fine-mapping: each region in its strongest trait."""
import pandas as pd
from common import OUT, out, read_sumstats

R = pd.read_csv(f"{OUT}/loci/regions.tsv", sep="\t")
cols = ["SNP", "CHR", "POS", "A1", "A2", "FREQ", "N", "BETA", "SE", "P"]
for t, g in R.groupby("best_top_trait"):
    d = read_sumstats(t, usecols=cols)
    for r in g.itertuples():
        s = d[(d.CHR == r.CHROM) & d.POS.between(max(1, r.region_start), r.region_end)]
        s.to_csv(out("finemap", "regions", f"{r.region_id}.tsv"), sep="\t", index=False)
    print(t, len(g), "regions", flush=True)
