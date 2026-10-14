"""Region table used by fine-mapping, colocalization and SMR: regions in genomic order (S01..),
lead span padded by 500 kb, with the strongest lead's trait.
usage: regions.py [loci.tsv] [loci_regions.tsv]"""
import sys
import pandas as pd
from common import E, out

loci = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else E["LOCI"], sep="\t")
reg = pd.read_csv(sys.argv[2] if len(sys.argv) > 2 else E["LOCI_REGIONS"], sep="\t")
reg = reg.sort_values(["CHROM", "start"]).reset_index(drop=True)
reg["region_id"] = [f"S{i + 1:02d}" for i in range(len(reg))]
lead2reg = {(int(r.CHROM), lid): r.region_id for r in reg.itertuples() for lid in str(r.lead_ids).split(",")}
loci["region_id"] = [lead2reg[(int(c), i)] for c, i in zip(loci.CHROM, loci.ID)]
loci.sort_values(["CHROM", "POS"]).to_csv(out("loci", "leads.tsv"), sep="\t", index=False)

PAD = 500_000
rows = []
for rid, g in loci.groupby("region_id"):
    b = g.loc[g.P.idxmin()]
    rows.append(dict(region_id=rid, CHROM=int(g.CHROM.iloc[0]),
                     region_start=max(0, int(g.POS.min()) - PAD), region_end=int(g.POS.max()) + PAD,
                     n_leads=len(g), best_ID=b.ID, best_POS=int(b.POS), best_P=float(b.P),
                     best_top_trait=b.top_trait,
                     nearest_pc_gene=",".join(pd.unique(g.nearest_pc_gene.astype(str)))))
R = pd.DataFrame(rows).sort_values(["CHROM", "region_start"])
R.to_csv(out("loci", "regions.tsv"), sep="\t", index=False)
print(len(R), "regions,", R.n_leads.sum(), "leads")
