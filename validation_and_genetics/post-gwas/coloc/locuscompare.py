"""Locus-compare check of each call: CONFIRMED when coloc's shared variant is in the top 5% of eQTL
-log10 P and Pearson r(-log10 P GWAS, -log10 P eQTL) > 0.3 over variants with either -log10 P > 2."""
import io, subprocess
import numpy as np, pandas as pd
from common import OUT, out

spec = pd.read_csv(f"{OUT}/loci/regions.tsv", sep="\t").set_index("region_id")
ds = pd.read_csv(f"{OUT}/coloc/qtl/datasets.tsv", sep="\t").set_index("dataset_id")
calls = pd.read_csv(f"{OUT}/coloc/calls.tsv", sep="\t")
key = lambda c, p, a, b: f"{c}:{p}:{min(str(a).upper(), str(b).upper())}_{max(str(a).upper(), str(b).upper())}"


def eqtl(r):
    s = spec.loc[r.region]
    if r.arm == "retina":
        d = pd.read_csv(f"{OUT}/coloc/retina_regions/chr{s.CHROM}.tsv.gz", sep="\t")
        d = d[(d.region == r.region) & (d.gene == r.molecular_trait_id)]
        return d.rename(columns={"effect": "ref", "other": "alt", "p": "p_eqtl"})[["chr", "pos", "ref", "alt", "p_eqtl"]]
    url = ds.loc[r.dataset_id, "ftp_path"].replace("ftp://", "https://")
    txt = subprocess.run(f'tabix "{url}" {s.CHROM}:{s.region_start}-{s.region_end}', shell=True,
                         capture_output=True, text=True, cwd=out("coloc", "tbi", "x")[:-1]).stdout
    d = pd.read_csv(io.StringIO(txt), sep="\t", header=None, usecols=[0, 1, 2, 3, 4, 8])
    d.columns = ["mt", "chr", "pos", "ref", "alt", "p_eqtl"]
    return d[d.mt == r.molecular_trait_id]


rows = []
for r in calls.itertuples():
    g = pd.read_csv(f"{OUT}/finemap/regions/{r.region}.tsv", sep="\t")
    g["k"] = [key(c, p, a, b) for c, p, a, b in zip(g.CHR, g.POS, g.A1, g.A2)]
    e = eqtl(r)
    e["k"] = [key(c, p, a, b) for c, p, a, b in zip(e.chr, e.pos, e.ref, e.alt)]
    M = g.drop_duplicates("k").merge(e.drop_duplicates("k")[["k", "p_eqtl"]], on="k")
    M["g"], M["e"] = -np.log10(M.P.clip(lower=1e-300)), -np.log10(M.p_eqtl.clip(lower=1e-300))
    sig = M[(M.g > 2) | (M.e > 2)]
    pear = np.corrcoef(sig.g, sig.e)[0, 1] if len(sig) > 20 else np.nan
    sh = M[M.k == r.shared_variant]
    pct = 100 * (M.e <= sh.e.iloc[0]).mean() if len(sh) else np.nan
    rows.append(dict(arm=r.arm, region=r.region, symbol=r.symbol, n=len(M), pearson=pear, shared_eqtl_pctile=pct,
                     verdict="CONFIRMED" if pct >= 95 and pear > 0.3 else "NOT_CONFIRMED"))
V = pd.DataFrame(rows)
V.to_csv(out("coloc", "locuscompare.tsv"), sep="\t", index=False)
print(V.round(3).to_string(index=False))
print("confirmed genes:", sorted(set(V[V.verdict == "CONFIRMED"].symbol)))
