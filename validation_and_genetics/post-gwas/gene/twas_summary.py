"""S-MultiXcan significance at 0.05/(features x 15) and 0.05/(features x M_eff); introns map to genes
by GENCODE overlap; genes cluster into 1 Mb physical loci."""
import glob, re, sqlite3
import numpy as np, pandas as pd
from common import E, FCM, MEFF, OUT, out

NFEAT = {"eqtl": 19352, "sqtl": 78648}
coords = pd.read_csv(f"{OUT}/ref/genes.tsv", sep="\t", header=None, names=["ensg", "symbol", "type", "chr", "start", "end"],
                     dtype={"chr": str}).drop_duplicates("ensg").set_index("ensg")
pc = pd.read_csv(f"{OUT}/ref/pc_genes.tsv", sep="\t", header=None, names=["chr", "start", "end", "strand", "gene"], dtype={"chr": str})
PC = {c: g.reset_index(drop=True) for c, g in pc.groupby("chr")}


def intron_gene(iid):
    c, s, e = re.match(r"intron_(\w+)_(\d+)_(\d+)$", iid).groups()
    s, e, g = int(s), int(e), PC[c]
    ov = g[(g.start <= e) & (g.end >= s)]
    if len(ov):
        return ov.iloc[int(np.argmin((ov.start - s).abs().values))].gene, c, s, e
    return g.iloc[int(np.argmin(np.minimum(np.abs(g.start - s), np.abs(g.end - e))))].gene, c, s, e


frames = []
for q in ("eqtl", "sqtl"):
    d = pd.concat([pd.read_csv(f"{OUT}/twas/smx/{t}_{q}.txt", sep="\t").assign(trait=t) for t in FCM]).dropna(subset=["pvalue"])
    if q == "eqtl":
        sym = {}
        for db in glob.glob(f"{E['MASHR_DIR']}/eqtl/*.db"):
            con = sqlite3.connect(db)
            sym.update(dict(con.execute("SELECT gene, genename FROM extra")))
            con.close()
        d["symbol"] = d.gene.map(sym).fillna(d.gene_name)
        ens = d.gene.str.split(".").str[0]
        for c in ("chr", "start", "end"):
            d[c] = ens.map(coords[c])
    else:
        mm = {i: intron_gene(i) for i in d.gene.unique()}
        for j, c in enumerate(("symbol", "chr", "start", "end")):
            d[c] = d.gene.map(lambda x: mm[x][j])
    d["layer"] = q
    d["sig_meff"] = d.pvalue < 0.05 / (NFEAT[q] * MEFF)
    d["sig_x15"] = d.pvalue < 0.05 / (NFEAT[q] * 15)
    frames.append(d)
A = pd.concat(frames)
S = A[A.sig_meff].copy()
S.to_csv(out("twas", "significant.tsv"), sep="\t", index=False)

g = S.sort_values("pvalue").drop_duplicates("symbol").dropna(subset=["chr"]).copy()
g["start"], g["end"] = g.start.astype(float), g.end.astype(float)
n_loci = 0
for c, x in g.sort_values(["chr", "start"]).groupby("chr"):
    end = None
    for r in x.itertuples():
        if end is None or r.start - end > 1_000_000:
            n_loci += 1
            end = r.end
        else:
            end = max(end, r.end)
e, s = set(S[S.layer == "eqtl"].symbol), set(S[S.layer == "sqtl"].symbol)
magma = set(pd.read_csv(f"{OUT}/magma/significant.tsv", sep="\t").symbol)
print(f"x15: {A[A.sig_x15].symbol.nunique()} genes; M_eff: {len(e | s)} genes ({len(e)} eQTL, {len(s)} sQTL, "
      f"{len(e & s)} both) at {n_loci} loci; also MAGMA-significant: {len((e | s) & magma)}")
