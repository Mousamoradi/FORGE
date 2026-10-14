"""MAGMA gene-level significance at 0.05/(genes x 15) and 0.05/(genes x M_eff)."""
import pandas as pd
from common import E, FCM, MEFF, OUT, out

sym = pd.read_csv(E["NCBI37_GENELOC"], sep="\t", header=None, usecols=[0, 5], names=["GENE", "symbol"])
g = pd.concat([pd.read_csv(f"{OUT}/magma/out/{t}.genes.out", sep=r"\s+").assign(trait=t) for t in FCM])
g = g.merge(sym, on="GENE")
NG = g.groupby("trait").size().max()
g["sig_x15"] = g.P < 0.05 / (NG * 15)
g["sig_meff"] = g.P < 0.05 / (NG * MEFF)
g[g.sig_meff].to_csv(out("magma", "significant.tsv"), sep="\t", index=False)
print(f"genes tested {NG}; x15: {g[g.sig_x15].symbol.nunique()} genes; M_eff: {g[g.sig_meff].symbol.nunique()} genes; "
      f"traits with >=1 gene (x15): {g[g.sig_x15].trait.nunique()}")
