"""Combine both arms, apply the call rule (PP4 >= 0.8, PP4/(PP3+PP4) >= 0.9, >= 500 shared variants)."""
import glob
import numpy as np, pandas as pd
from common import OUT, out

sym = pd.read_csv(f"{OUT}/ref/genes.tsv", sep="\t", header=None, usecols=[0, 1], names=["ensg", "symbol"])
sym = dict(zip(sym.ensg, sym.symbol))
cat = lambda p: pd.concat([pd.read_csv(f) for f in glob.glob(p)], ignore_index=True)

G = cat(f"{OUT}/coloc/gtex/*.tsv.gz")
G["ensg"] = G.molecular_trait_id.str.extract(r"(ENSG\d+)")[0].fillna(G.gene_id.str.replace(r"\..*", "", regex=True))
G = G.assign(arm="GTEx", n_shared=G.nsnps, shared_variant=G.hit1.astype(str))
R = cat(f"{OUT}/coloc/retina/*.tsv.gz")
R["ensg"] = R.gene.str.replace(r"\..*", "", regex=True)
R = R.assign(arm="retina", n_shared=R.n_common, shared_variant=R.shared_vk, molecular_trait_id=R.gene,
             dataset_id="EyeGEx", tissue="retina", quant="ge")
A = pd.concat([G, R], ignore_index=True)
A["symbol"] = A.ensg.map(sym).fillna(A.ensg)
A["ratio"] = A["PP.H4.abf"] / (A["PP.H3.abf"] + A["PP.H4.abf"])
A["called"] = (A["PP.H4.abf"] >= 0.8) & (A.ratio >= 0.9) & (A.n_shared >= 500)
A.to_csv(out("coloc", "all_tests.tsv"), sep="\t", index=False)

wk = pd.read_csv(f"{OUT}/coloc/qtl/worklist.tsv", sep="\t")
key = ["region", "dataset_id", "molecular_trait_id"]
cov = len(G[key].drop_duplicates()) / len(wk[key].drop_duplicates())
calls = (A[A.called].sort_values("PP.H4.abf", ascending=False)
         .groupby(["arm", "region", "symbol"], as_index=False).first())
calls.to_csv(out("coloc", "calls.tsv"), sep="\t", index=False)
print(f"tests: GTEx {len(G)}, retina {len(R)}, total {len(A)}; GTEx tuple coverage {100 * cov:.1f}%")
print(calls[["arm", "region", "symbol", "trait", "PP.H4.abf"]].to_string(index=False))
