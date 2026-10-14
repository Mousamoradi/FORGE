import pandas as pd
from common import OUT

cs = pd.read_csv(f"{OUT}/finemap/credible_sets.tsv", sep="\t")
dg = pd.read_csv(f"{OUT}/finemap/diagnostics.tsv", sep="\t")
print(f"credible sets {len(cs)} in {cs.region.nunique()} of {len(dg)} regions; median purity {cs.purity.median():.3f}")
print(f"single-variant {(cs.cs_size == 1).sum()}; <= 10 variants {(cs.cs_size <= 10).sum()}; "
      f"median PSD lambda {dg.lambda_psd.median():.3f}")
