"""Bonferroni within each family (retina expression, GTEx expression, GTEx splicing); a gene is called
when p_SMR < 0.05/n_family and p_HEIDI > 0.05."""
import glob, os
import pandas as pd
from common import OUT, out

fams = {"retina_expression": "out_retina", "gtex_expression": "out_expr", "gtex_splicing": "out_sqtl"}
rows, called = [], set()
for fam, sub in fams.items():
    d = pd.concat([pd.read_csv(f, sep="\t").assign(tissue=os.path.basename(f).split(".")[0], trait=os.path.basename(f).split(".")[1])
                   for f in glob.glob(f"{OUT}/smr/{sub}/*.smr")])
    thr = 0.05 / len(d)
    d["called"] = (d.p_SMR < thr) & (d.p_HEIDI > 0.05)
    d.assign(family=fam, bonferroni=thr).to_csv(out("smr", f"{fam}.tsv"), sep="\t", index=False)
    genes = sorted(set(d[d.called].Gene))
    called |= set(genes)
    print(f"{fam}: {len(d)} tests, threshold {thr:.3g}, called genes {genes}")
print(len(called), "genes called in any family")
