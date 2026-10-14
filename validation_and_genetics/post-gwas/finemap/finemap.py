"""Run SuSiE-RSS in every region with UK Biobank LD and collect credible sets."""
import os, subprocess
import numpy as np, pandas as pd
from common import E, OUT, out

HERE = os.path.dirname(os.path.abspath(__file__))
R = pd.read_csv(f"{OUT}/loci/regions.tsv", sep="\t")
rows, diag = [], []
for r in R.itertuples():
    f = f"{OUT}/ld/ukb/{r.region_id}.npz"
    if not os.path.exists(f):
        continue
    z = np.load(f, allow_pickle=True)
    M = pd.DataFrame(dict(SNP=z["snp"].astype(str), POS=z["pos"], A1=z["a1"].astype(str), A2=z["a2"].astype(str),
                          FREQ=z["freq"], N=z["n"], BETA=z["beta"], SE=z["se"], P=z["p"]))
    LD = z["R"].astype(np.float64)
    pref = out("finemap", "susie", r.region_id)
    M.to_csv(f"{pref}.dat.tsv", sep="\t", index=False)
    ((LD + LD.T) / 2).tofile(f"{pref}.R.bin")
    subprocess.run([E["RSCRIPT"], f"{HERE}/susie_rss.R", pref], check=True)
    dg = dict(l.split() for l in open(f"{pref}.diag.txt"))
    diag.append(dict(region=r.region_id, n_variants=len(M), lambda_psd=float(dg["lambda"]), s_rss=float(dg["s_rss"])))
    for ln in open(f"{pref}.cs.txt"):
        if not ln.strip():
            continue
        nm, size, cov, top, pip, pur, idx = ln.rstrip("\n").split("\t")
        t = M.iloc[int(top) - 1]
        rows.append(dict(region=r.region_id, chr=r.CHROM, trait=r.best_top_trait, signal=nm, cs_size=int(size),
                         cs_coverage=float(cov), purity=float(pur), PIP=float(pip), SNP=t.SNP, POS=int(t.POS),
                         A1=t.A1, A2=t.A2, BETA=t.BETA, SE=t.SE, P=t.P,
                         cs_members=";".join(M.iloc[[int(i) - 1 for i in idx.split(",")]].SNP)))
    print(r.region_id, len(M), "variants", flush=True)

pd.DataFrame(rows).to_csv(out("finemap", "credible_sets.tsv"), sep="\t", index=False)
pd.DataFrame(diag).to_csv(out("finemap", "diagnostics.tsv"), sep="\t", index=False)
