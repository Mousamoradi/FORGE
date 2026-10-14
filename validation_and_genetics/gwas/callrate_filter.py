"""Per-variant call-rate filter (N >= 90% of 39,381) and per-trait QC on REGENIE output.
usage: callrate_filter.py fcm|wovf <regenie_dir>"""
import gzip, os, sys
import numpy as np, pandas as pd
from common import traits, out, N_GWAS

prefix, rdir = sys.argv[1], sys.argv[2]
NMIN = int(np.ceil(0.90 * N_GWAS))
cl = out("clump", prefix, "x")[:-1]
os.makedirs(f"{cl}/ss_f", exist_ok=True)

rows, union = [], set()
for t in traits(prefix):
    n_tot = n_keep = n_sig = 0
    chi = []
    with gzip.open(f"{rdir}/assoc_{t}.regenie.gz", "rt") as fh, open(f"{cl}/ss_f/{t}.tsv", "w") as ss:
        fh.readline()
        ss.write("ID\tLOG10P\n")
        for line in fh:
            f = line.split()
            n_tot += 1
            if int(f[6]) < NMIN:
                continue
            n_keep += 1
            chi.append(float(f[10]))
            lp = float(f[11])
            n_sig += lp > 7.30103
            if lp > 5:
                union.add(f[2])
            ss.write(f"{f[2]}\t{f[11]}\n")
    lam = float(np.median(chi) / 0.4549364)
    rows.append(dict(pheno=t, n_variants_prefilter=n_tot, n_variants=n_keep,
                     lambda_GC=round(lam, 4), n_sig_5e8=n_sig))
    print(t, n_keep, round(lam, 4), n_sig, flush=True)

qc = pd.DataFrame(rows)
qc.to_csv(out(f"gwas_qc_{prefix}.tsv"), sep="\t", index=False)
open(f"{cl}/union_p1e5.ids", "w").write("\n".join(sorted(union)) + "\n")
print(f"lambda_GC {qc.lambda_GC.min():.3f}-{qc.lambda_GC.max():.3f} (median {qc.lambda_GC.median():.3f})")
