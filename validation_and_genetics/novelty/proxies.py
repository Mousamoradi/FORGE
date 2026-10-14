"""LD proxies (r2 >= 0.2) of the leads: union of UK Biobank LD (fine-mapping regions) and the
MGBB European panel (+/-1 Mb, PLINK 2). usage: proxies.py [leads.tsv]"""
import glob, os, subprocess, sys
import numpy as np, pandas as pd
from common import E, OUT, out

leads = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else f"{OUT}/loci/leads.tsv", sep="\t")
rows = []

if "region_id" in leads:
    for reg, g in leads.groupby("region_id"):
        f = f"{OUT}/ld/ukb/{reg}.npz"
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        snp, pos, R = z["snp"].astype(str), z["pos"], z["R"]
        idx = {s: i for i, s in enumerate(snp)}
        for l in g.itertuples():
            i = idx.get(str(l.ID))
            if i is None:
                continue
            r2 = R[i].astype(float) ** 2
            for j in np.where((r2 >= 0.2) & (snp != l.ID))[0]:
                rows.append((l.ID, int(l.CHROM), int(pos[j]), r2[j]))

bim = pd.read_csv(E["MGBB_LD_PANEL"] + ".bim", sep="\t", header=None, names=["chr", "id", "cm", "pos", "a1", "a2"], dtype=str)
lk = {}
for c, i, p, a1, a2 in bim.itertuples(index=False):
    lk.setdefault((c, int(p)), []).append((i, {a1.upper(), a2.upper()}))
pid = {}
for l in leads.itertuples():
    cand = lk.get((str(l.CHROM), int(l.POS)), [])
    hit = next((i for i, al in cand if al == {str(l.EA).upper(), str(l.NEA).upper()}), None)
    if hit is None and len(cand) == 1:
        hit = cand[0][0]
    if hit:
        pid[hit] = l.ID
W = out("novelty", "x")[:-1]
open(f"{W}/lead_ids.txt", "w").write("\n".join(pid) + "\n")
subprocess.run([E["PLINK2"], "--bfile", E["MGBB_LD_PANEL"], "--r2-unphased", "cols=chrom,pos,id",
                "--ld-snp-list", f"{W}/lead_ids.txt", "--ld-window-kb", "1000", "--ld-window-r2", "0.2",
                "--ld-window", "9999999", "--out", f"{W}/mgbb"], check=True)
v = pd.read_csv(f"{W}/mgbb.vcor", sep="\t")
v = v[v.ID_A != v.ID_B]
rows += [(pid[a], int(c), int(p), float(r)) for a, c, p, r in zip(v.ID_A, v.CHROM_B, v.POS_B, v.UNPHASED_R2)]

P = pd.DataFrame(rows, columns=["lead_ID", "proxy_chr", "proxy_pos", "r2"])
P = P.groupby(["lead_ID", "proxy_chr", "proxy_pos"], as_index=False).r2.max()
P.to_csv(f"{W}/proxies.tsv", sep="\t", index=False)
print(len(P), "proxy rows for", P.lead_ID.nunique(), "leads")
