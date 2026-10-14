"""Pool per-trait index variants, re-clump across traits, merge leads within 500 kb into regions,
annotate the nearest gene (GENCODE v47) and attach lead effects.
usage: define_loci.py fcm|wovf <regenie_dir>"""
import gzip, re, subprocess, sys
import numpy as np, pandas as pd
from common import E, traits, out

prefix, rdir = sys.argv[1], sys.argv[2]
TR = traits(prefix)
C = out("clump", prefix, "x")[:-1]
num = lambda t: int(re.search(r"(\d+)$", t).group(1))

per = []
for t in TR:
    try:
        d = pd.read_csv(f"{C}/per_trait/{t}.clumps", sep="\t")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        continue
    d.columns = [c.lstrip("#") for c in d.columns]
    per.append(d[["CHROM", "POS", "ID", "P"]].assign(pheno=t))
per = pd.concat(per, ignore_index=True)
print("per-trait index variants:", len(per))

pool = (per.groupby(["CHROM", "POS", "ID"])
        .agg(minP=("P", "min"), n_traits=("pheno", "nunique"),
             traits=("pheno", lambda s: ",".join(sorted(set(s), key=num))))
        .reset_index())
pool[["ID", "minP"]].rename(columns={"minP": "P"}).to_csv(f"{C}/pooled.tsv", sep="\t", index=False)
subprocess.run([E["PLINK2"], "--bfile", f"{C}/ldref/sub", "--clump", f"{C}/pooled.tsv",
                "--clump-id-field", "ID", "--clump-p-field", "P", "--clump-p1", "5e-8",
                "--clump-p2", "1", "--clump-r2", "0.1", "--clump-kb", "500", "--clump-unphased",
                "--out", f"{C}/study_loci"], check=True, capture_output=True)
loci = pd.read_csv(f"{C}/study_loci.clumps", sep="\t")
loci.columns = [c.lstrip("#") for c in loci.columns]
loci = loci[["CHROM", "POS", "ID", "P"]].merge(pool[["ID", "n_traits", "traits"]], on="ID")
loci = loci.sort_values(["CHROM", "POS"]).reset_index(drop=True)

reg, cur = [], None
for r in loci.itertuples():
    if cur is None or r.CHROM != cur["CHROM"] or r.POS - cur["end"] > 500_000:
        if cur:
            reg.append(cur)
        cur = dict(CHROM=r.CHROM, start=r.POS, end=r.POS, n_leads=0, lead_ids=[],
                   best_P=r.P, best_ID=r.ID, traits=set())
    cur["end"] = r.POS
    cur["n_leads"] += 1
    cur["lead_ids"].append(r.ID)
    cur["traits"] |= set(r.traits.split(","))
    if r.P < cur["best_P"]:
        cur["best_P"], cur["best_ID"] = r.P, r.ID
reg.append(cur)
REG = pd.DataFrame([{**x, "lead_ids": ",".join(x["lead_ids"]), "n_traits": len(x["traits"]),
                     "traits": ",".join(sorted(x["traits"], key=num))} for x in reg])
REG.to_csv(out(f"loci_regions_{prefix}.tsv"), sep="\t", index=False)

genes, feats = [], []
with gzip.open(E["GENCODE_V47"], "rt") as fh:
    for line in fh:
        if line[0] == "#":
            continue
        f = line.rstrip("\n").split("\t")
        if not re.fullmatch(r"chr\d+", f[0]):
            continue
        c, s, e = int(f[0][3:]), int(f[3]), int(f[4])
        gt = re.search(r'gene_type "([^"]+)"', f[8]).group(1)
        if f[2] == "gene":
            genes.append((c, s, e, re.search(r'gene_name "([^"]+)"', f[8]).group(1), gt))
        elif f[2] in ("CDS", "exon", "UTR"):
            feats.append((c, s, e, f[2], gt))
G = pd.DataFrame(genes, columns=["chrom", "start", "end", "gene", "biotype"])
F = pd.DataFrame(feats, columns=["chrom", "start", "end", "feature", "biotype"])
PC = G[G.biotype == "protein_coding"]


def nearest(c, p, sub):
    s = sub[sub.chrom == c]
    d = np.where((p >= s.start) & (p <= s.end), 0,
                 np.minimum(np.abs(s.start - p), np.abs(s.end - p)))
    i = int(np.argmin(d))
    return s.gene.values[i], int(d[i])


def location(c, p):
    fs = F[(F.chrom == c) & (F.start <= p) & (F.end >= p)]
    if (fs.feature == "CDS").any():
        return "CDS"
    if (fs.feature == "UTR").any():
        return "UTR"
    if ((fs.feature == "exon") & (fs.biotype != "protein_coding")).any():
        return "exon of non-coding transcript"
    if (fs.feature == "exon").any():
        return "exon (non-CDS)"
    inpc = PC[(PC.chrom == c) & (PC.start <= p) & (PC.end >= p)]
    return "intronic" if len(inpc) else "intergenic"


for col, sub in (("pc", PC), ("any", G)):
    res = [nearest(int(r.CHROM), int(r.POS), sub) for r in loci.itertuples()]
    loci[f"nearest_{col}_gene"] = [x[0] for x in res]
    loci[f"dist_{col}_bp"] = [x[1] for x in res]
loci["location_class"] = [location(int(r.CHROM), int(r.POS)) for r in loci.itertuples()]

want, best = set(loci.ID), {}
for t in TR:
    with gzip.open(f"{rdir}/assoc_{t}.regenie.gz", "rt") as fh:
        fh.readline()
        for line in fh:
            f = line.split()
            if f[2] in want and (f[2] not in best or float(f[11]) > best[f[2]]["lp"]):
                best[f[2]] = dict(ID=f[2], top_trait=t, EA=f[4], NEA=f[3], EAF=float(f[5]),
                                  N=int(f[6]), BETA=float(f[8]), SE=float(f[9]), lp=float(f[11]))
loci = loci.merge(pd.DataFrame(best.values()).drop(columns="lp"), on="ID")
loci = loci.sort_values("P").reset_index(drop=True)
loci.insert(0, "locus", range(1, len(loci) + 1))
loci.to_csv(out(f"loci_{prefix}.tsv"), sep="\t", index=False)

print(f"loci {len(loci)}, regions {len(REG)}, multi-trait loci {(loci.n_traits > 1).sum()}")
for lab, thr in (("5e-8", 5e-8), ("5e-8/M_eff", 1e-8), ("5e-8/15", 5e-8 / 15)):
    print(f"loci at {lab}: {(loci.P < thr).sum()}")
