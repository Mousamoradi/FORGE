"""GTEx BESD per dataset. Expression: one probe per gene at the GENCODE v44 TSS. Splicing: one probe per
intron at its start. Probes must lie in a probe window; rsID-named variants only; A1 = alt, Freq = ac/an."""
import gzip, os, re, subprocess
import numpy as np, pandas as pd
from common import E, OUT

COLS = ["molecular_trait_id", "chromosome", "position", "ref", "alt", "variant", "ma_samples", "maf", "pvalue", "beta",
        "se", "type", "ac", "an", "r2", "molecular_trait_object_id", "gene_id", "median_tpm", "rsid"]
tss, sym = {}, {}
with gzip.open(E["GENCODE_V44"], "rt") as fh:
    for line in fh:
        f = line.split("\t", 8)
        if line[0] == "#" or f[2] != "gene":
            continue
        g = re.search(r'gene_id "([^"]+)"', f[8]).group(1).split(".")[0]
        nm = re.search(r'gene_name "([^"]+)"', f[8])
        tss[g] = int(f[3]) if f[6] == "+" else int(f[4])
        sym[g] = nm.group(1) if nm else g
P = pd.read_csv(f"{OUT}/smr/windows.csv").query("kind == 'probe'")
pw = {c: g[["start", "end"]].values for c, g in P.groupby("chr")}
inwin = lambda c, b: c in pw and bool(((pw[c][:, 0] <= b) & (pw[c][:, 1] >= b)).any())
ds = pd.read_csv(f"{OUT}/coloc/qtl/datasets.tsv", sep="\t")

for r in ds.itertuples():
    layer = "expr" if r.quant_method == "ge" else "sqtl"
    d = pd.read_csv(f"{OUT}/smr/eqtlcat/{r.dataset_id}.region.tsv.gz", sep="\t", header=None, names=COLS, comment="#",
                    dtype={"molecular_trait_id": str, "chromosome": str, "rsid": str, "gene_id": str})
    d = d[d.chromosome.str.fullmatch(r"\d+")]
    d["chromosome"] = d.chromosome.astype(int)
    pr = d.drop_duplicates("molecular_trait_id")[["molecular_trait_id", "gene_id", "chromosome"]].copy()
    pr.columns = ["probe", "gene", "chr"]
    pr["gene"] = pr.gene.str.split(".").str[0]
    if layer == "expr":
        pr["gene"] = pr.probe.str.split(".").str[0]
        pr["bp"] = pr.gene.map(tss)
        pr = pr.dropna(subset=["bp"])
    else:
        pr["bp"] = pr.probe.str.split(":").str[1].astype(int)
    pr["bp"] = pr.bp.astype(int)
    pr = pr[[inwin(c, b) for c, b in zip(pr.chr, pr.bp)]].set_index("probe")
    d = d[d.molecular_trait_id.isin(pr.index) & d.rsid.str.startswith("rs", na=False)]
    d = d[np.isfinite(d.se) & (d.se > 0)].drop_duplicates(["molecular_trait_id", "rsid"])
    stem = f"{OUT}/smr/gtex/{r.twas_tissue}.{layer}"
    os.makedirs(stem, exist_ok=True)
    with open(f"{stem}.flist", "w") as fl:
        fl.write("Chr\tProbeID\tGeneticDistance\tProbeBp\tGene\tOrientation\tPathOfEsd\n")
        for pid, x in d.groupby("molecular_trait_id", sort=False):
            path = os.path.abspath(f"{stem}/{re.sub(r'[^A-Za-z0-9_.+-]', '_', pid)}.esd")
            pd.DataFrame({"Chr": x.chromosome, "SNP": x.rsid, "Bp": x.position, "A1": x.alt, "A2": x.ref,
                          "Freq": x.ac / x.an, "Beta": x.beta, "se": x.se, "p": x.pvalue}) \
                .to_csv(path, sep="\t", index=False, float_format="%.6g")
            g = pr.loc[pid, "gene"]
            fl.write(f"{int(pr.loc[pid, 'chr'])}\t{pid}\t0\t{int(pr.loc[pid, 'bp'])}\t{sym.get(g, g)}\tNA\t{path}\n")
    subprocess.run([E["SMR"], "--eqtl-flist", f"{stem}.flist", "--make-besd", "--out", stem], check=True)
    print(r.twas_tissue, layer, len(pr), flush=True)
