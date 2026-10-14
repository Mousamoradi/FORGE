"""eQTL Catalogue GTEx v8 (QTS000015), 18 tissues x {ge, leafcutter}: credible sets, the
(region, dataset, molecular trait, credible set) worklist, and the SuSiE lbf_variable rows of every
worklist molecular trait. usage: fetch_qtl.py [datasets|cs|worklist|lbf] ..."""
import gzip, os, subprocess, sys, urllib.request
import pandas as pd
from common import E, OUT, out

TISSUE = {"adipose_subcutaneous": "Adipose_Subcutaneous", "artery_tibial": "Artery_Tibial", "blood": "Whole_Blood",
          "brain_amygdala": "Brain_Amygdala", "brain_anterior_cingulate_cortex": "Brain_Anterior_cingulate_cortex_BA24",
          "brain_caudate": "Brain_Caudate_basal_ganglia", "brain_cerebellar_hemisphere": "Brain_Cerebellar_Hemisphere",
          "brain_cerebellum": "Brain_Cerebellum", "brain_cortex": "Brain_Cortex", "brain_frontal_cortex": "Brain_Frontal_Cortex_BA9",
          "brain_hippocampus": "Brain_Hippocampus", "brain_hypothalamus": "Brain_Hypothalamus",
          "brain_nucleus_accumbens": "Brain_Nucleus_accumbens_basal_ganglia", "brain_putamen": "Brain_Putamen_basal_ganglia",
          "brain_spinal_cord": "Brain_Spinal_cord_cervical_c-1", "brain_substantia_nigra": "Brain_Substantia_nigra",
          "fibroblast": "Cells_Cultured_fibroblasts", "nerve_tibial": "Nerve_Tibial"}
Q = out("coloc", "qtl", "x")[:-1]
https = lambda u: u.replace("ftp://ftp.ebi.ac.uk", "https://ftp.ebi.ac.uk")
steps = sys.argv[1:] or ["datasets", "cs", "worklist", "lbf"]

if "datasets" in steps:
    t = pd.read_csv(E["EQTLCAT_PATHS"], sep="\t")
    t = t[(t.study_id == "QTS000015") & t.quant_method.isin(["ge", "leafcutter"]) & t.sample_group.isin(TISSUE)].copy()
    t["twas_tissue"] = t.sample_group.map(TISSUE)
    t = t.sort_values(["quant_method", "sample_group"])
    t[["dataset_id", "sample_group", "twas_tissue", "quant_method", "sample_size", "ftp_cs_path", "ftp_lbf_path", "ftp_path"]] \
        .to_csv(f"{Q}/datasets.tsv", sep="\t", index=False)
    print(len(t), "datasets")
ds = pd.read_csv(f"{Q}/datasets.tsv", sep="\t")

if "cs" in steps:
    os.makedirs(f"{Q}/cs", exist_ok=True)
    for d in ds.itertuples():
        f = f"{Q}/cs/{d.dataset_id}.credible_sets.tsv.gz"
        if not os.path.exists(f):
            subprocess.run(["curl", "-sfL", "--retry", "5", "-o", f, https(d.ftp_cs_path)], check=True)

if "worklist" in steps:
    R = pd.read_csv(f"{OUT}/loci/regions.tsv", sep="\t")
    rows = []
    for d in ds.itertuples():
        t = pd.read_csv(f"{Q}/cs/{d.dataset_id}.credible_sets.tsv.gz", sep="\t",
                        usecols=["molecular_trait_id", "gene_id", "cs_id", "variant"])
        sp = t.variant.str.split("_", n=2, expand=True)
        t["chr"], t["pos"] = sp[0].str.replace("chr", ""), pd.to_numeric(sp[1], errors="coerce")
        for r in R.itertuples():
            s = t[(t.chr == str(r.CHROM)) & t.pos.between(r.region_start, r.region_end)]
            if len(s):
                rows.append(s[["molecular_trait_id", "gene_id", "cs_id"]].drop_duplicates()
                            .assign(region=r.region_id, dataset_id=d.dataset_id))
    wk = pd.concat(rows)
    wk.to_csv(f"{Q}/worklist.tsv", sep="\t", index=False)
    print(len(wk), "worklist rows in", wk.region.nunique(), "regions")

if "lbf" in steps:
    wk = pd.read_csv(f"{Q}/worklist.tsv", sep="\t")
    os.makedirs(f"{Q}/lbf", exist_ok=True)
    for d in ds.itertuples():
        keep = set(wk[wk.dataset_id == d.dataset_id].molecular_trait_id)
        f = f"{Q}/lbf/{d.dataset_id}.lbf.tsv.gz"
        if not keep or os.path.exists(f):
            continue
        with urllib.request.urlopen(https(d.ftp_lbf_path)) as src, gzip.open(src, "rt") as g, gzip.open(f + ".part", "wt") as o:
            o.write(next(g))
            for line in g:
                if line.split("\t", 1)[0] in keep:
                    o.write(line)
        os.replace(f + ".part", f)
        print(d.dataset_id, flush=True)
