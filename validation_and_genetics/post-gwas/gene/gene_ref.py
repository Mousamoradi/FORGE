"""GENCODE v44 basic gene tables: all genes (ensg, symbol, type, chr, start, end) and protein-coding
genes (chr, start, end, strand, symbol)."""
import gzip, re
from common import E, out

allg, pc = [], []
with gzip.open(E["GENCODE_V44"], "rt") as f:
    for line in f:
        if line[0] == "#":
            continue
        p = line.split("\t")
        if p[2] != "gene":
            continue
        gid = re.search(r'gene_id "([^"]+)"', p[8]).group(1).split(".")[0]
        nm = re.search(r'gene_name "([^"]+)"', p[8])
        gt = re.search(r'gene_type "([^"]+)"', p[8]).group(1)
        nm = nm.group(1) if nm else ""
        c = p[0].replace("chr", "")
        allg.append((gid, nm, gt, c, p[3], p[4]))
        if gt == "protein_coding":
            pc.append((c, p[3], p[4], p[6], nm))
open(out("ref", "genes.tsv"), "w").write("\n".join("\t".join(r) for r in allg) + "\n")
open(out("ref", "pc_genes.tsv"), "w").write("\n".join("\t".join(r) for r in pc) + "\n")
print(len(allg), "genes,", len(pc), "protein-coding")
