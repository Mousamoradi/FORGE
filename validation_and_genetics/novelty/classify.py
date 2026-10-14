"""Novelty of each lead against the reference: LEAD (exact chr:pos) > PROXY (an LD proxy at a
reference position) > PROXIMITY (reference variant within 500 kb) > novel.
usage: classify.py <leads.tsv> [--proxies proxies.tsv] [--exclude SOURCE]"""
import argparse
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from common import E, out

ap = argparse.ArgumentParser()
ap.add_argument("leads")
ap.add_argument("--proxies")
ap.add_argument("--exclude", default="")
ap.add_argument("--label", default="loci")
a = ap.parse_args()

ref = pd.read_csv(E["NOVELTY_REFERENCE"], sep="\t")
ref = ref[~ref.ref_source.isin(a.exclude.split(","))]
ref = {s: g.drop_duplicates(["chr", "pos"]) for s, g in ref.groupby("ref_source")}
leads = pd.read_csv(a.leads, sep="\t")
prox = pd.read_csv(a.proxies, sep="\t") if a.proxies else pd.DataFrame(columns=["lead_ID", "proxy_chr", "proxy_pos", "r2"])
RANK = {"LEAD": 3, "PROXY": 2, "PROXIMITY": 1, "": 0}


def regions(df):
    lab, n = [], 0
    for c, g in df.sort_values(["CHROM", "POS"]).groupby("CHROM"):
        prev = None
        for i, p in zip(g.index, g.POS):
            if prev is None or p - prev > 500_000:
                n += 1
            lab.append((i, n))
            prev = p
    return pd.Series(dict(lab))


leads["region"] = regions(leads)
res = []
for thr in (0.2, 0.8):
    ps = {k: set(zip(g.proxy_chr, g.proxy_pos)) for k, g in prox[prox.r2 >= thr].groupby("lead_ID")}
    for l in leads.itertuples():
        best = ""
        for s, d in ref.items():
            same = d[d.chr == l.CHROM]
            dist = (same.pos - l.POS).abs()
            if (dist == 0).any():
                c = "LEAD"
            elif any((l.CHROM, p) in ps.get(l.ID, ()) for p in same.pos):
                c = "PROXY"
            elif (dist <= 500_000).any():
                c = "PROXIMITY"
            else:
                c = ""
            best = max(best, c, key=RANK.get)
        res.append(dict(r2=thr, ID=l.ID, CHROM=l.CHROM, POS=l.POS, P=l.P, region=l.region,
                        best_class=best or "novel"))

R = pd.DataFrame(res)
R.to_csv(out("novelty", f"{a.label}_classified.tsv"), sep="\t", index=False)
for thr, g in R.groupby("r2"):
    nov = g.best_class == "novel"
    lp = -np.log10(g.P)
    m = g[g.P < 1e-8]
    mixed = sum(x.best_class.eq("novel").any() and x.best_class.ne("novel").any() for _, x in g.groupby("region"))
    print(f"r2>={thr}: novel {nov.sum()} in {g[nov].region.nunique()} regions; "
          f"{g.best_class.value_counts().to_dict()}; M_eff tier {len(m)} loci, "
          f"{(m.best_class == 'novel').sum()} novel in {m[m.best_class == 'novel'].region.nunique()} regions; "
          f"median -log10P {lp[nov].median():.2f} vs {lp[~nov].median():.2f}, "
          f"MW P = {mannwhitneyu(lp[nov], lp[~nov]).pvalue:.2f}; mixed regions {mixed}")
