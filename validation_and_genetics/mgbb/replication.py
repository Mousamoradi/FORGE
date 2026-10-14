"""Replication of discovery lead variant-axis pairs in MGBB. Inference is clustered on 1 Mb loci:
locus-block sign-flip permutation for sign concordance, locus-block bootstrap for r and the Deming slope
(lambda = mean SE^2 MGBB / mean SE^2 discovery). Power under a normal approximation, ncp = |beta_disc| / SE_MGBB."""
import math
import numpy as np, pandas as pd
from scipy import stats
from common import E, OUT, out

SEED = 20260730
COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}
flip = lambda a: "".join(COMP.get(b, "N") for b in str(a)[::-1])

ls = pd.read_csv(E["MGBB_LEADSET"], sep="\t")
ls = ls[(ls.tier == "lead") & ls.trait.str.startswith("fcm_z") & ~ls.ambiguous]
rows = []
for t, g in ls.groupby("trait"):
    m = pd.read_csv(f"{OUT}/mgbb/glm/mgbb.{t}.glm.linear", sep="\t")
    m = m[m.TEST == "ADD"].copy()
    m["other"] = np.where(m.A1 == m.ALT, m.REF, m.ALT)
    m = m.drop_duplicates(["#CHROM", "POS"]).set_index(["#CHROM", "POS"])
    for r in g.itertuples():
        if (r.chrom, r.pos_b38) not in m.index:
            continue
        x = m.loc[(r.chrom, r.pos_b38)]
        a1, ot, ea, oa = str(x.A1).upper(), str(x.other).upper(), r.effect_allele.upper(), r.other_allele.upper()
        if {ea, oa} == {a1, ot}:
            b = x.BETA if ea == a1 else -x.BETA
        elif {flip(ea), flip(oa)} == {a1, ot}:
            b = x.BETA if flip(ea) == a1 else -x.BETA
        else:
            continue
        rows.append(dict(trait=t, chrom=r.chrom, pos=r.pos_b38, x=r.BETA, sx=r.SE, y=b, sy=x.SE))
d = pd.DataFrame(rows)
d.to_csv(out("mgbb", "replication_pairs.tsv"), sep="\t", index=False)

pairs = sorted(set(zip(d.chrom, d.pos)))
lid, cur, prev = {}, 0, None
for c, p in pairs:
    if prev is None or c != prev[0] or p - prev[1] > 1_000_000:
        cur += 1
    lid[(c, p)], prev = cur, (c, p)
locus = np.array([lid[k] for k in zip(d.chrom, d.pos)])
x, y, sx, sy = (d[c].to_numpy(float) for c in ("x", "y", "sx", "sy"))
conc = ((x > 0) == (y > 0)).astype(int)
uniq = np.unique(locus)


def boot(fn, seed=SEED + 2, B=10000):
    rng = np.random.default_rng(seed)
    idx = {l: np.where(locus == l)[0] for l in uniq}
    v = []
    for _ in range(B):
        i = np.concatenate([idx[l] for l in rng.choice(uniq, len(uniq), replace=True)])
        try:
            s = fn(i)
            if s is not None and np.isfinite(s):
                v.append(s)
        except Exception:
            pass
    return np.percentile(v, [2.5, 97.5])


def deming(i, lam):
    xb, yb = x[i].mean(), y[i].mean()
    sxx, syy, sxy = ((x[i] - xb) ** 2).sum(), ((y[i] - yb) ** 2).sum(), ((x[i] - xb) * (y[i] - yb)).sum()
    return ((syy - lam * sxx) + math.sqrt((syy - lam * sxx) ** 2 + 4 * lam * sxy ** 2)) / (2 * sxy)


n_l = np.array([(locus == l).sum() for l in uniq])
c_l = np.array([conc[locus == l].sum() for l in uniq])
K = np.where(np.random.default_rng(SEED + 1).random((50000, len(uniq))) < 0.5, n_l - c_l, c_l).sum(1)
p_perm = (1 + (K >= conc.sum()).sum()) / 50001
design = np.sum((n_l - 2 * c_l) ** 2) / len(d)
lam = np.mean(sy ** 2) / np.mean(sx ** 2)
all_i = np.arange(len(d))
r_ci = boot(lambda i: stats.pearsonr(x[i], y[i])[0] if len(np.unique(x[i])) > 1 else None)
dm_ci = boot(lambda i: deming(i, lam))
ncp = np.abs(x) / sy
pw = {"one-sided 0.05": stats.norm.cdf(ncp - stats.norm.isf(0.05)),
      "two-sided 0.05": stats.norm.cdf(ncp - stats.norm.isf(0.025)) + stats.norm.cdf(-ncp - stats.norm.isf(0.025)),
      "one-sided Bonferroni": stats.norm.cdf(ncp - stats.norm.isf(0.05 / len(d))),
      "two-sided Bonferroni": stats.norm.cdf(ncp - stats.norm.isf(0.025 / len(d)))}

print(f"pairs {len(d)}, variants {len(pairs)}, loci {len(uniq)}; design effect {design:.2f}, effective n {len(d) / design:.1f}")
print(f"sign concordance {conc.sum()}/{len(d)} = {100 * conc.mean():.1f}%, locus-block permutation P = {p_perm:.2g}")
print(f"r = {stats.pearsonr(x, y)[0]:.2f} ({r_ci[0]:.2f}-{r_ci[1]:.2f})")
print(f"Deming lambda {lam:.1f}, slope {deming(all_i, lam):.2f} ({dm_ci[0]:.2f}-{dm_ci[1]:.2f})")
print(f"median power (one-sided 0.05) {np.median(pw['one-sided 0.05']):.3f}; pairs with >= 80% power: "
      + ", ".join(f"{k} {int((v >= 0.8).sum())}" for k, v in pw.items()))
