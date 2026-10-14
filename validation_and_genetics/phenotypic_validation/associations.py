"""Phenotypic validation in the genotyped European set (N = 44,477).
UKB_OUTCOMES: one row per participant with age, sex, PC1-PC10, fcm_z1..fcm_z15 and the outcome
columns listed below (binary 0/1 or quantitative)."""
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from common import E, FCM, MEFF, bh, out

FAMILIES = {
    "criterion": ["poag_icd", "glaucoma_any", "cataract_any", "AMD_any", "DR_any", "retdetach_H33"],
    "convergent": ["IOPg", "CRF", "CupDiscRatio"],
    "discriminant": ["neg_osteoarthritis", "neg_asthma", "neg_gallstones", "neg_inguinalhernia",
                     "neg_breastcancer", "neg_mi", "neg_depression"],
}
COV = ["age", "sex"] + [f"PC{i}" for i in range(1, 11)]
STUDYWIDE = 0.05 / (MEFF * len(FAMILIES))


def blom(x):
    r = x.rank(method="average")
    z = pd.Series(stats.norm.ppf((r - 0.375) / (x.notna().sum() + 0.25)), index=x.index)
    return (z - z.mean()) / z.std(ddof=0)


d = pd.read_csv(E["UKB_OUTCOMES"], sep="\t")
for a in FCM:
    d[a] = blom(d[a])

rows = []
for fam, outcomes in FAMILIES.items():
    for y in outcomes:
        binary = set(d[y].dropna().unique()) <= {0, 1}
        block = []
        for a in FCM:
            s = d[[a, y] + COV].dropna()
            X0 = sm.add_constant(s[COV].astype(float))
            X1 = sm.add_constant(s[[a] + COV].astype(float))
            fit = (lambda X: sm.Logit(s[y], X).fit(disp=0)) if binary else (lambda X: sm.OLS(s[y], X).fit())
            m1, m0 = fit(X1), fit(X0)
            lrt = 2 * (m1.llf - m0.llf)
            b, se = m1.params[a], m1.bse[a]
            block.append(dict(endo=a, outcome=y, family=fam, n=len(s),
                              n_case=int(s[y].sum()) if binary else np.nan, beta=b, se=se,
                              effect=np.exp(b) if binary else b,
                              lo=np.exp(b - 1.96 * se) if binary else b - 1.96 * se,
                              hi=np.exp(b + 1.96 * se) if binary else b + 1.96 * se,
                              p=stats.chi2.sf(lrt, 1)))
        bl = pd.DataFrame(block)
        bl["fdr"] = bh(bl.p)
        bl["studywide_sig"] = bl.p < STUDYWIDE
        rows.append(bl)
        print(f"{fam:12s} {y:20s} FDR<0.05 {(bl.fdr < 0.05).sum():2d}/15  study-wide {bl.studywide_sig.sum():2d}/15  "
              f"max effect {bl.effect.max():.3f}  min P {bl.p.min():.2e}")

pd.concat(rows).to_csv(out("phenotypic", "associations.tsv"), sep="\t", index=False)
