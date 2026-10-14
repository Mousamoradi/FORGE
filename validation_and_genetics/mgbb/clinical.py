"""POAG ~ z(axis) + age + sex + PC1-PC10 in unrelated European MGBB participants with POAG status;
logistic regression, odds ratio per SD, Benjamini-Hochberg across the 15 axes."""
import numpy as np, pandas as pd, statsmodels.api as sm
from common import E, FCM, OUT, bh

W = f"{OUT}/mgbb"
ph = pd.read_csv(f"{W}/pheno_rint.tsv", sep="\t", dtype={"IID": str})
for a in E.get("MGBB_FLIP", "").split(","):
    if a:
        ph[a] = -ph[a]
cov = pd.read_csv(f"{W}/covar_full.tsv", sep="\t", dtype={"IID": str})
keep = set(pd.read_csv(f"{W}/unrelated.king.cutoff.in.id", sep="\t", dtype=str).iloc[:, -1])
st = pd.read_csv(E["MGBB_POAG"], dtype=str)
st = st.assign(IID=st.iloc[:, 0].str.strip('" '), g=st.iloc[:, 4].str.strip('" ').str.lower())
st = st[st.g.isin(["case", "control"])].assign(y=lambda s: (s.g == "case").astype(int))[["IID", "y"]]

d = ph.merge(cov, on="IID").merge(st, on="IID")
d = d[d.IID.isin(keep)]
C = ["age", "sex"] + [f"PC{i}" for i in range(1, 11)]
rows = []
for a in FCM:
    z = (d[a] - d[a].mean()) / d[a].std(ddof=0)
    m = sm.Logit(d.y, sm.add_constant(pd.concat([z.rename(a), d[C]], axis=1).astype(float))).fit(method="newton", maxiter=200, disp=0)
    b, se = m.params[a], m.bse[a]
    rows.append(dict(axis=a, OR=np.exp(b), lo=np.exp(b - 1.96 * se), hi=np.exp(b + 1.96 * se), p=m.pvalues[a]))
R = pd.DataFrame(rows)
R["fdr"] = bh(R.p)
R.to_csv(f"{W}/clinical_poag.tsv", sep="\t", index=False)
top = R.loc[(np.log(R.OR)).abs().idxmax()]
print(f"n = {len(d)} ({d.y.sum()} cases, {(1 - d.y).sum()} controls); FDR < 0.05: {(R.fdr < 0.05).sum()}/15; "
      f"largest OR {top.OR:.2f} ({top.lo:.2f}-{top.hi:.2f}) for {top.axis}")
