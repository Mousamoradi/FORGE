"""MGBB analysis inputs: imaged participants linked to genotypes, European subset, rank-inverse-normal
latents (within the subset), sex, age and spherical equivalent (mean over eyes and visits; missing
values mean-imputed with an indicator)."""
from collections import defaultdict
import numpy as np, pandas as pd
from scipy.stats import norm, rankdata
from common import E, FCM, out

xw = pd.read_csv(E["MGBB_XWALK"], dtype=str)
fam = set(pd.read_csv(E["MGBB_BFILE"] + ".fam", sep=r"\s+", header=None, dtype=str)[1])
eur = set(open(E["MGBB_EUR"]).read().split())
lat = pd.read_csv(E["MGBB_LATENTS"], dtype={"patient_id": str})
lat = lat.rename(columns={f"z{i}": f"fcm_z{i}" for i in range(1, 16)})

link = defaultdict(set)
for bb, mrn in zip(xw.iloc[:, 0], xw.iloc[:, 3]):
    if isinstance(mrn, str) and mrn != "NA":
        link[mrn.strip()].add(bb.strip())
lat["IID"] = [next((b for b in link.get(p.strip(), ()) if b in fam), "") for p in lat.patient_id]
lat = lat[lat.IID.isin(eur)].copy()
for a in FCM:
    lat[a] = norm.ppf((rankdata(lat[a], method="average") - 0.5) / len(lat))

cov = pd.read_csv(E["MGBB_COVAR"], sep="\t", dtype={"IID": str})[["IID", "sex", "Age"]].rename(columns={"Age": "age"})
ref = pd.read_csv(E["MGBB_REFRACTION"], dtype=str)
se = defaultdict(list)
for bb, od, os_ in zip(ref["BioBank_Subject_Id"], ref["Spherical_Equivalent.od"], ref["Spherical_Equivalent.os"]):
    for v in (od, os_):
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f):
            se[bb.strip().strip('"')].append(f)
se = {k: np.mean(v) for k, v in se.items() if v}
lat = lat.merge(cov, on="IID", how="left")
lat["SE_NA"] = (~lat.IID.isin(se)).astype(int)
lat["SE"] = [f"{v:.4f}" for v in lat.IID.map(se).fillna(np.mean([se[i] for i in lat.IID if i in se]))]
lat["FID"] = lat.IID

lat[["FID", "IID"]].to_csv(out("mgbb", "sample_eur.keep"), sep="\t", header=False, index=False)
lat[["FID", "IID"] + FCM].to_csv(out("mgbb", "pheno_rint.tsv"), sep="\t", index=False, float_format="%.6f")
lat[["FID", "IID", "sex", "age", "SE", "SE_NA"]].to_csv(out("mgbb", "covar_base.tsv"), sep="\t", index=False)
print(len(lat), "European participants with latents and genotypes")
