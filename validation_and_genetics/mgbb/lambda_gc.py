import glob
import numpy as np, pandas as pd
from common import OUT

lam = {}
for f in sorted(glob.glob(f"{OUT}/mgbb/glm/mgbb.fcm_z*.glm.linear*")):
    t = pd.read_csv(f, sep="\t", usecols=["T_STAT"]).T_STAT.dropna()
    lam[f.split("mgbb.")[-1].split(".glm")[0]] = np.median(t ** 2) / 0.4549364
s = pd.Series(lam)
print(s.round(4).to_string())
print(f"median {s.median():.3f}, range {s.min():.3f}-{s.max():.3f}")
