"""VF-guided (15 axes) versus No-VF (20 axes): LDSC h2, h2 Z and mean chi2, Mann-Whitney U."""
import glob, os, re
import pandas as pd
from scipy.stats import mannwhitneyu
from common import OUT


def parse(f):
    t = open(f).read()
    h2, se = map(float, re.search(r"Total Observed scale h2: (\S+) \((\S+)\)", t).groups())
    chi = float(re.search(r"Mean Chi\^2: (\S+)", t).group(1))
    return dict(trait=os.path.basename(f)[:-4], h2=h2, se=se, z=h2 / se, meanchi2=chi)


h = pd.DataFrame([parse(f) for f in glob.glob(f"{OUT}/ldsc/h2/*.log")])
vf, nv = h[h.trait.str.startswith("fcm")], h[h.trait.str.startswith("wovf")]
top = nv.nlargest(15, "z")
mw = lambda a, b: mannwhitneyu(a, b, alternative="two-sided").pvalue
print(f"h2 Z median {vf.z.median():.2f} vs {nv.z.median():.2f}  P = {mw(vf.z, nv.z):.2g}")
print(f"h2 median {vf.h2.median():.3f} vs {nv.h2.median():.3f}  P = {mw(vf.h2, nv.h2):.2g}")
print(f"mean chi2 - 1 median ratio {(vf.meanchi2.median() - 1) / (nv.meanchi2.median() - 1):.2f}  "
      f"P = {mw(vf.meanchi2, nv.meanchi2):.2g}")
print(f"top-15 No-VF h2 Z median {top.z.median():.2f}  P = {mw(vf.z, top.z):.2g}")
print(f"Z >= 4: {(vf.z >= 4).sum()}/{len(vf)} vs {(nv.z >= 4).sum()}/{len(nv)}")
