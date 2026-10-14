"""Retina BESD: every EyeGEx gene whose TSS lies in a probe window. SE = |beta| / |t_404(P/2)|;
A1 is the EyeGEx effect allele; rsID-named variants only."""
import io, os, subprocess
import numpy as np, pandas as pd
from scipy.stats import t as tdist
from common import E, OUT

W = f"{OUT}/smr/retina"
os.makedirs(f"{W}/esd", exist_ok=True)
SYM = pd.read_csv(f"{OUT}/ref/genes.tsv", sep="\t", header=None, usecols=[0, 1]).set_index(0)[1].to_dict()
P = pd.read_csv(f"{OUT}/smr/windows.csv").query("kind == 'probe'")
with open(f"{W}/retina.flist", "w") as fl:
    fl.write("Chr\tProbeID\tGeneticDistance\tProbeBp\tGene\tOrientation\tPathOfEsd\n")
    for c in sorted(P.chr.unique()):
        w = P[P.chr == c][["start", "end"]].values
        txt = subprocess.run(f"gzip -dc {E['EYEGEX']}/new_nominal.eQTL.chr{c}.with_thresholds.txt.gz", shell=True,
                             capture_output=True, text=True, check=True).stdout
        d = pd.read_csv(io.StringIO(txt), sep="\t", header=None, skiprows=1, dtype={7: str})
        d = d[[((w[:, 0] <= x) & (w[:, 1] >= x)).any() for x in d[2]]]
        d = d[d[7].str.startswith("rs")].copy()
        d["rsid"] = d[7].str.split(":").str[0]
        for g, x in d.groupby(0, sort=False):
            x = x.drop_duplicates("rsid")
            p = np.clip(x[13].astype(float).values, 1e-320, 1.0)
            b = x[14].astype(float).values
            e = pd.DataFrame({"Chr": c, "SNP": x.rsid.values, "Bp": x[9].astype(int).values, "A1": x[11].values,
                              "A2": x[12].values, "Freq": "NA", "Beta": b, "se": np.abs(b) / np.abs(tdist.ppf(p / 2, 404)), "p": p})
            e = e[np.isfinite(e.se) & (e.se > 0)]
            if e.empty:
                continue
            path = os.path.abspath(f"{W}/esd/{g}.esd")
            e.to_csv(path, sep="\t", index=False, float_format="%.6g")
            fl.write(f"{c}\t{g}\t0\t{int(x[2].iloc[0])}\t{SYM.get(g.split('.')[0], g)}\t{x[4].iloc[0]}\t{path}\n")
subprocess.run([E["SMR"], "--eqtl-flist", f"{W}/retina.flist", "--make-besd", "--out", f"{W}/retina_eyegex"], check=True)
