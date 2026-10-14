import os
import pandas as pd

E = os.environ
OUT = E.get("OUT", "out")
FCM = [f"fcm_z{i}" for i in range(1, 16)]
NOVF = [f"wovf_z{i}" for i in range(1, 21)]
N_GWAS = 39381
MEFF = 5

SSF = {"rsid": "SNP", "chromosome": "CHR", "base_pair_location": "POS",
       "effect_allele": "A1", "other_allele": "A2", "effect_allele_frequency": "FREQ",
       "n": "N", "beta": "BETA", "standard_error": "SE", "p_value": "P"}


def traits(prefix="fcm"):
    return FCM if prefix == "fcm" else NOVF


def sumstats_path(trait, root=None):
    return os.path.join(root or E["SUMSTATS"], f"{trait}.tsv.gz")


def read_sumstats(trait, root=None, usecols=None):
    d = pd.read_csv(sumstats_path(trait, root), sep="\t", dtype={"chromosome": str})
    d = d.rename(columns=SSF)
    d = d[d.CHR.str.fullmatch(r"\d+")].copy()
    d["CHR"] = d.CHR.astype(int)
    alt = d["variant_id"] if "variant_id" in d else (d.CHR.astype(str) + ":" + d.POS.astype(str) + ":" + d.A2 + ":" + d.A1)
    d["SNP"] = d["SNP"].fillna(alt) if "SNP" in d else alt
    return d[usecols] if usecols else d


def out(*parts):
    p = os.path.join(OUT, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def bh(p):
    import numpy as np
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    r = np.empty(n)
    r[o] = np.minimum.accumulate((p[o] * n / (np.arange(n) + 1))[::-1])[::-1]
    return np.clip(r, 0, 1)
