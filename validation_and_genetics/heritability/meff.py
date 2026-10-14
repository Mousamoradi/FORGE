"""Li-Ji effective number of endophenotypes from the phenotypic Pearson matrix.
usage: meff.py [table.tsv]   (default UKB_OUTCOMES; any table with fcm_z1..fcm_z15)"""
import sys
import numpy as np, pandas as pd
from common import E, FCM


def li_ji(R):
    ev = np.clip(np.linalg.eigvalsh(np.asarray(R, float)), 0, None)
    return float(np.sum((ev >= 1) + (ev - np.floor(ev))))


if __name__ == "__main__":
    d = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else E["UKB_OUTCOMES"], sep="\t", usecols=FCM)
    print(f"N = {len(d)}  M_eff (Li-Ji) = {li_ji(d.corr()):.2f}")
