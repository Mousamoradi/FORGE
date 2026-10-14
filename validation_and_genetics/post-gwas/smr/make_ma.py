import numpy as np, pandas as pd
from common import FCM, OUT, out, read_sumstats

win = pd.read_csv(f"{OUT}/smr/windows.csv").query("kind == 'variant'")[["chr", "start", "end"]].values
for t in FCM:
    d = read_sumstats(t)
    m = np.zeros(len(d), bool)
    for c, lo, hi in win:
        m |= (d.CHR.values == c) & (d.POS.values >= lo) & (d.POS.values <= hi)
    o = d[m][["SNP", "A1", "A2", "FREQ", "BETA", "SE", "P", "N"]]
    o.columns = ["SNP", "A1", "A2", "freq", "b", "se", "p", "n"]
    o.to_csv(out("smr", "ma", f"{t}.ma"), sep=" ", index=False)
    print(t, len(o), flush=True)
