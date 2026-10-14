"""Re-key each trait's summary statistics from rsID to GTEx varID, matching on rsID and unordered allele
pair against the MASHR model weights (the MASHR covariances are varID-keyed)."""
import glob, sqlite3
from common import E, FCM, out, read_sumstats

m = {}
for db in sorted(glob.glob(f"{E['MASHR_DIR']}/*/*.db")):
    con = sqlite3.connect(db)
    for rs, vid, ref, eff in con.execute("SELECT DISTINCT rsid, varID, ref_allele, eff_allele FROM weights"):
        m.setdefault((rs, *sorted((ref, eff))), vid)
    con.close()
print(len(m), "model variants")

for t in FCM:
    d = read_sumstats(t, usecols=["SNP", "A1", "A2", "BETA", "SE", "P", "N", "FREQ"])
    k = [(s, *sorted((a, b))) for s, a, b in zip(d.SNP, d.A1, d.A2)]
    d["SNP"] = [m.get(x) for x in k]
    d = d.dropna(subset=["SNP"])
    d.columns = ["SNP", "effect_allele", "non_effect_allele", "beta", "se", "pvalue", "N", "freq"]
    d.to_csv(out("twas", "gwas", f"{t}.txt.gz"), sep="\t", index=False)
    print(t, len(d), flush=True)
