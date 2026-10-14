"""Incremental discrimination: four cumulative ridge-logistic models, nested 5-fold CV, five outer seeds.
AUC_TABLE: complete-case analysis set with age, sex, IOPg, RNFL, PRS, fcm_z1..fcm_z15 and the outcomes."""
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from common import E, FCM, out

OUTCOMES = ["poag_icd", "glaucoma_any"]
SEEDS = [42, 7, 2024, 101, 999]
GRID = {"clf__C": np.logspace(-3, 2, 12)}
B = 2000
BASE = ["age", "sex"]
MODELS = {"M1": BASE, "M2": BASE + ["IOPg", "RNFL"], "M3": BASE + ["IOPg", "RNFL", "PRS"],
          "M4": BASE + ["IOPg", "RNFL", "PRS"] + FCM,
          "M4_permuted": BASE + ["IOPg", "RNFL", "PRS"] + [f"perm_{a}" for a in FCM]}


def oof(X, y, seed):
    pred = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        pipe = Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(penalty="l2", solver="lbfgs", max_iter=4000))])
        gs = GridSearchCV(pipe, GRID, scoring="roc_auc", n_jobs=-1,
                          cv=StratifiedKFold(5, shuffle=True, random_state=seed + 1)).fit(X[tr], y[tr])
        pred[te] = gs.best_estimator_.predict_proba(X[te])[:, 1]
    return pred


def delong(y, p1, p2):
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    V = []
    for p in (p1, p2):
        x, z = p[pos], p[neg]
        psi = (x[:, None] > z[None, :]) + 0.5 * (x[:, None] == z[None, :])
        V.append((psi.mean(1), psi.mean(0)))
    v10 = np.cov(np.vstack([V[0][0], V[1][0]]))
    v01 = np.cov(np.vstack([V[0][1], V[1][1]]))
    S = v10 / m + v01 / n
    return np.sqrt(S[0, 0] + S[1, 1] - 2 * S[0, 1])


def boot_se(y, p, rng):
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    a = [roc_auc_score(y[s], p[s]) for s in
         (np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))]) for _ in range(B))]
    return np.std(a, ddof=1)


d = pd.read_csv(E["AUC_TABLE"], sep="\t")
d = d.dropna(subset=BASE + ["IOPg", "RNFL", "PRS"] + FCM)
perm = np.random.default_rng(0).permutation(len(d))
for a in FCM:
    d[f"perm_{a}"] = d[a].values[perm]

auc_rows, test_rows = [], []
for y in OUTCOMES:
    s = d.dropna(subset=[y])
    yy = s[y].astype(int).values
    preds = {k: [] for k in MODELS}
    for seed in SEEDS:
        for k, f in MODELS.items():
            preds[k].append(oof(s[f].values.astype(float), yy, seed))
    rng = np.random.default_rng(1)
    for k in MODELS:
        a = np.array([roc_auc_score(yy, p) for p in preds[k]])
        se_b = np.mean([boot_se(yy, p, rng) for p in preds[k]])
        se = np.hypot(se_b, a.std(ddof=1))
        auc_rows.append(dict(outcome=y, model=k, n=len(s), n_case=int(yy.sum()), auc=a.mean(),
                             lo=a.mean() - 1.96 * se, hi=a.mean() + 1.96 * se,
                             se_boot=se_b, se_seed=a.std(ddof=1), se_comb=se))
    for ref in ("M1", "M2", "M3"):
        dl = np.array([roc_auc_score(yy, p4) - roc_auc_score(yy, p0) for p4, p0 in zip(preds["M4"], preds[ref])])
        se_d = np.mean([delong(yy, p4, p0) for p4, p0 in zip(preds["M4"], preds[ref])])
        se = np.hypot(se_d, dl.std(ddof=1))
        z = dl.mean() / se
        test_rows.append(dict(outcome=y, comparison=f"M4 vs {ref}", delta_auc=dl.mean(),
                              lo=dl.mean() - 1.96 * se, hi=dl.mean() + 1.96 * se, se_delong=se_d,
                              se_seed=dl.std(ddof=1), se_comb=se, z=z, p_value=2 * stats.norm.sf(abs(z))))

A, T = pd.DataFrame(auc_rows), pd.DataFrame(test_rows)
A.to_csv(out("phenotypic", "auc_models.tsv"), sep="\t", index=False)
T.to_csv(out("phenotypic", "auc_tests.tsv"), sep="\t", index=False)
print(A.round(4).to_string(index=False))
print(T.round(4).to_string(index=False))
