"""
FORGE — shared phenotyping toolkit (stage 2 of the pipeline, Fig. 1A/1B).

Everything that the FCM and GMM runners and the external-cohort inference
script have in common lives here:

  * beta-VAE (StandardVAE) + training loop
  * Fuzzy c-means (FuzzyCMeans) with saved-centroid prediction
  * Distribution-diversity analysis (peaks / entropy / KL / bimodality / diversity score)
  * Clustering-quality metrics and the manuscript composite score
        composite = mean( Silhouette,
                          Calinski-Harabasz normalised to [0,1],
                          inverted Davies-Bouldin,
                          Diversity score )
  * Latent-dimension x cluster-number grid search (FCM or GMM)
  * Low-membership reassignment
  * Normalised cross-correlation (NCC) of representative RNFL maps per cluster
  * Figures (latent scatter, metric bars, d x k composite grid, NCC panels)
  * Transfer-artifact I/O (scaler, VAE, cluster model) shared with inference

No cohort-specific paths live here; see the run_* scripts and
inference/transfer_inference.py for configuration.
"""

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    import cv2
except ImportError:  # NCC falls back to PIL
    cv2 = None
from PIL import Image

warnings.filterwarnings('ignore')

CONDITIONS = ('with_vf', 'without_vf')


# ==================== beta-VAE ====================

class StandardVAE(nn.Module):
    """Fully connected beta-VAE used to compress 256-D encoder embeddings
    into a d-dimensional latent space. The posterior mean mu is the latent
    representation used everywhere downstream (deterministic)."""

    def __init__(self, input_dim, latent_dim, hidden_dims=(256, 128, 64, 32), dropout_rate=0.1):
        super().__init__()
        hidden_dims = list(hidden_dims)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims

        enc, prev = [], input_dim
        for h in hidden_dims:
            enc += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout_rate)]
            prev = h
        self.encoder = nn.Sequential(*enc)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        dec, prev = [], latent_dim
        for h in reversed(hidden_dims):
            dec += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout_rate)]
            prev = h
        dec.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

    @torch.no_grad()
    def get_latent_embedding(self, x):
        mu, _ = self.encode(x)
        return mu


def vae_loss(x_recon, x, mu, logvar, beta):
    recon = F.mse_loss(x_recon, x, reduction='mean')
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl


def train_vae(model, train_loader, val_loader, *, epochs, lr, beta, patience, device, verbose=True):
    """Adam + ReduceLROnPlateau, early stopping on validation loss."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    best_val, best_state, wait = float('inf'), None, 0

    for epoch in range(epochs):
        model.train()
        for (bx,) in train_loader:
            bx = bx.to(device)
            optimizer.zero_grad()
            xr, mu, lv, _ = model(bx)
            loss = vae_loss(xr, bx, mu, lv, beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (bx,) in val_loader:
                bx = bx.to(device)
                xr, mu, lv, _ = model(bx)
                val_loss += vae_loss(xr, bx, mu, lv, beta).item()
        val_loss /= max(len(val_loader), 1)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val, wait = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if verbose and (epoch + 1) % 50 == 0:
            print(f"      epoch {epoch + 1}/{epochs}  val_loss={val_loss:.4f}")
        if wait >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def encode_latents(model, data_scaled, device, batch_size=256):
    """Posterior means (mu) for an array of standardised embeddings."""
    model.eval()
    out = []
    for i in range(0, len(data_scaled), batch_size):
        bx = torch.as_tensor(data_scaled[i:i + batch_size], dtype=torch.float32).to(device)
        out.append(model.get_latent_embedding(bx).cpu().numpy())
    return np.concatenate(out, axis=0)


def fit_vae_for_dim(data, latent_dim, cfg, seed=42):
    """Standardise `data` (N x 256), train a beta-VAE with `latent_dim`, and
    return (scaler, model, latents) where latents are the posterior means.

    An 80/20 split of the rows is used only for early stopping."""
    rng = np.random.RandomState(seed)
    scaler = StandardScaler().fit(data)
    data_s = scaler.transform(data).astype(np.float32)

    idx = rng.permutation(len(data_s))
    n_train = int(0.8 * len(idx))
    tr = DataLoader(TensorDataset(torch.from_numpy(data_s[idx[:n_train]])),
                    batch_size=cfg.batch_size, shuffle=True)
    va = DataLoader(TensorDataset(torch.from_numpy(data_s[idx[n_train:]])),
                    batch_size=cfg.batch_size)

    torch.manual_seed(seed)
    model = StandardVAE(data.shape[1], latent_dim, cfg.hidden_dims, cfg.dropout_rate).to(cfg.device)
    model = train_vae(model, tr, va, epochs=cfg.epochs, lr=cfg.learning_rate, beta=cfg.beta,
                      patience=cfg.early_stopping_patience, device=cfg.device, verbose=cfg.verbose)
    latents = encode_latents(model, data_s, cfg.device)
    return scaler, model, latents


# ==================== FUZZY C-MEANS ====================

class FuzzyCMeans:
    """Bezdek fuzzy c-means. `predict_proba` works from saved centroids, so
    external cohorts can be assigned against frozen MEE centers."""

    def __init__(self, n_clusters=8, m=1.5, max_iter=150, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.u = None  # (N, K) memberships of the training data

    @classmethod
    def from_centers(cls, centers, m=1.5):
        obj = cls(n_clusters=len(centers), m=m)
        obj.cluster_centers_ = np.asarray(centers, dtype=np.float64)
        return obj

    def _membership(self, X, centers):
        d = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        d = np.fmax(d, np.finfo(float).eps)
        u = 1.0 / d ** (2.0 / (self.m - 1))
        return u / u.sum(axis=1, keepdims=True)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.RandomState(self.random_state)
        u = rng.rand(len(X), self.n_clusters)
        u /= u.sum(axis=1, keepdims=True)
        for _ in range(self.max_iter):
            um = u ** self.m
            centers = (um.T @ X) / um.sum(axis=0)[:, None]
            u_new = self._membership(X, centers)
            if np.linalg.norm(u_new - u) < self.tol:
                u = u_new
                break
            u = u_new
        self.cluster_centers_ = centers
        self.u = u
        return self

    def predict_proba(self, X):
        return self._membership(np.asarray(X, dtype=np.float64), self.cluster_centers_)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def fit_predict(self, X):
        return self.fit(X).predict(X)

    def partition_coefficient(self):
        return float((self.u ** 2).sum() / len(self.u))

    def partition_entropy(self):
        return float(-(self.u * np.log(self.u + 1e-10)).sum() / len(self.u))


# ==================== DISTRIBUTION DIVERSITY ====================

def analyze_distribution_diversity(latents, n_bins=50):
    """Per-dimension histogram peaks, entropy, KL(N(mu,s^2)||N(0,1)),
    bimodality, and the composite diversity score used in Table 2."""
    n, d = latents.shape
    per_dim = {k: [] for k in ['peaks_per_dim', 'peak_locations', 'peak_separations',
                               'distribution_entropy', 'kl_divergence', 'kurtosis',
                               'bimodality_coefficient']}
    all_peaks = []
    for j in range(d):
        x = latents[:, j]
        hist, edges = np.histogram(x, bins=n_bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        peaks, _ = find_peaks(hist, prominence=0.01, distance=3)
        per_dim['peaks_per_dim'].append(len(peaks))
        if len(peaks):
            locs = centers[peaks]
            per_dim['peak_locations'].append(locs)
            all_peaks.extend(locs)
            per_dim['peak_separations'].append(float(np.mean(np.diff(np.sort(locs)))) if len(locs) > 1 else 0.0)
        else:
            per_dim['peak_locations'].append(np.array([]))
            per_dim['peak_separations'].append(0.0)

        p = hist / (hist.sum() + 1e-10)
        p = p[p > 0]
        per_dim['distribution_entropy'].append(float(-np.sum(p * np.log(p + 1e-10))))

        kurt, skew = stats.kurtosis(x), stats.skew(x)
        per_dim['kurtosis'].append(float(kurt))
        per_dim['bimodality_coefficient'].append(
            float((skew ** 2 + 1) / (kurt + 3 * ((n - 1) ** 2 / ((n - 2) * (n - 3))))))

        mu, sd = float(np.mean(x)), float(np.std(x))
        per_dim['kl_divergence'].append(float(0.5 * (mu ** 2 + sd ** 2 - 1.0 - np.log(sd ** 2 + 1e-10))))

    peaks_arr = np.array(per_dim['peaks_per_dim'])
    seps = [s for s in per_dim['peak_separations'] if s > 0]
    summary = {
        'latent_dim': d,
        'mean_kl_divergence': float(np.mean(per_dim['kl_divergence'])),
        'total_kl_divergence': float(np.sum(per_dim['kl_divergence'])),
        'total_peaks': int(peaks_arr.sum()),
        'avg_peaks_per_dim': float(peaks_arr.mean()),
        'peak_location_diversity': float(np.std(all_peaks)) if all_peaks else 0.0,
        'avg_peak_separation': float(np.mean(seps)) if seps else 0.0,
        'mean_entropy': float(np.mean(per_dim['distribution_entropy'])),
        'mean_bimodality': float(np.mean(per_dim['bimodality_coefficient'])),
        'multimodal_ratio': float((peaks_arr >= 2).sum() / d),
    }
    summary['diversity_score'] = float(
        0.25 * min(summary['avg_peaks_per_dim'] / 3, 1) +
        0.25 * min(summary['avg_peak_separation'] / 2, 1) +
        0.25 * min(summary['mean_entropy'] / 3, 1) +
        0.25 * summary['multimodal_ratio'])
    return summary, per_dim


# ==================== CLUSTER METRICS ====================

def normalise_ch(ch):
    return float(min(ch / 1000.0, 1.0))


def invert_db(db):
    return float(1.0 - min(db / 10.0, 1.0))


def clustering_metrics(latents, labels, diversity_score):
    """Silhouette, CH, DB (raw and normalised) and the manuscript composite."""
    m = {}
    if len(np.unique(labels)) > 1:
        m['silhouette'] = float(silhouette_score(latents, labels))
        m['calinski_harabasz'] = float(calinski_harabasz_score(latents, labels))
        m['davies_bouldin'] = float(davies_bouldin_score(latents, labels))
    else:
        m['silhouette'], m['calinski_harabasz'], m['davies_bouldin'] = -1.0, 0.0, float('inf')
    m['calinski_harabasz_norm'] = normalise_ch(m['calinski_harabasz'])
    m['davies_bouldin_inv'] = invert_db(m['davies_bouldin'])
    m['diversity_score'] = float(diversity_score)
    m['composite_score'] = float(np.mean([m['silhouette'], m['calinski_harabasz_norm'],
                                          m['davies_bouldin_inv'], m['diversity_score']]))
    return m


def fit_cluster_model(method, latents, k, *, m=1.5, covariance_type='full', seed=42):
    """Fit FCM or GMM on VAE latents. Returns (model, labels, proba (N x K))."""
    if method == 'fcm':
        model = FuzzyCMeans(n_clusters=k, m=m, random_state=seed).fit(latents)
        proba = model.u
    elif method == 'gmm':
        model = GaussianMixture(n_components=k, covariance_type=covariance_type,
                                random_state=seed, n_init=10).fit(latents)
        proba = model.predict_proba(latents)
    else:
        raise ValueError(method)
    return model, np.argmax(proba, axis=1), proba


def predict_cluster_model(model, latents):
    """Assign new latents with a fitted FCM / GMM. Returns (labels, proba)."""
    proba = model.predict_proba(latents)
    return np.argmax(proba, axis=1), proba


# ==================== GRID SEARCH ====================

def grid_search(data, method, cfg, condition, out_dir, seed=42):
    """Sweep latent dimension x cluster number (x fuzziness m or covariance).

    For each latent dim one beta-VAE is trained; every cluster configuration is
    then evaluated on its posterior-mean latents with the composite score.
    Returns a dict with per-dim artifacts and a results DataFrame."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, per_dim = [], {}

    print(f"\n{'=' * 70}\nGRID SEARCH [{method.upper()}] — {condition}\n{'=' * 70}")
    for d in cfg.latent_dims:
        print(f"\n  latent dim = {d}")
        scaler, vae, latents = fit_vae_for_dim(data, d, cfg, seed=seed)
        div_summary, div_detail = analyze_distribution_diversity(latents, cfg.n_histogram_bins)
        print(f"    diversity score {div_summary['diversity_score']:.4f} | "
              f"mean KL {div_summary['mean_kl_divergence']:.4f} | "
              f"peak-location diversity {div_summary['peak_location_diversity']:.3f}")
        per_dim[d] = {'scaler': scaler, 'vae': vae, 'latents': latents,
                      'diversity': div_summary, 'diversity_detail': div_detail}

        variants = ([('m', m) for m in cfg.fuzziness_m_range] if method == 'fcm'
                    else [('covariance_type', c) for c in cfg.covariance_types])
        best_here = None
        for k in cfg.n_clusters_range:
            for key, val in variants:
                model, labels, _ = fit_cluster_model(method, latents, k, seed=seed, **{key: val})
                met = clustering_metrics(latents, labels, div_summary['diversity_score'])
                row = {'latent_dim': d, 'n_clusters': k, key: val, **met,
                       'mean_kl_divergence': div_summary['mean_kl_divergence'],
                       'mean_entropy': div_summary['mean_entropy'],
                       'peak_location_diversity': div_summary['peak_location_diversity']}
                if method == 'fcm':
                    row['partition_coefficient'] = model.partition_coefficient()
                    row['partition_entropy'] = model.partition_entropy()
                else:
                    row['bic'] = float(model.bic(latents))
                    row['aic'] = float(model.aic(latents))
                rows.append(row)
                if best_here is None or met['composite_score'] > best_here[0]:
                    best_here = (met['composite_score'], k, val)
        print(f"    best at {d}D: k={best_here[1]} ({best_here[2]}) composite={best_here[0]:.4f}")

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / f'grid_search_{method}_{condition}.csv', index=False)
    pd.DataFrame([per_dim[d]['diversity'] for d in cfg.latent_dims]).to_csv(
        out_dir / f'diversity_{method}_{condition}.csv', index=False)
    best = results.loc[results['composite_score'].idxmax()]
    print(f"\n  >> grid optimum: {int(best['latent_dim'])}D, k={int(best['n_clusters'])}, "
          f"composite={best['composite_score']:.4f}")
    return {'results': results, 'per_dim': per_dim, 'grid_best': best}


def final_fit(search, method, spec, condition, cfg, out_dir, seed=42):
    """Fit the final cluster model at `spec` (dict with latent_dim, n_clusters and
    m / covariance_type) on that dim's latents, compute Table-2 metrics, and
    save the transfer artifacts. Returns a dict with latents, labels, proba, metrics."""
    out_dir = Path(out_dir)
    d, k = int(spec['latent_dim']), int(spec['n_clusters'])
    if d not in search['per_dim']:
        raise ValueError(f"latent_dim {d} was not part of the grid {list(search['per_dim'])}")
    entry = search['per_dim'][d]
    latents = entry['latents']
    extra = ({'m': spec.get('m', 1.5)} if method == 'fcm'
             else {'covariance_type': spec.get('covariance_type', 'full')})
    model, labels, proba = fit_cluster_model(method, latents, k, seed=seed, **extra)
    metrics = clustering_metrics(latents, labels, entry['diversity']['diversity_score'])
    metrics.update({'latent_dim': d, 'n_clusters': k, **extra,
                    'n_samples': int(len(latents)),
                    'mean_kl_divergence': entry['diversity']['mean_kl_divergence'],
                    'mean_entropy': entry['diversity']['mean_entropy'],
                    'peak_location_diversity': entry['diversity']['peak_location_diversity']})

    print(f"\n  FINAL [{method.upper()} {condition}] {d}D, k={k} {extra}")
    for key in ['silhouette', 'calinski_harabasz_norm', 'davies_bouldin_inv',
                'mean_kl_divergence', 'mean_entropy', 'peak_location_diversity',
                'diversity_score', 'composite_score']:
        print(f"    {key:<26}{metrics[key]:.4f}")

    save_transfer_artifacts(out_dir, method, condition, entry['scaler'], entry['vae'],
                            model, latents, spec, metrics)
    return {'latents': latents, 'labels': labels, 'proba': proba, 'model': model,
            'metrics': metrics, 'scaler': entry['scaler'], 'vae': entry['vae']}


# ==================== ARTIFACT I/O ====================

def artifact_paths(root, method, condition):
    root = Path(root)
    return {
        'scaler': root / f'scaler_{method}_{condition}.pkl',
        'vae': root / f'vae_{method}_{condition}.pt',
        'cluster': root / f'{method}_model_{condition}.pkl',
        'spec': root / f'spec_{method}_{condition}.json',
        'latents_npy': root / f'latents_{method}_{condition}.npy',
        'latents_csv': root / f'latents_{method}_{condition}.csv',
        'metrics': root / f'metrics_{method}_{condition}.json',
    }


def save_transfer_artifacts(root, method, condition, scaler, vae, cluster_model, latents, spec, metrics):
    p = artifact_paths(root, method, condition)
    joblib.dump(scaler, p['scaler'])
    torch.save({'state_dict': vae.state_dict(), 'input_dim': vae.input_dim,
                'latent_dim': vae.latent_dim, 'hidden_dims': vae.hidden_dims}, p['vae'])
    joblib.dump(cluster_model, p['cluster'])
    with open(p['spec'], 'w') as f:
        json.dump({'method': method, 'condition': condition, **{k: (float(v) if isinstance(v, (np.floating,)) else v)
                                                                 for k, v in spec.items()}}, f, indent=2)
    with open(p['metrics'], 'w') as f:
        json.dump(metrics, f, indent=2)
    np.save(p['latents_npy'], latents)
    pd.DataFrame(latents, columns=[f'D{i + 1}' for i in range(latents.shape[1])]).to_csv(p['latents_csv'], index=False)
    print(f"    artifacts -> {root}")


def load_transfer_artifacts(root, method, condition, device='cpu'):
    """Frozen scaler + VAE + cluster model saved by `final_fit`."""
    p = artifact_paths(root, method, condition)
    for key in ['scaler', 'vae', 'cluster', 'spec']:
        if not p[key].exists():
            raise FileNotFoundError(p[key])
    scaler = joblib.load(p['scaler'])
    ck = torch.load(p['vae'], map_location=device, weights_only=False)
    vae = StandardVAE(ck['input_dim'], ck['latent_dim'], ck['hidden_dims']).to(device)
    vae.load_state_dict(ck['state_dict'])
    vae.eval()
    for q in vae.parameters():
        q.requires_grad_(False)
    cluster_model = joblib.load(p['cluster'])
    with open(p['spec']) as f:
        spec = json.load(f)
    return scaler, vae, cluster_model, spec


def project_to_mee_latent(embeddings, scaler, vae, device='cpu', batch_size=512):
    """Pure feature transfer: MEE scaler as a transform only, then frozen VAE mu."""
    return encode_latents(vae, scaler.transform(embeddings).astype(np.float32), device, batch_size)


# ==================== REASSIGNMENT ====================

def reassign_low_membership(proba, threshold=0.7):
    """Samples whose max membership < threshold are moved to their next-best
    cluster. Returns (labels, n_reassigned). proba is (N, K)."""
    labels = np.argmax(proba, axis=1)
    if threshold is None:
        return labels, 0
    low = np.where(proba.max(axis=1) < threshold)[0]
    n = 0
    for i in low:
        order = np.argsort(proba[i])[::-1]
        if len(order) > 1 and order[1] != labels[i]:
            labels[i] = order[1]
            n += 1
    return labels, n


# ==================== NCC ====================

def read_rgb(path):
    if not os.path.exists(path):
        return None
    if cv2 is not None:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        return np.asarray(Image.open(path).convert('RGB'))
    except Exception:
        return None


def _resize(img, hw):
    if cv2 is not None:
        return cv2.resize(img, (hw[1], hw[0]))
    return np.asarray(Image.fromarray(img.astype(np.uint8)).resize((hw[1], hw[0])))


def normalized_cross_correlation(a, b):
    """Channel-wise NCC averaged over RGB (colour preserved), clipped to [-1, 1]."""
    if a is None or b is None:
        return np.nan
    a, b = a.astype(np.float64), b.astype(np.float64)
    if a.shape != b.shape:
        hw = (min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1]))
        a, b = _resize(a, hw), _resize(b, hw)
    if a.ndim == 3:
        vals = []
        for c in range(a.shape[2]):
            x = (a[..., c] - a[..., c].mean()) / (a[..., c].std() + 1e-10)
            y = (b[..., c] - b[..., c].mean()) / (b[..., c].std() + 1e-10)
            vals.append(np.mean(x * y))
        ncc = float(np.mean(vals))
    else:
        x = (a - a.mean()) / (a.std() + 1e-10)
        y = (b - b.mean()) / (b.std() + 1e-10)
        ncc = float(np.mean(x * y))
    return float(np.clip(ncc, -1.0, 1.0))


def ncc_representatives(image_paths, proba, labels, *, n_candidates=20, n_show=5,
                        min_avg_ncc=0.08, alpha=0.5, beta=0.5):
    """For every cluster: take the `n_candidates` highest-membership scans, build
    their NCC matrix, drop candidates whose mean NCC to the others is below
    `min_avg_ncc`, rank the rest by alpha*membership + beta*mean-NCC (each
    min-max scaled) and keep the top `n_show`.

    Returns {cluster: {'indices', 'images', 'memberships', 'avg_ncc', 'ncc_matrix'}}
    and a summary DataFrame (per-cluster mean NCC and n)."""
    out, summary = {}, []
    for c in range(proba.shape[1]):
        members = np.where(labels == c)[0]
        summary_row = {'cluster': c, 'n_scans': int(len(members))}
        if len(members) < 2:
            summary.append({**summary_row, 'mean_ncc': np.nan, 'n_representatives': 0})
            continue
        order = members[np.argsort(proba[members, c])[::-1]]
        imgs, idxs = [], []
        for i in order:
            img = read_rgb(image_paths[i])
            if img is not None:
                imgs.append(img)
                idxs.append(i)
            if len(imgs) >= n_candidates:
                break
        if len(imgs) < 2:
            summary.append({**summary_row, 'mean_ncc': np.nan, 'n_representatives': 0})
            continue
        n = len(imgs)
        mat = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                mat[i, j] = mat[j, i] = normalized_cross_correlation(imgs[i], imgs[j])
        avg = np.array([np.mean(mat[i, np.arange(n) != i]) for i in range(n)])
        keep = np.where(avg >= min_avg_ncc)[0]
        if len(keep) == 0:
            summary.append({**summary_row, 'mean_ncc': float(np.mean(avg)), 'n_representatives': 0})
            continue
        mem = proba[np.array(idxs)[keep], c]
        nm = (mem - mem.min()) / (mem.max() - mem.min() + 1e-10)
        nn_ = (avg[keep] - avg[keep].min()) / (avg[keep].max() - avg[keep].min() + 1e-10)
        top = keep[np.argsort(alpha * nm + beta * nn_)[::-1][:n_show]]
        sub = mat[np.ix_(top, top)]
        out[c] = {'indices': [idxs[i] for i in top], 'images': [imgs[i] for i in top],
                  'memberships': proba[np.array(idxs)[top], c].tolist(),
                  'avg_ncc': avg[top].tolist(), 'ncc_matrix': sub}
        off = sub[~np.eye(len(top), dtype=bool)] if len(top) > 1 else np.array([np.nan])
        summary.append({**summary_row, 'mean_ncc': float(np.nanmean(off)), 'n_representatives': int(len(top))})
    return out, pd.DataFrame(summary)


def pooled_ncc(reps):
    vals = []
    for r in reps.values():
        n = r['ncc_matrix'].shape[0]
        if n > 1:
            vals.extend(r['ncc_matrix'][~np.eye(n, dtype=bool)])
    return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else (np.nan, np.nan, 0)


# ==================== FIGURES ====================

def plot_ncc_panels(reps, eye_labels, title, save_path, clusters_per_figure=4, dpi=200):
    """One row per cluster: NCC matrix + the representative RNFL maps."""
    ids = sorted(reps)
    if not ids:
        return
    for start in range(0, len(ids), clusters_per_figure):
        chunk = ids[start:start + clusters_per_figure]
        fig = plt.figure(figsize=(26, 5.2 * len(chunk)))
        outer = gridspec.GridSpec(len(chunk), 1, figure=fig, hspace=0.45)
        for r, c in enumerate(chunk):
            rep = reps[c]
            inner = gridspec.GridSpecFromSubplotSpec(1, 7, subplot_spec=outer[r, 0], wspace=0.08,
                                                     width_ratios=[1.8, 0.12, 1.2, 1.2, 1.2, 1.2, 1.2])
            ax = fig.add_subplot(inner[0, 0])
            mat = rep['ncc_matrix']
            im = ax.imshow(mat, cmap='RdBu_r', vmin=-1, vmax=1)
            n = mat.shape[0]
            ax.set_xticks(range(n)); ax.set_yticks(range(n))
            ax.set_xticklabels([f'R{i + 1}' for i in range(n)]); ax.set_yticklabels([f'R{i + 1}' for i in range(n)])
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center',
                            color='white' if abs(mat[i, j]) > 0.5 else 'black', fontsize=11)
            ax.set_title(f'Cluster {c + 1}\nNCC matrix ({n}x{n})', fontsize=16, fontweight='bold')
            cax = make_axes_locatable(ax).append_axes('right', size='6%', pad=0.08)
            fig.colorbar(im, cax=cax).set_label('NCC')
            for i, img in enumerate(rep['images'][:5]):
                a = fig.add_subplot(inner[0, 2 + i])
                a.imshow(img); a.axis('off')
                eye = eye_labels[rep['indices'][i]] if eye_labels is not None else ''
                a.set_title(f"R{i + 1} {eye}\nM {rep['memberships'][i]:.2f} | NCC {rep['avg_ncc'][i]:.2f}", fontsize=13)
        fig.suptitle(title, fontsize=18, y=0.995)
        fig.savefig(f"{save_path}_clusters{chunk[0] + 1}-{chunk[-1] + 1}.png", dpi=dpi, bbox_inches='tight')
        plt.close(fig)


def plot_latent_scatter(fits, method, save_path, dpi=200):
    """2-D projection (first two latent dims) with cluster colours and centroids;
    one panel per condition (Fig. 2A/B)."""
    fig, axes = plt.subplots(1, len(fits), figsize=(10 * len(fits), 7), squeeze=False)
    for ax, (cond, fit) in zip(axes[0], fits.items()):
        z, lab, k = fit['latents'], fit['labels'], fit['metrics']['n_clusters']
        cmap = plt.cm.tab10 if k <= 10 else plt.cm.tab20
        for c in range(k):
            msk = lab == c
            if msk.any():
                ax.scatter(z[msk, 0], z[msk, 1], s=28, alpha=0.75, color=cmap(c), edgecolors='black',
                           linewidths=0.3, label=f'C{c + 1}')
        cents = fit['model'].cluster_centers_ if method == 'fcm' else fit['model'].means_
        ax.scatter(cents[:, 0], cents[:, 1], marker='X', s=220, c='darkred', edgecolors='white',
                   linewidths=1.5, zorder=10, label='Centroids')
        ax.set_title(f"{method.upper()}: {cond.replace('_', ' ').upper()}", fontsize=18, fontweight='bold')
        ax.set_xlabel('Latent Dimension 1'); ax.set_ylabel('Latent Dimension 2')
        ax.text(0.02, 0.03, f"Latent = {fit['metrics']['latent_dim']}D", transform=ax.transAxes,
                bbox=dict(facecolor='white', edgecolor='black', alpha=0.85))
        ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.01, 0.5))
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_metric_bars(fits, method, save_path, dpi=200):
    """Table-2 metrics side by side (Fig. 2C style) with % change in composite."""
    names = ['Composite\nScore', 'Silhouette', 'Calinski-\nHarabasz (norm)', 'Davies-\nBouldin (inv)', 'Diversity\nScore']
    keys = ['composite_score', 'silhouette', 'calinski_harabasz_norm', 'davies_bouldin_inv', 'diversity_score']
    fig, ax = plt.subplots(figsize=(13, 6))
    conds = list(fits)
    w = 0.8 / len(conds)
    x = np.arange(len(keys))
    for i, cond in enumerate(conds):
        vals = [fits[cond]['metrics'][k] for k in keys]
        bars = ax.bar(x + (i - (len(conds) - 1) / 2) * w, vals, w, edgecolor='black',
                      label=f"{method.upper()}: {cond.replace('_', ' ').upper()}")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9, rotation=90)
    if 'with_vf' in fits and 'without_vf' in fits:
        a, b = fits['with_vf']['metrics']['composite_score'], fits['without_vf']['metrics']['composite_score']
        ax.set_title(f"Composite score {100 * (a - b) / b:+.1f}% with VF guidance", fontsize=14)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.2); ax.set_ylabel('Score'); ax.legend(); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_composite_grid(results, method, condition, spec, save_path, dpi=200):
    """Latent dimension (x) vs cluster number (y) coloured by the best composite
    score over m / covariance (Fig. 1A). The pinned spec is marked in red."""
    best = results.loc[results.groupby(['latent_dim', 'n_clusters'])['composite_score'].idxmax()]
    grid = best.pivot(index='n_clusters', columns='latent_dim', values='composite_score')
    fig, ax = plt.subplots(figsize=(0.75 * len(grid.columns) + 3, 0.5 * len(grid.index) + 2))
    xs, ys = np.meshgrid(np.arange(len(grid.columns)), np.arange(len(grid.index)))
    sc = ax.scatter(xs, ys, c=grid.values, s=260, cmap='RdYlBu_r', vmin=0, vmax=1, edgecolors='none')
    if spec is not None and spec['latent_dim'] in grid.columns and spec['n_clusters'] in grid.index:
        ax.scatter(list(grid.columns).index(spec['latent_dim']), list(grid.index).index(spec['n_clusters']),
                   s=320, facecolors='red', edgecolors='black', zorder=5)
        v = grid.loc[spec['n_clusters'], spec['latent_dim']]
        ax.annotate(f'{v:.3f}', (list(grid.columns).index(spec['latent_dim']), list(grid.index).index(spec['n_clusters'])),
                    xytext=(-40, -30), textcoords='offset points', fontsize=16, fontweight='bold', color='navy',
                    arrowprops=dict(arrowstyle='->', lw=2, color='navy'))
    ax.set_xticks(range(len(grid.columns))); ax.set_xticklabels(grid.columns)
    ax.set_yticks(range(len(grid.index))); ax.set_yticklabels(grid.index)
    ax.invert_yaxis()
    ax.set_xlabel('Latent Dimension (d)'); ax.set_ylabel('N of clusters (k)')
    ax.set_title(f"{method.upper()}: {condition.replace('_', ' ').upper()}", fontsize=14, fontweight='bold')
    fig.colorbar(sc, ax=ax).set_label('Composite Score')
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_latent_histograms(per_dim, condition, method, save_path, max_dims=16, dpi=200):
    """Density of every latent dimension at the final spec (Fig. 1C 'D1..D15' row)."""
    latents = per_dim['latents']
    d = min(latents.shape[1], max_dims)
    ncol = 4
    nrow = int(np.ceil(d / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow), squeeze=False)
    for j in range(nrow * ncol):
        ax = axes[j // ncol, j % ncol]
        if j >= d:
            ax.set_visible(False)
            continue
        x = latents[:, j]
        ax.hist(x, bins=40, density=True, color='steelblue', alpha=0.8)
        ax.set_title(f'D{j + 1}', fontsize=11)
        ax.set_yticks([])
    fig.suptitle(f"{method.upper()} {condition.replace('_', ' ').upper()} — {latents.shape[1]}D latent densities", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


# ==================== METADATA HELPERS ====================

def rnfl_image_paths(metadata, image_dir, ext='.jpg'):
    """Reconstruct `ID_EYE_DATE_INDEX.jpg` paths from the training metadata CSV."""
    stems = metadata[['ID', 'Eye', 'Date', 'Index']].astype(str).agg('_'.join, axis=1)
    return [os.path.join(image_dir, s + ext) for s in stems]


def load_condition_inputs(cfg):
    """Load the two embedding sets written by training/train_forge.py together
    with their row-aligned metadata. Optionally restrict to one CV fold's
    held-out scans (`cfg.fold_filter`). Returns (data dict, metadata)."""
    meta = pd.read_csv(cfg.metadata_csv)
    data = {'with_vf': np.load(cfg.with_vf_embeddings),
            'without_vf': np.load(cfg.without_vf_embeddings)}
    for k, v in data.items():
        if len(v) != len(meta):
            raise ValueError(f"{k}: {len(v)} embeddings vs {len(meta)} metadata rows — "
                             "the .npy and .csv must come from the same training run")
    if getattr(cfg, 'fold_filter', None) is not None:
        keep = (meta['fold'] == cfg.fold_filter).values
        meta = meta[keep].reset_index(drop=True)
        data = {k: v[keep] for k, v in data.items()}
        print(f"Restricted to held-out scans of fold {cfg.fold_filter}: n = {len(meta)}")
    print(f"WITH VF    embeddings: {data['with_vf'].shape}")
    print(f"WITHOUT VF embeddings: {data['without_vf'].shape}")
    return data, meta


# ==================== END-TO-END PIPELINE (shared by run_vae_fcm / run_vae_gmm) ====================

def run_pipeline(method, cfg):
    """Grid search -> pinned final fit -> cluster assignment -> figures -> NCC,
    for both conditions. `cfg` is the Config class of the runner script."""
    torch.manual_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    out_root = Path(cfg.results_dir) / method
    out_root.mkdir(parents=True, exist_ok=True)

    data, meta = load_condition_inputs(cfg)
    fits, summary_rows = {}, []

    for cond in CONDITIONS:
        search = grid_search(data[cond], method, cfg, cond, out_root, seed=cfg.random_seed)
        spec = cfg.final_spec.get(cond) if cfg.final_spec else None
        if spec is None:  # fall back to the grid optimum
            b = search['grid_best']
            spec = {'latent_dim': int(b['latent_dim']), 'n_clusters': int(b['n_clusters']),
                    **({'m': float(b['m'])} if method == 'fcm' else {'covariance_type': b['covariance_type']})}
            print(f"  no final_spec pinned for {cond}; using grid optimum {spec}")
        fit = final_fit(search, method, spec, cond, cfg, out_root, seed=cfg.random_seed)

        # Hard assignment with optional low-membership reassignment
        labels, n_re = reassign_low_membership(fit['proba'], cfg.membership_threshold)
        fit['labels_reassigned'] = labels
        assign = meta.copy()
        assign['cluster'] = labels + 1                      # 1-based, as in the figures
        assign['max_membership'] = fit['proba'].max(axis=1)
        for j in range(fit['proba'].shape[1]):
            assign[f'membership_C{j + 1}'] = fit['proba'][:, j]
        for j in range(fit['latents'].shape[1]):
            assign[f'D{j + 1}'] = fit['latents'][:, j]
        assign.to_csv(out_root / f'cluster_assignments_{method}_{cond}.csv', index=False)
        print(f"    reassigned {n_re} low-membership scans (threshold {cfg.membership_threshold})")

        plot_composite_grid(search['results'], method, cond, spec,
                            out_root / f'fig_composite_grid_{method}_{cond}.png', cfg.dpi)
        plot_latent_histograms(search['per_dim'][spec['latent_dim']], cond, method,
                               out_root / f'fig_latent_densities_{method}_{cond}.png', dpi=cfg.dpi)

        # NCC on representative RNFL maps
        if cfg.perform_ncc_analysis:
            paths = rnfl_image_paths(meta, cfg.rnfl_image_dir)
            reps, ncc_df = ncc_representatives(paths, fit['proba'], labels,
                                               n_candidates=cfg.n_candidate_images, n_show=cfg.n_sample_images,
                                               min_avg_ncc=cfg.min_ncc_threshold,
                                               alpha=cfg.ncc_alpha, beta=cfg.ncc_beta)
            ncc_df.to_csv(out_root / f'ncc_per_cluster_{method}_{cond}.csv', index=False)
            mean_ncc, sd_ncc, n_pairs = pooled_ncc(reps)
            fit['metrics'].update({'mean_ncc': mean_ncc, 'sd_ncc': sd_ncc, 'n_ncc_pairs': n_pairs})
            print(f"    NCC: {mean_ncc:.4f} ± {sd_ncc:.4f} over {n_pairs} pairs")
            plot_ncc_panels(reps, meta['Eye'].values, f"{method.upper()} {cond.replace('_', ' ').upper()} — representative RNFL maps",
                            str(out_root / f'fig_ncc_{method}_{cond}'), dpi=cfg.dpi)
            with open(artifact_paths(out_root, method, cond)['metrics'], 'w') as f:
                json.dump(fit['metrics'], f, indent=2)

        fits[cond] = fit
        summary_rows.append({'method': method.upper(), 'condition': cond, **fit['metrics']})

    plot_latent_scatter(fits, method, out_root / f'fig_latent_scatter_{method}.png', cfg.dpi)
    plot_metric_bars(fits, method, out_root / f'fig_metrics_{method}.png', cfg.dpi)

    table = pd.DataFrame(summary_rows)
    table.to_csv(out_root / f'table2_{method}.csv', index=False)
    a, b = fits['with_vf']['metrics']['composite_score'], fits['without_vf']['metrics']['composite_score']
    print(f"\n{'=' * 70}\n{method.upper()} composite: with VF {a:.4f} vs without VF {b:.4f} "
          f"({100 * (a - b) / b:+.1f}%)\nResults -> {out_root}\n{'=' * 70}")
    return fits, table
