"""
FORGE stage 2 — beta-VAE + Gaussian-mixture endophenotyping (MEE development cohort).

Robustness check for the FCM pipeline (run_vae_fcm.py) with a mechanistically
different clustering algorithm. Same inputs, same sweep logic and the same
composite score; only the cluster model differs.

Sweep : latent dimension x cluster number (k = 4..15) x covariance in {full, diag}
Final : pinned manuscript configuration (GMM with VF: 20D / 13 clusters, full
        covariance; without VF: 20D / 11 clusters) -> transfer artifacts
        (scaler, VAE, GMM) consumed by inference/transfer_inference.py.

    FORGE_DATA_ROOT=/path/to/data FORGE_OUTPUT_DIR=./outputs/embeddings \
    python phenotyping/run_vae_gmm.py
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forge_phenotyping import run_pipeline  # noqa: E402


class Config:
    # ---------- inputs (outputs of training/train_forge.py) ----------
    eye_selection = 'both'
    embeddings_dir = os.environ.get('FORGE_OUTPUT_DIR', './outputs/embeddings')
    with_vf_embeddings = os.path.join(embeddings_dir, f'entangled_rnfl_embeddings_{eye_selection}_oof.npy')
    without_vf_embeddings = os.path.join(embeddings_dir, f'entangled_rnfl_octonly_embeddings_{eye_selection}_oof.npy')
    metadata_csv = os.path.join(embeddings_dir, f'oof_metadata_{eye_selection}.csv')
    fold_filter = None

    # ---------- RNFL maps for NCC (ID_EYE_DATE_INDEX.jpg) ----------
    rnfl_image_dir = os.path.join(os.environ.get('FORGE_DATA_ROOT', './data'), 'rnfl')

    # ---------- outputs ----------
    results_dir = os.environ.get('FORGE_PHENOTYPING_DIR', './outputs/phenotyping')

    # ---------- beta-VAE ----------
    latent_dims = [2, 5, 10, 15, 20]
    hidden_dims = [256, 128, 64, 32]
    dropout_rate = 0.1
    beta = 1.25                 # KL weight of the beta-VAE
    batch_size = 64
    learning_rate = 1e-3
    epochs = 200
    early_stopping_patience = 20

    # ---------- GMM grid ----------
    n_clusters_range = list(range(4, 16))
    covariance_types = ['full', 'diag']
    fuzziness_m_range = None    # FCM only

    # ---------- pinned manuscript configuration (set to None to use the grid optimum) ----------
    final_spec = {
        'with_vf':    {'latent_dim': 20, 'n_clusters': 13, 'covariance_type': 'full'},
        'without_vf': {'latent_dim': 20, 'n_clusters': 11, 'covariance_type': 'full'},
    }

    # ---------- diversity / assignment / NCC ----------
    n_histogram_bins = 50
    membership_threshold = 0.7  # posterior-probability threshold for next-best reassignment; None = off
    perform_ncc_analysis = True
    n_candidate_images = 20
    n_sample_images = 5
    min_ncc_threshold = 0.08
    ncc_alpha = 0.5
    ncc_beta = 0.5

    # ---------- runtime ----------
    dpi = 200
    verbose = True
    random_seed = 42
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    t0 = time.time()
    run_pipeline('gmm', Config)
    print(f"elapsed {(time.time() - t0) / 60:.1f} min")
