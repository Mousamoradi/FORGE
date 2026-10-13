"""
FORGE stage 2 — beta-VAE + fuzzy c-means endophenotyping (MEE development cohort).

Input : the 256-D RNFL embeddings written by training/train_forge.py
            WITH VF    : VF-guided path      entangled_rnfl_embeddings_<eye>_oof.npy
            WITHOUT VF : null-VF (OCT-only)  entangled_rnfl_octonly_embeddings_<eye>_oof.npy
        + the row-aligned metadata           oof_metadata_<eye>.csv
Sweep : latent dimension x cluster number (k = 4..15) x fuzziness m in {1.5, 2.0, 2.5, 3.0}
Score : composite = mean(Silhouette, CH-norm, DB-inv, diversity score)   (Table 2)
Final : pinned manuscript configuration (FCM with VF: 15D / 9 clusters / m = 1.5;
        without VF: 20D / 12 clusters / m = 1.5) -> transfer artifacts
        (scaler, VAE, FCM) consumed by inference/transfer_inference.py.

    FORGE_DATA_ROOT=/path/to/data FORGE_OUTPUT_DIR=./outputs/embeddings \
    python phenotyping/run_vae_fcm.py
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
    fold_filter = None          # e.g. 0 -> only the held-out scans of fold 0; None -> all OOF scans

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

    # ---------- FCM grid ----------
    n_clusters_range = list(range(4, 16))
    fuzziness_m_range = [1.5, 2.0, 2.5, 3.0]
    covariance_types = None     # GMM only

    # ---------- pinned manuscript configuration (set to None to use the grid optimum) ----------
    final_spec = {
        'with_vf':    {'latent_dim': 15, 'n_clusters': 9,  'm': 1.5},
        'without_vf': {'latent_dim': 20, 'n_clusters': 12, 'm': 1.5},
    }

    # ---------- diversity / assignment / NCC ----------
    n_histogram_bins = 50
    membership_threshold = 0.7  # scans with max membership below this move to their next-best cluster; None = off
    perform_ncc_analysis = True
    n_candidate_images = 20     # highest-membership scans considered per cluster
    n_sample_images = 5         # representatives shown per cluster
    min_ncc_threshold = 0.08    # candidates with mean NCC below this are dropped
    ncc_alpha = 0.5             # ranking weight: membership
    ncc_beta = 0.5              # ranking weight: mean NCC

    # ---------- runtime ----------
    dpi = 200
    verbose = True
    random_seed = 42
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    t0 = time.time()
    run_pipeline('fcm', Config)
    print(f"elapsed {(time.time() - t0) / 60:.1f} min")
