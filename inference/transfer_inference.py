"""
FORGE stage 3 — frozen MEE -> external cohort transfer (UK Biobank, MGB Biobank; Fig. 1C).

Pure feature transfer, no retraining / refitting / re-clustering on external data:

  1. artifact-corrected mRNFL maps  --[frozen FORGE encoder, OCT-only path]-->  256-D embeddings
     The visual-field input is replaced by the learned null embedding, because
     external cohorts have no paired perimetry.
  2. 256-D embeddings --[MEE scaler (transform only) + frozen MEE beta-VAE, posterior mean]--> d-D latents
  3. latents --[frozen MEE FCM centroids (m = 1.5) or GMM components]--> memberships / clusters
  4. one latent vector per participant: most recent scan per eye, both eyes averaged
     on the participant's most recent date  (input for the GWAS)
  5. NCC of representative RNFL maps per cluster + 2-D latent scatter

Artifacts come from training/train_forge.py (encoder checkpoint) and
phenotyping/run_vae_fcm.py or run_vae_gmm.py (scaler / VAE / cluster model).

    FORGE_COHORT=ukbb FORGE_EXTERNAL_IMAGES=/path/to/ukbb/rnfl_inpainted \
    python inference/transfer_inference.py
"""

import os
import re
import sys
import json
import time
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, 'training'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'phenotyping'))

from train_forge import EntangledVFRNFLModel, Config as TrainConfig, get_medical_transforms  # noqa: E402
from forge_phenotyping import (load_transfer_artifacts, project_to_mee_latent, predict_cluster_model,  # noqa: E402
                               reassign_low_membership, ncc_representatives, pooled_ncc, plot_ncc_panels)

warnings.filterwarnings('ignore')


# ==================== CONFIGURATION ====================

class Config:
    cohort = os.environ.get('FORGE_COHORT', 'ukbb')              # label used in output file names

    # ---------- external images (artifact-corrected mRNFL maps) ----------
    # File names: <participant>_<eye>[_<token>...].jpg ; eye in {OD, OS, R, L, RIGHT, LEFT}.
    # A token of the form YYYYMMDD or YYYY-MM-DD is taken as the acquisition date
    # (used to pick the most recent scan per eye). Other tokens are kept as `visit`.
    input_images_dir = os.environ.get('FORGE_EXTERNAL_IMAGES', './data/external/rnfl')
    image_extensions = ('.jpg', '.jpeg', '.png')

    # ---------- frozen MEE artifacts ----------
    encoder_checkpoint = os.path.join(os.environ.get('FORGE_MODEL_DIR', './outputs/models'),
                                      'best_entangled_vf_rnfl_model_both_fold0.pt')
    phenotyping_dir = os.environ.get('FORGE_PHENOTYPING_DIR', './outputs/phenotyping')
    method = 'fcm'          # 'fcm' (primary, 15 endophenotypes) or 'gmm' (robustness check)
    condition = 'with_vf'   # VF-guided MEE latent space / cluster models are transferred;
                            # the external images themselves always go through the OCT-only path

    # ---------- outputs ----------
    output_dir = os.environ.get('FORGE_INFERENCE_DIR', f'./outputs/inference_{cohort}')

    # ---------- assignment / NCC ----------
    membership_threshold = None     # next-best reassignment (None = off; MEE used 0.7)
    perform_ncc_analysis = True
    n_candidate_images = 20
    n_sample_images = 5
    min_ncc_threshold = 0.08
    ncc_alpha = 0.5
    ncc_beta = 0.5

    # ---------- runtime ----------
    batch_size = 64
    num_workers = 4
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    random_seed = 42
    dpi = 200


config = Config()
torch.manual_seed(config.random_seed)
np.random.seed(config.random_seed)

EYE_MAP = {'OD': 'OD', 'R': 'OD', 'RIGHT': 'OD', 'RE': 'OD',
           'OS': 'OS', 'L': 'OS', 'LEFT': 'OS', 'LE': 'OS'}
DATE_RE = re.compile(r'^(\d{8}|\d{4}-\d{2}-\d{2})$')


# ==================== DATA ====================

def parse_filename(filename):
    stem = os.path.splitext(filename)[0]
    parts = stem.split('_')
    if len(parts) < 2 or parts[1].upper() not in EYE_MAP:
        return None
    rec = {'filename': filename, 'participant_id': parts[0], 'eye': EYE_MAP[parts[1].upper()],
           'date': None, 'visit': '_'.join(parts[2:])}
    for tok in parts[2:]:
        if DATE_RE.match(tok):
            rec['date'] = tok.replace('-', '')
            break
    return rec


class ExternalRNFLDataset(Dataset):
    def __init__(self, metadata, image_dir, transform):
        self.meta = metadata.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        row = self.meta.iloc[i]
        img = Image.open(os.path.join(self.image_dir, row['filename'])).convert('RGB')
        eye = torch.tensor(0 if row['eye'] == 'OD' else 1, dtype=torch.long)
        return self.transform(img), eye


def scan_images():
    print(f"\n[1/5] Scanning {config.input_images_dir}")
    files = sorted(f for f in os.listdir(config.input_images_dir) if f.lower().endswith(config.image_extensions))
    records = [r for r in (parse_filename(f) for f in files) if r is not None]
    if not records:
        raise ValueError("No images with a parsable <participant>_<eye>_... name were found")
    meta = pd.DataFrame(records)
    print(f"  {len(files)} files, {len(meta)} parsable, {meta['participant_id'].nunique()} participants")
    print(f"  eyes: " + ", ".join(f"{e} = {n}" for e, n in meta['eye'].value_counts().items()))
    if meta['date'].isna().all():
        print("  no date token found in file names -> per-participant vectors will average all scans per eye")
    return meta


# ==================== ENCODER (OCT-only path) ====================

def load_encoder():
    print(f"\n[2/5] Loading frozen FORGE encoder: {config.encoder_checkpoint}")
    ckpt = torch.load(config.encoder_checkpoint, map_location=config.device, weights_only=False)
    model = EntangledVFRNFLModel(TrainConfig, num_classes=2)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(config.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"  checkpoint fold {ckpt.get('fold', '?')}, epoch {int(ckpt.get('epoch', -1)) + 1}")
    return model


@torch.no_grad()
def extract_embeddings(model, meta):
    loader = DataLoader(ExternalRNFLDataset(meta, config.input_images_dir, get_medical_transforms(training=False)),
                        batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    out = []
    for imgs, eyes in tqdm(loader, desc="  encoding (null-VF path)"):
        o = model(imgs.to(config.device), None, eyes.to(config.device), inference_mode=True)
        out.append(o['rnfl_embedding'].float().cpu().numpy())
    emb = np.concatenate(out, axis=0)
    print(f"  embeddings: {emb.shape}")
    return emb


# ==================== PARTICIPANT-LEVEL AGGREGATION ====================

def aggregate_per_participant(meta, latents):
    """One vector per participant (Methods, 'Latent vector extraction and GWAS'):
    keep the most recent scan per eye; take the participant's most recent date
    across eyes and average the eye vectors acquired on that date. Without dates,
    all scans of an eye are averaged, then the two eyes are averaged."""
    d = latents.shape[1]
    cols = [f'D{i + 1}' for i in range(d)]
    df = pd.concat([meta[['participant_id', 'eye', 'date']].reset_index(drop=True),
                    pd.DataFrame(latents, columns=cols)], axis=1)
    has_date = df['date'].notna().all()
    if has_date:
        df = df.sort_values('date')
        per_eye = df.groupby(['participant_id', 'eye']).tail(1)            # most recent per eye
        last = per_eye.groupby('participant_id')['date'].transform('max')
        per_eye = per_eye[per_eye['date'] == last]                          # eyes acquired on the latest date
        agg = per_eye.groupby('participant_id')[cols].mean()
        agg['date'] = per_eye.groupby('participant_id')['date'].first()
    else:
        per_eye = df.groupby(['participant_id', 'eye'])[cols].mean().reset_index()
        agg = per_eye.groupby('participant_id')[cols].mean()
    agg['n_eyes'] = per_eye.groupby('participant_id')['eye'].nunique()
    return agg.reset_index()


# ==================== MAIN ====================

def main():
    t0 = time.time()
    os.makedirs(config.output_dir, exist_ok=True)
    print("=" * 80)
    print(f"FORGE TRANSFER INFERENCE — MEE -> {config.cohort.upper()}  "
          f"[{config.method.upper()} / {config.condition} artifacts, OCT-only encoder path]")
    print("=" * 80)

    meta = scan_images()
    model = load_encoder()
    emb = extract_embeddings(model, meta)
    del model
    torch.cuda.empty_cache()

    print(f"\n[3/5] Projecting into the MEE latent space ({config.method}, {config.condition})")
    scaler, vae, cluster_model, spec = load_transfer_artifacts(config.phenotyping_dir + f'/{config.method}',
                                                               config.method, config.condition, config.device)
    latents = project_to_mee_latent(emb, scaler, vae, config.device)
    labels0, proba = predict_cluster_model(cluster_model, latents)
    labels, n_re = reassign_low_membership(proba, config.membership_threshold)
    k = proba.shape[1]
    print(f"  latents {latents.shape}; {k} clusters (spec {spec}); reassigned {n_re}")
    for c in range(k):
        n = int((labels == c).sum())
        print(f"    C{c + 1}: {n:>7,} scans ({100 * n / len(labels):5.1f}%)  mean membership "
              f"{proba[labels == c, c].mean() if n else 0:.3f}")

    print(f"\n[4/5] Saving")
    tag = f"{config.cohort}_{config.method}_{config.condition}"
    np.save(os.path.join(config.output_dir, f'{tag}_rnfl_embeddings_256d.npy'), emb)
    np.save(os.path.join(config.output_dir, f'{tag}_latents_{latents.shape[1]}d.npy'), latents)
    np.save(os.path.join(config.output_dir, f'{tag}_memberships_{k}c.npy'), proba)

    scans = meta.copy()
    scans['cluster'] = labels + 1
    scans['max_membership'] = proba.max(axis=1)
    for j in range(k):
        scans[f'membership_C{j + 1}'] = proba[:, j]
    for j in range(latents.shape[1]):
        scans[f'D{j + 1}'] = latents[:, j]
    scans.to_csv(os.path.join(config.output_dir, f'{tag}_scan_level.csv'), index=False)

    participants = aggregate_per_participant(meta, latents)
    participants.to_csv(os.path.join(config.output_dir, f'{tag}_participant_level.csv'), index=False)
    print(f"  scan-level: {len(scans)} rows; participant-level: {len(participants)} rows "
          f"-> {tag}_participant_level.csv (GWAS phenotype file)")

    summary = {'cohort': config.cohort, 'n_images': int(len(meta)),
               'n_participants': int(meta['participant_id'].nunique()),
               'method': config.method, 'condition': config.condition, 'spec': spec,
               'encoder_checkpoint': config.encoder_checkpoint,
               'cluster_sizes': {f'C{c + 1}': int((labels == c).sum()) for c in range(k)}}

    print(f"\n[5/5] Figures")
    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.cm.tab10 if k <= 10 else plt.cm.tab20
    for c in range(k):
        m = labels == c
        if m.any():
            ax.scatter(latents[m, 0], latents[m, 1], s=6, alpha=0.5, color=cmap(c), label=f'C{c + 1} (n={m.sum():,})')
    ax.set_xlabel('VAE dimension 1'); ax.set_ylabel('VAE dimension 2')
    ax.set_title(f'{config.cohort.upper()} in MEE space ({config.method.upper()}, {latents.shape[1]}D)')
    ax.legend(fontsize=8, markerscale=3, loc='center left', bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(config.output_dir, f'fig_{tag}_latent_scatter.png'), dpi=config.dpi, bbox_inches='tight')
    plt.close(fig)

    if config.perform_ncc_analysis:
        paths = [os.path.join(config.input_images_dir, f) for f in meta['filename']]
        reps, ncc_df = ncc_representatives(paths, proba, labels, n_candidates=config.n_candidate_images,
                                           n_show=config.n_sample_images, min_avg_ncc=config.min_ncc_threshold,
                                           alpha=config.ncc_alpha, beta=config.ncc_beta)
        ncc_df.to_csv(os.path.join(config.output_dir, f'{tag}_ncc_per_cluster.csv'), index=False)
        mean_ncc, sd_ncc, n_pairs = pooled_ncc(reps)
        summary.update({'mean_ncc': mean_ncc, 'sd_ncc': sd_ncc, 'n_ncc_pairs': n_pairs})
        print(f"  NCC {mean_ncc:.4f} ± {sd_ncc:.4f} over {n_pairs} pairs")
        plot_ncc_panels(reps, meta['eye'].values, f"{config.cohort.upper()} — {config.method.upper()} representative RNFL maps",
                        os.path.join(config.output_dir, f'fig_{tag}_ncc'), dpi=config.dpi)

    with open(os.path.join(config.output_dir, f'{tag}_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min -> {config.output_dir}")
    return scans, participants


if __name__ == '__main__':
    main()
