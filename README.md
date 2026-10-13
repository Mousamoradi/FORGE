# FORGE: Functionally guided OCT Representation for Glaucoma Endophenotyping

**Mousa Moradi\*, Liyin Chen\*, Yan Zhao, Sayuri Sekimitsu, Niloufar Bineshfar, Janey L. Wiggs, Mohammad Eslami, Mengyu Wang, Tobias Elze, Ayellet V. Segrè, Nazlee Zebardast\*\***

Harvard Ophthalmology AI Lab, Schepens Eye Research Institute, Massachusetts Eye and Ear, Harvard Medical School · Broad Institute of MIT and Harvard

\*Equal contribution · \*\*Corresponding author: Nazlee_Zebardast@MEEI.HARVARD.EDU

---

FORGE is a cross-modal, privileged-learning framework for glaucoma endophenotyping. It uses the **visual field (VF) as privileged supervision during training** to shape an OCT macular RNFL (mRNFL) representation, and a **learned null embedding** makes the VF dispensable at inference, so endophenotypes can be assigned from **structural OCT alone**. This lets a model trained on a small, richly phenotyped clinical dataset (Mass Eye and Ear) be deployed, without refitting, to population-scale imaging biobanks that lack perimetry (UK Biobank, Mass General Brigham Biobank) for genome-wide discovery.

![FORGE framework](figures/fig1_forge_framework.png)

**Fig. 1. FORGE framework.** (A) Paired OCT RNFL and VF images are encoded by a shared encoder trained with privileged VF supervision; a β-VAE sweep over latent dimension and cluster number (FCM and GMM) selects the configuration by composite clustering score (FCM, d = 15, k = 9, 0.766; GMM, d = 20, k = 13, 0.701). (B) Latent- and pixel-level analysis: diversity-scored latent embeddings (with vs. without VF) and per-cluster normalized cross-correlation (NCC) matrices with representative RNFL maps. (C) Learned endophenotypes transfer cross-cohort to UK Biobank and MGB Biobank for GWAS discovery and replication, followed by post-GWAS analyses; OCT-only inference is enabled by replacing the VF input with the learned null embedding.

## Key results

| | |
|---|---|
| Development cohort | 5,372 paired OCT–VF examinations, 1,845 patients, 2,819 eyes (Mass Eye and Ear) |
| Endophenotypes | 15 continuous mRNFL latent axes (FCM, 15D) → 9 phenotypic clusters; GMM (20D) → 13 clusters as a robustness check |
| Effect of VF guidance | Composite clustering score **+22.0 %** for FCM (0.766 vs 0.628) and **+11.3 %** for GMM (0.701 vs 0.630) over an identically trained OCT-only ablation |
| External transfer (no refitting) | 74,077 UK Biobank images (47,252 participants) and 3,119 MGB Biobank images (1,020 participants); per-cluster mean RNFL thickness concordant across cohorts (Lin's CCC 0.74) |
| Clinical validity (UKBB) | 14/15 axes associated with POAG (max OR 2.17 per SD); 15 endophenotypes add +0.023 AUROC over age, sex, IOP, RNFL thickness and a POAG polygenic score |
| GWAS (39,381 UKBB Europeans) | **53 LD-independent loci** in 41 regions, **23 novel** for glaucoma/IOP; all 15 axes heritable (LDSC h² 0.06–0.12) |
| Replication (711 MGBB Europeans) | 75 % sign concordance (P = 0.002), Deming slope 0.74 |
| VF guidance vs ablation | Median heritability Z 6.15 vs 4.76 (P = 9 × 10⁻⁵); only VF-guided axes resolve retinal ganglion-cell enrichment |
| Prioritized genes | 28 putative causal genes; *ARHGEF7*, *CNKSR3*, *ADCY5* at novel loci supported by retina colocalization and SMR |

### Visual-field guidance improves the learned representation

![VF guidance ablation](figures/fig2_vf_guidance_ablation.png)

**Fig. 2.** (A, B) Latent-space projections and cluster assignments for FCM (9 clusters, 15D) and GMM (13 clusters, 20D), with VF guidance (left) vs. the OCT-only ablation (right). (C) Clustering quality metrics on the held-out MEE test partition (n = 538): composite score, Silhouette, normalized Calinski–Harabasz, inverted Davies–Bouldin and diversity score. The gain is consistent across every metric and both clustering algorithms.

### Four lines of genetic evidence prioritize 28 genes

![Gene prioritization](figures/fig7_gene_prioritization.png)

**Fig. 7.** (A) Support from MAGMA gene-based association, S-PrediXcan/S-MultiXcan transcriptome- and splicing-wide association, colocalization and SMR for the 28 genes supported by at least two analyses. *ARHGEF7*, *CNKSR3* and *ADCY5* (orange) are at loci novel to POAG, IOP and our previous OCT-only endophenotype GWAS. (B) Locus-comparison plots at those three loci (retina cis-eQTL vs. GWAS −log₁₀ P), with SuSiE 95 % credible-set members and the coloc shared variant.

## Repository layout

```
FORGE/
├── README.md
├── LICENSE                       MIT
├── requirements.txt              minimum library versions
├── training/
│   └── train_forge.py            VF-guided (privileged) contrastive encoder + patient-level 5-fold CV — Fig. 1A
└── figures/
    ├── fig1_forge_framework.png
    ├── fig2_vf_guidance_ablation.png
    └── fig7_gene_prioritization.png
```

`training/train_forge.py` is the complete, self-contained training script for the FORGE encoder (shared ResNet-50 backbone, modality-specific encoders, cross-attention + FiLM entangled fusion, MoCo queue, learned null-VF embedding, and the multi-objective loss). Downstream phenotyping (β-VAE sweep, FCM/GMM clustering, NCC, cross-cohort transfer) and the GWAS / post-GWAS pipelines are described in [Pipeline overview](#pipeline-overview) below.

## Installation

```bash
git clone git@github.com:Mousamoradi/FORGE.git
cd FORGE
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python ≥ 3.9 and a CUDA-capable GPU for training (the paper used NVIDIA H100 80 GB; ≈2–3 min per epoch on the 5,372-scan MEE dataset). CPU execution works for inference but is slow for training.

## Data preparation

Clinical data from Mass Eye and Ear cannot be redistributed (IRB); UK Biobank and MGB Biobank data are available through their respective access procedures. To train on your own paired data, use the following layout:

```
<data_root>/
├── rnfl/                      artifact-corrected mRNFL thickness maps (RGB .jpg)
│   ├── 10023_OD_20190412_1.jpg
│   └── ...
├── vf/                        matched visual-field images (RGB .jpg)
│   ├── 10023_OD_20190412_1.jpg
│   └── ...
└── progression_labels.csv     columns: ID, Eye, Date, index, Progression_Ensemble
```

* File names follow `ID_EYE_DATE_INDEX.jpg` with `EYE ∈ {OD, OS}`. RNFL and VF images are paired by exact match on all four fields (OCT–VF temporal matching within ±6 months is done upstream).
* `Progression_Ensemble` is the binary VF-progression label (0 = not worsening, 1 = worsening) from the `vfprogression` ensemble criteria (MD slope, Schell method, VFI). Images without a label are still used for the self-supervised objectives (`ignore_index = -1`).
* Inputs expected by the script are the **artifact-corrected** maps: OCT scans (Zeiss CIRRUS HD-OCT, 200 × 200 macular cube, 6 × 6 mm) with signal strength ≥ 7 and ≤ 5 % missing pixels, inpainted with the partial-convolution U-Net described in the paper (Methods, *Artifact Correction*).

## Training

```bash
FORGE_DATA_ROOT=/path/to/data \
FORGE_OUTPUT_DIR=./outputs/embeddings \
FORGE_MODEL_DIR=./outputs/models \
python training/train_forge.py
```

All hyperparameters live in the `Config` class at the top of `train_forge.py`:

| Group | Setting |
|---|---|
| Backbone | ImageNet-pretrained ResNet-50, shared by both modalities → 512-D shared bottleneck |
| Modality encoders | 256-D RNFL and VF encoders with residual projection |
| Entangled fusion | 8-head cross-attention (RNFL ↔ VF) + FiLM modulation (`z'_RNFL = γ(z_VF) ⊙ z_RNFL + β(z_VF)`) → 256-D joint embedding |
| Privileged VF | learned `null_vf_embedding` (256-D); VF dropout p = 0.1 during training |
| Contrastive | MoCo queue K = 4096, momentum m = 0.999, temperature τ = 0.1 |
| Loss (Eq. 6) | L = λ₁·L_MoCo + λ₂·L_cross + λ₃·L_div + λ₄·L_prog with **λ = (0.3, 0.15, 0.15, 0.4)**, selected with Optuna (20 trials) on validation clustering metrics. L_cross = cross-modal consistency between the joint embedding and each modality (Eq. 3); L_div = anti-collapse margin on RNFL–VF cosine similarity (Eq. 4); L_prog = progression cross-entropy on the joint and RNFL heads (Eq. 5). |
| Optimization | AdamW, lr 5 × 10⁻⁵ (2× for fusion/classifier heads), wd 5 × 10⁻⁵, 15-epoch warm-up + cosine, grad-clip 0.5, AMP, batch 16, ≤ 150 epochs, early stopping (patience 9) on validation loss − 0.2 × accuracy |
| Cross-validation | **Patient-level, label-stratified five-fold CV** (`StratifiedGroupKFold`, groups = patient ID, strata = patient's majority progression label). In each fold 80 % of patients are used for training + validation (a further patient-level split gives 70 % train / 10 % val) and 20 % are held out for testing. All examinations and both eyes of a patient belong to exactly one fold; the script asserts this. |

A fresh model is trained in every fold, the best checkpoint of that fold is reloaded, held-out metrics are computed on the test patients, and the held-out scans' embeddings are written. Across the five folds every scan is held out exactly once, giving a complete out-of-fold embedding set for the downstream β-VAE / clustering stage. Set `Config.folds_to_run = [0]` to run a single fold.

**Outputs** (`FORGE_MODEL_DIR` / `FORGE_OUTPUT_DIR`, default `./outputs/`)

* `models/best_entangled_vf_rnfl_model_both_fold{k}.pt` — best checkpoint per fold (model/optimizer state, config, history)
* `embeddings/cv_fold_assignment_both.csv` — fold of every scan (`ID, Eye, Date, Index, class_label, fold`)
* `embeddings/entangled_{joint,rnfl,vf}_embeddings_both_fold{k}.npy` — held-out 256-D embeddings of fold *k*, VF-guided path
* `embeddings/entangled_{joint,rnfl}_octonly_embeddings_both_fold{k}.npy` — same scans through the OCT-only (null-VF) path
* `embeddings/test_metadata_both_fold{k}.csv` — row order of the per-fold `.npy` files
* `embeddings/entangled_*_embeddings_both_oof.npy` + `oof_metadata_both.csv` — all scans, out-of-fold, in the order of `oof_metadata_both.csv` (written when all five folds run)
* `embeddings/cv_summary_both.json` — per-fold best epoch, validation loss/accuracy, held-out accuracy and AUROC, and their mean ± SD

### OCT-only inference (learned null embedding)

The trained model runs on RNFL alone by passing `vf_img=None`; the learned null embedding is substituted for the missing modality:

```python
import torch
from training.train_forge import EntangledVFRNFLModel, config, get_medical_transforms

ckpt = torch.load("outputs/models/best_entangled_vf_rnfl_model_both_fold0.pt",
                  map_location="cuda", weights_only=False)
model = EntangledVFRNFLModel(config, num_classes=2).cuda().eval()
model.load_state_dict(ckpt["model_state_dict"])

x = get_medical_transforms(training=False)(rnfl_pil_image).unsqueeze(0).cuda()
eye = torch.tensor([0]).cuda()                       # 0 = OD, 1 = OS
with torch.no_grad():
    out = model(x, vf_img=None, eye_labels=eye, inference_mode=True)
rnfl_embedding = out["rnfl_embedding"]               # 256-D, used for all external cohorts
```

This is the path used to phenotype UK Biobank and MGB Biobank, neither of which has paired perimetry.

## Pipeline overview

1. **Curation & artifact correction** (Fig. 9 in the paper). Quality filtering (signal strength ≥ 7, ≤ 5 % missing), ±6-month OCT–VF temporal matching, and inpainting with a partial-convolution U-Net trained on the maps' own missing-pixel masks.
2. **VF-guided encoder** — `training/train_forge.py` (this repo). Produces 256-D RNFL embeddings.
3. **Distribution-aware latent modeling.** Embeddings are standardized and projected through a β-VAE; a grid sweep over latent dimension (d = 2–20) and cluster number (k = 4–15) with fuzzy c-means (m ∈ {1.5, 2.0, 2.5, 3.0}) and GMM (full/diagonal covariance) selects the configuration by a composite of Silhouette, normalized Calinski–Harabasz, inverted Davies–Bouldin and diversity score. Selected: **FCM 15D / k = 9 (m = 1.5)** and **GMM 20D / k = 13**. Within-cluster coherence is quantified by NCC.
4. **Cross-cohort transfer.** The frozen encoder, scaler, VAE encoder (posterior mean μ) and cluster models are applied as-is to UKBB and MGBB through the OCT-only path. One phenotype vector per participant: most recent scan per eye, both eyes averaged on the most recent date.
5. **Clinical validity.** Criterion / convergent / discriminant validity against 16 outcomes and incremental AUROC with nested-CV ridge logistic regression.
6. **GWAS & post-GWAS.** REGENIE v4.1.2 on the 15 rank-inverse-normalized axes (age, sex, array, refractive error, 10 PCs); PLINK 2 LD clumping; LDSC and GCTA-GREML heritability; replication in MGBB (PLINK 2 `--glm`); SuSiE-RSS fine-mapping; MAGMA; S-PrediXcan / S-MultiXcan (GTEx v8 MASHR); coloc (GTEx + EyeGEx retina); SMR/HEIDI; ECLIPSER cell-type enrichment.

## Citation

If you use FORGE, please cite:

```bibtex
@article{moradi2026forge,
  title   = {FORGE: Functionally guided OCT Representation for Glaucoma Endophenotyping},
  author  = {Moradi, Mousa and Chen, Liyin and Zhao, Yan and Sekimitsu, Sayuri and Bineshfar, Niloufar and
             Wiggs, Janey L. and Eslami, Mohammad and Wang, Mengyu and Elze, Tobias and Segr{\`e}, Ayellet V. and
             Zebardast, Nazlee},
  year    = {2026},
  Preprint: https://www.medrxiv.org/content/10.64898/2026.05.08.26352729v3
}
```

## Data and code availability

* Code: this repository (https://github.com/Mousamoradi/FORGE).
* Data supporting the findings can be obtained from the corresponding author upon reasonable request. UK Biobank GWAS summary statistics for the 15 endophenotypes will be shared publicly upon publication.

## Acknowledgments

This research was funded by the National Institutes of Health (R01 EY036518, R01 EY030575, R01 EY038687, R01 EY036222, and P30 EY003790) and Research to Prevent Blindness. This research was conducted using the UK Biobank Resource and the Mass General Brigham Biobank.

## License

Released under the [MIT License](LICENSE).
