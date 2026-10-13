"""
FORGE: Functionally guided OCT Representation for Glaucoma Endophenotyping
ENTANGLED VF-RNFL CONTRASTIVE LEARNING MODEL

Shared ResNet-50 backbone, modality-specific encoders, cross-attention + FiLM
entangled fusion, momentum-contrastive (MoCo) queue and a learned null-VF
embedding that enables OCT-only inference.

Loss (paper Eq. 6):  L = λ1·L_MoCo + λ2·L_cross + λ3·L_div + λ4·L_prog
                     λ = (0.3, 0.15, 0.15, 0.4)

Evaluation: patient-level, label-stratified five-fold cross-validation. In each
fold 80% of patients are used for training + validation (70% / 10%) and 20% are
held out for testing; all scans and both eyes of a patient belong to one fold.
Per-fold checkpoints, held-out metrics and out-of-fold 256-D embeddings (VF-guided
and OCT-only/null-VF paths) are written to FORGE_OUTPUT_DIR / FORGE_MODEL_DIR.
"""

import os
import json
import warnings
from glob import glob
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================

class Config:
    """Configuration for entangled VF-RNFL model"""
    
    # ==================== DIRECTORIES ====================
    # Override the data root with the FORGE_DATA_ROOT environment variable, e.g.
    #   FORGE_DATA_ROOT=/path/to/data python training/train_forge.py
    # Expected layout (see README "Data preparation"):
    #   <data_root>/rnfl/   artifact-corrected mRNFL thickness maps  (ID_EYE_DATE_INDEX.jpg)
    #   <data_root>/vf/     matched visual-field images               (ID_EYE_DATE_INDEX.jpg)
    #   <data_root>/progression_labels.csv   columns: ID, Eye, Date, index, Progression_Ensemble
    data_root = os.environ.get('FORGE_DATA_ROOT', './data')
    rnfl_dir = os.path.join(data_root, 'rnfl')
    vf_dir = os.path.join(data_root, 'vf')
    labels_csv = os.path.join(data_root, 'progression_labels.csv')
    output_dir = os.environ.get('FORGE_OUTPUT_DIR', './outputs/embeddings')
    model_dir = os.environ.get('FORGE_MODEL_DIR', './outputs/models')
    
    # ==================== EYE SELECTION ====================
    eye_selection = 'both'  # 'both', 'OD', or 'OS'
    
    # ==================== ENTANGLEMENT PARAMETERS ====================
    # Shared representation dimensions
    shared_feature_dim = 512  # Shared bottleneck dimension
    modality_feature_dim = 256  # Modality-specific features
    final_embedding_dim = 256  # Final joint embedding

    # Cross-attention parameters
    num_attention_heads = 8
    attention_dim = 512

    # FiLM (Feature-wise Linear Modulation) parameters
    film_hidden_dim = 128

    # ==================== MODEL PARAMETERS ====================
    moco_k = 4096
    moco_m = 0.999
    moco_t = 0.1

    # ==================== LOSS WEIGHTS (paper Eq. 6) ====================
    # L_total = λ1·L_MoCo + λ2·L_cross + λ3·L_div + λ4·L_prog
    # Values selected with Optuna (20 trials) on validation clustering metrics.
    lambda_moco = 0.3    # λ1: momentum-contrastive loss (Eq. 2)
    lambda_cross = 0.15  # λ2: cross-modal consistency, joint vs. RNFL/VF (Eq. 3)
    lambda_div = 0.15    # λ3: diversity / anti-collapse loss (Eq. 4)
    lambda_prog = 0.4    # λ4: progression classification (Eq. 5)

    # Auxiliary terms NOT part of Eq. 6 — disabled by default (weight 0.0).
    # Kept so the ablations can be re-run; set > 0 to enable.
    lambda_attention = 0.0  # cross-modal attention entropy regularizer
    lambda_tabular = 0.0    # auxiliary age/sex/MD regression head

    # ==================== TRAINING PARAMETERS ====================
    batch_size = 16
    num_epochs = 150
    base_lr = 5e-5
    weight_decay = 5e-5
    gradient_clip = 0.5
    warmup_epochs = 15
    dropout_rate = 0.3

    # ==================== CROSS-VALIDATION (patient-level) ====================
    # Five-fold CV with patient-level grouping and label stratification: every
    # examination and both eyes of a patient fall in exactly one fold. In each
    # fold 80% of patients are used for training + validation and the remaining
    # 20% are held out for testing. Within the 80%, a further patient-level
    # split sets aside `val_fraction` of those patients for validation / early
    # stopping (0.125 of 80% = 10% of all patients).
    n_folds = 5
    val_fraction = 0.125
    folds_to_run = None  # None = all folds, or e.g. [0, 1] to run a subset

    # ==================== OTHER ====================
    num_workers = 8
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    seed = 42
    patience = 9
    save_every = 10

config = Config()

# Set random seeds
torch.manual_seed(config.seed)
np.random.seed(config.seed)
random.seed(config.seed)

def print_banner():
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.model_dir, exist_ok=True)

    print("\n" + "="*70)
    print(f"ENTANGLED VF-RNFL CONTRASTIVE LEARNING")
    print("="*70)
    print(f"Eye selection: {config.eye_selection.upper()}")
    print("\nEntanglement Architecture:")
    print(f"  - Shared bottleneck: {config.shared_feature_dim}D")
    print(f"  - Cross-attention heads: {config.num_attention_heads}")
    print(f"  - FiLM modulation enabled")
    print(f"  - Joint embedding: {config.final_embedding_dim}D")
    print("\nLoss Weights (Eq. 6):")
    print(f"  - λ1 MoCo:          {config.lambda_moco}")
    print(f"  - λ2 Cross-modal:   {config.lambda_cross}")
    print(f"  - λ3 Diversity:     {config.lambda_div}")
    print(f"  - λ4 Progression:   {config.lambda_prog}")
    print(f"\nCross-validation: {config.n_folds}-fold, patient-level, "
          f"val fraction {config.val_fraction} of train+val")
    print("="*70 + "\n")


# ==================== FEATURE-WISE LINEAR MODULATION (FiLM) ====================

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation for VF->RNFL feature conditioning"""
    
    def __init__(self, feature_dim, condition_dim, hidden_dim=128):
        super().__init__()
        
        # Generate scale and shift parameters from conditioning features
        self.film_generator = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * feature_dim)  # gamma and beta
        )
        
    def forward(self, x, condition):
        """
        x: features to modulate [B, feature_dim]
        condition: conditioning features (e.g., VF) [B, condition_dim]
        """
        film_params = self.film_generator(condition)
        gamma, beta = film_params.chunk(2, dim=-1)
        
        # Apply affine transformation
        return gamma * x + beta

# ==================== CROSS-MODAL ATTENTION ====================

class CrossModalAttention(nn.Module):
    """Multi-head cross-attention between RNFL and VF features"""
    
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        assert self.head_dim * num_heads == dim
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, query, key, value, mask=None):
        """
        query: RNFL features [B, dim]
        key, value: VF features [B, dim]
        """
        B = query.size(0)
        
        # Add sequence dimension for attention
        query = query.unsqueeze(1)  # [B, 1, dim]
        key = key.unsqueeze(1)      # [B, 1, dim]  
        value = value.unsqueeze(1)  # [B, 1, dim]
        
        # Project to Q, K, V
        Q = self.q_proj(query).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, V)  # [B, num_heads, 1, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, 1, self.dim)
        out = self.out_proj(out).squeeze(1)  # [B, dim]
        
        return out, attn_weights.squeeze()

# ==================== SHARED ENCODER BACKBONE ====================

class SharedEncoderBackbone(nn.Module):
    """Shared convolutional layers for both RNFL and VF"""
    
    def __init__(self, shared_feature_dim=512):
        super().__init__()
        
        try:  # torchvision >= 0.13 weights API
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        except (AttributeError, TypeError):  # older torchvision
            resnet = models.resnet50(pretrained=True)
        
        # Shared convolutional layers (full ResNet50 backbone)
        self.shared_conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.shared_layer1 = resnet.layer1   # 64 -> 256
        self.shared_layer2 = resnet.layer2   # 256 -> 512  
        self.shared_layer3 = resnet.layer3   # 512 -> 1024
        self.shared_layer4 = resnet.layer4   # 1024 -> 2048
        
        # Adaptive pooling to fixed size
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Projection to shared feature space
        self.shared_projection = nn.Sequential(
            nn.Linear(2048, shared_feature_dim),  # FIXED: 2048 -> shared_feature_dim
            nn.LayerNorm(shared_feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        """Extract shared features from input image"""
        x = self.shared_conv1(x)
        x = self.shared_layer1(x)
        x = self.shared_layer2(x)
        x = self.shared_layer3(x)  # Added layer3
        x = self.shared_layer4(x)  # Added layer4
        
        x = self.global_pool(x).flatten(1)  # [B, 2048] - FIXED
        shared_features = self.shared_projection(x)  # [B, shared_feature_dim]
        
        return shared_features

# ==================== MODALITY-SPECIFIC ENCODERS ====================

class ModalitySpecificEncoder(nn.Module):
    """Modality-specific encoder that processes shared features"""
    
    def __init__(self, shared_dim, modality_dim, dropout=0.1):
        super().__init__()
        
        self.modality_encoder = nn.Sequential(
            nn.Linear(shared_dim, modality_dim * 2),
            nn.LayerNorm(modality_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(modality_dim * 2, modality_dim),
            nn.LayerNorm(modality_dim),
            nn.ReLU()
        )
        
        # Residual connection
        if shared_dim == modality_dim:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Linear(shared_dim, modality_dim)
        
    def forward(self, shared_features):
        """Process shared features into modality-specific representation"""
        modality_features = self.modality_encoder(shared_features)
        residual = self.residual(shared_features)
        
        return modality_features + residual

# ==================== ENTANGLED FUSION MODULE ====================

class EntangledFusionModule(nn.Module):
    """Core entanglement module that fuses RNFL and VF representations"""
    
    def __init__(self, config):
        super().__init__()
        
        self.shared_dim = config.shared_feature_dim
        self.modality_dim = config.modality_feature_dim
        self.final_dim = config.final_embedding_dim
        
        # Cross-attention for RNFL attending to VF
        self.rnfl_to_vf_attention = CrossModalAttention(
            self.modality_dim, config.num_attention_heads
        )
        
        # Cross-attention for VF attending to RNFL  
        self.vf_to_rnfl_attention = CrossModalAttention(
            self.modality_dim, config.num_attention_heads
        )
        
        # FiLM layers for feature modulation
        self.vf_film_rnfl = FiLMLayer(
            self.modality_dim, self.modality_dim, config.film_hidden_dim
        )
        
        self.rnfl_film_vf = FiLMLayer(
            self.modality_dim, self.modality_dim, config.film_hidden_dim
        )
        
        # Joint fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.modality_dim * 2, self.final_dim),
            nn.LayerNorm(self.final_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.final_dim, self.final_dim)
        )
        
        # Separate heads for contrastive learning
        self.rnfl_head = nn.Sequential(
            nn.Linear(self.final_dim, self.final_dim),
            nn.ReLU(),
            nn.Linear(self.final_dim, self.final_dim)
        )
        
        self.vf_head = nn.Sequential(
            nn.Linear(self.final_dim, self.final_dim),
            nn.ReLU(), 
            nn.Linear(self.final_dim, self.final_dim)
        )
        
    def forward(self, rnfl_features, vf_features):
        """
        Entangled fusion of RNFL and VF features
        
        Args:
            rnfl_features: [B, modality_dim]
            vf_features: [B, modality_dim]
            
        Returns:
            Dict with entangled representations
        """
        
        # Cross-attention: each modality attends to the other
        rnfl_attended, rnfl_attn_weights = self.rnfl_to_vf_attention(
            rnfl_features, vf_features, vf_features
        )
        
        vf_attended, vf_attn_weights = self.vf_to_rnfl_attention(
            vf_features, rnfl_features, rnfl_features
        )
        
        # Feature modulation via FiLM
        rnfl_modulated = self.vf_film_rnfl(rnfl_attended, vf_features)
        vf_modulated = self.rnfl_film_vf(vf_attended, rnfl_features)
        
        # Joint representation
        joint_features = torch.cat([rnfl_modulated, vf_modulated], dim=-1)
        joint_embedding = self.fusion_layer(joint_features)
        
        # Modality-specific heads from joint embedding
        rnfl_final = self.rnfl_head(joint_embedding)
        vf_final = self.vf_head(joint_embedding)
        
        return {
            'joint_embedding': joint_embedding,
            'rnfl_embedding': rnfl_final,
            'vf_embedding': vf_final,
            'rnfl_attended': rnfl_attended,
            'vf_attended': vf_attended,
            'rnfl_modulated': rnfl_modulated,
            'vf_modulated': vf_modulated,
            'rnfl_attention': rnfl_attn_weights,
            'vf_attention': vf_attn_weights
        }

# ==================== ENTANGLED VF-RNFL MODEL ====================

class EntangledVFRNFLModel(nn.Module):
    """Complete entangled VF-RNFL contrastive learning model"""
    
    def __init__(self, config, num_classes):
        super().__init__()
        
        self.config = config
        self.num_classes = num_classes
        
        # Shared encoder backbone
        self.shared_encoder = SharedEncoderBackbone(config.shared_feature_dim)
        
        # Modality-specific encoders
        self.rnfl_encoder = ModalitySpecificEncoder(
            config.shared_feature_dim, config.modality_feature_dim
        )
        
        self.vf_encoder = ModalitySpecificEncoder(
            config.shared_feature_dim, config.modality_feature_dim
        )
        
        # Entangled fusion module
        self.fusion_module = EntangledFusionModule(config)
        
        # Learned null VF embedding for inference without VF
        self.null_vf_embedding = nn.Parameter(
            torch.randn(1, config.modality_feature_dim) * 0.02
        )
        
        # VF dropout probability for training robustness
        self.vf_dropout_prob = 0.1
        
        # Eye embedding
        self.eye_embedding = nn.Embedding(2, 16)
        
        # Context integration
        self.context_integration = nn.Sequential(
            nn.Linear(config.final_embedding_dim + 16, config.final_embedding_dim),
            nn.LayerNorm(config.final_embedding_dim),
            nn.ReLU()
        )
        
        # Classification heads
        self.joint_classifier = nn.Sequential(
            nn.Linear(config.final_embedding_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        self.rnfl_classifier = nn.Sequential(
            nn.Linear(config.final_embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # Tabular predictor
        self.tabular_predictor = nn.Sequential(
            nn.Linear(config.final_embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3)  # age, gender, md
        )
        
        # MoCo queue for entangled embeddings
        self.K = config.moco_k
        self.m = config.moco_m
        self.T = config.moco_t
        
        # Initialize momentum encoder (copy of main encoder)
        self.momentum_shared_encoder = SharedEncoderBackbone(config.shared_feature_dim)
        self.momentum_rnfl_encoder = ModalitySpecificEncoder(
            config.shared_feature_dim, config.modality_feature_dim
        )
        self.momentum_vf_encoder = ModalitySpecificEncoder(
            config.shared_feature_dim, config.modality_feature_dim
        )
        self.momentum_fusion_module = EntangledFusionModule(config)
        
        # Initialize momentum encoders
        self._init_momentum_encoders()
        
        # Queues
        self.register_buffer("queue", torch.randn(config.final_embedding_dim, self.K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
    def _init_momentum_encoders(self):
        """Initialize momentum encoders as copies of main encoders"""
        # Initialize momentum encoder (copy of main encoder)
        self.momentum_shared_encoder = SharedEncoderBackbone(self.config.shared_feature_dim)
        self.momentum_rnfl_encoder = ModalitySpecificEncoder(
            self.config.shared_feature_dim, self.config.modality_feature_dim
        )
        self.momentum_vf_encoder = ModalitySpecificEncoder(
            self.config.shared_feature_dim, self.config.modality_feature_dim
        )
        self.momentum_fusion_module = EntangledFusionModule(self.config)
        
        # Copy parameters from main to momentum encoders
        self._copy_main_to_momentum()
        
        # Disable gradients for momentum encoders
        for param in self._get_momentum_parameters():
            param.requires_grad = False
    
    def _copy_main_to_momentum(self):
        """Copy parameters from main encoders to momentum encoders"""
        main_params = self._get_main_encoder_parameters()
        momentum_params = self._get_momentum_parameters()
        
        for param_q, param_k in zip(main_params, momentum_params):
            param_k.data.copy_(param_q.data)
    
    def _get_main_encoder_parameters(self):
        """Get parameters of main encoders only (excludes classifiers)"""
        return (list(self.shared_encoder.parameters()) + 
                list(self.rnfl_encoder.parameters()) + 
                list(self.vf_encoder.parameters()) + 
                list(self.fusion_module.parameters()))
    
    def _get_momentum_parameters(self):
        """Get parameters of momentum encoder"""
        return (list(self.momentum_shared_encoder.parameters()) + 
                list(self.momentum_rnfl_encoder.parameters()) + 
                list(self.momentum_vf_encoder.parameters()) + 
                list(self.momentum_fusion_module.parameters()))
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder - FIXED to only update encoder params"""
        main_params = self._get_main_encoder_parameters()
        momentum_params = self._get_momentum_parameters()
        
        for param_q, param_k in zip(main_params, momentum_params):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update the queue with new keys"""
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0, "Queue size should be divisible by batch size"
        
        # Replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        
        self.queue_ptr[0] = ptr
    
    def forward_entangled(self, rnfl_img, vf_img, eye_labels=None, momentum=False, use_null_vf=False):
        """Forward pass through entangled architecture with proper VF handling"""
        
        # Choose encoder (main or momentum)
        if momentum:
            shared_encoder = self.momentum_shared_encoder
            rnfl_encoder = self.momentum_rnfl_encoder  
            vf_encoder = self.momentum_vf_encoder
            fusion_module = self.momentum_fusion_module
        else:
            shared_encoder = self.shared_encoder
            rnfl_encoder = self.rnfl_encoder
            vf_encoder = self.vf_encoder
            fusion_module = self.fusion_module
        
        # Extract shared features for RNFL
        rnfl_shared = shared_encoder(rnfl_img)
        rnfl_features = rnfl_encoder(rnfl_shared)
        
        # Handle VF features
        if use_null_vf:
            # Use learned null VF embedding for true RNFL-only inference
            batch_size = rnfl_img.size(0)
            vf_features = self.null_vf_embedding.expand(batch_size, -1)
        else:
            # Normal VF processing
            vf_shared = shared_encoder(vf_img)
            vf_features = vf_encoder(vf_shared)
            
            # Training-time VF dropout for robustness
            if self.training and torch.rand(1).item() < self.vf_dropout_prob:
                # Randomly replace with null VF embedding during training
                batch_size = vf_features.size(0)
                vf_features = self.null_vf_embedding.expand(batch_size, -1)
        
        # Entangled fusion
        fusion_output = fusion_module(rnfl_features, vf_features)
        
        # Eye context integration (only for main encoder)
        if eye_labels is not None and not momentum:
            eye_emb = self.eye_embedding(eye_labels)
            joint_embedding = torch.cat([fusion_output['joint_embedding'], eye_emb], dim=1)
            joint_embedding = self.context_integration(joint_embedding)
        else:
            joint_embedding = fusion_output['joint_embedding']
        
        fusion_output['joint_embedding'] = joint_embedding
        return fusion_output
    
    def forward(self, rnfl_img, vf_img=None, eye_labels=None, 
                return_logits=False, inference_mode=False):
        """
        Main forward pass with proper VF handling
        
        Args:
            rnfl_img: RNFL images [B, 3, H, W]
            vf_img: VF images [B, 3, H, W] (None for RNFL-only inference)
            eye_labels: Eye labels [B] (0=OD, 1=OS)
            return_logits: Whether to return classification logits
            inference_mode: Whether in inference mode
        """
        
        # Determine VF handling strategy
        if vf_img is None:
            # True RNFL-only inference using null VF embedding
            use_null_vf = True
            vf_img = rnfl_img  # Dummy input (won't be used due to use_null_vf=True)
        else:
            use_null_vf = False
        
        # Main encoder forward pass
        main_output = self.forward_entangled(rnfl_img, vf_img, eye_labels, 
                                           momentum=False, use_null_vf=use_null_vf)
        
        outputs = {
            'joint_embedding': main_output['joint_embedding'],
            'rnfl_embedding': main_output['rnfl_embedding'], 
            'vf_embedding': main_output['vf_embedding'],
            'attention_weights': {
                'rnfl_attention': main_output['rnfl_attention'],
                'vf_attention': main_output['vf_attention']
            }
        }
        
        if return_logits:
            # Classification
            outputs['joint_logits'] = self.joint_classifier(main_output['joint_embedding'])
            outputs['rnfl_logits'] = self.rnfl_classifier(main_output['rnfl_embedding'])
            
            # Tabular prediction
            outputs['tabular_pred'] = self.tabular_predictor(main_output['joint_embedding'])
        
        if inference_mode:
            return outputs
        
        # Training mode: momentum encoder + contrastive learning
        # Note: During training, we always have both RNFL and VF
        with torch.no_grad():
            self._momentum_update_key_encoder()
            momentum_output = self.forward_entangled(rnfl_img, vf_img, eye_labels, 
                                                   momentum=True, use_null_vf=use_null_vf)
        
        # Contrastive learning setup
        q = F.normalize(main_output['joint_embedding'], dim=1)
        k = F.normalize(momentum_output['joint_embedding'], dim=1)
        
        # Positive pairs
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        # Negative pairs
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        # Logits
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
        
        outputs.update({
            'moco_logits': logits,
            'moco_labels': labels,
            'momentum_joint': momentum_output['joint_embedding']
        })
        
        # Update queue
        self._dequeue_and_enqueue(k)
        
        return outputs

# ==================== ENTANGLEMENT LOSS FUNCTIONS ====================

class EntanglementLoss(nn.Module):
    """Cross-modal consistency (Eq. 3) and diversity (Eq. 4) losses.

    Returned separately so each can carry its own weight (λ2, λ3 in Eq. 6).
    """

    def __init__(self, collapse_margin=0.5):
        super().__init__()
        self.collapse_margin = collapse_margin

    def forward(self, rnfl_emb, vf_emb, joint_emb):
        """
        Returns:
            cross_loss: L_cross = 1 - 0.5 * [cos(z_joint, z_RNFL) + cos(z_joint, z_VF)]
            div_loss:   L_div   = max(0, mean_i cos(z_RNFL^i, z_VF^i) - 0.5)
        """

        # Normalize embeddings
        rnfl_norm = F.normalize(rnfl_emb, dim=1)
        vf_norm = F.normalize(vf_emb, dim=1)
        joint_norm = F.normalize(joint_emb, dim=1)

        # Eq. 3: joint embedding should align with both modalities
        joint_rnfl_sim = F.cosine_similarity(joint_norm, rnfl_norm, dim=1)
        joint_vf_sim = F.cosine_similarity(joint_norm, vf_norm, dim=1)
        cross_loss = 1 - (joint_rnfl_sim.mean() + joint_vf_sim.mean()) / 2

        # Eq. 4: modalities must remain distinguishable (prevent collapse)
        rnfl_vf_sim = F.cosine_similarity(rnfl_norm, vf_norm, dim=1)
        div_loss = torch.relu(rnfl_vf_sim.mean() - self.collapse_margin)

        return cross_loss, div_loss

class CrossModalAttentionLoss(nn.Module):
    """Loss to encourage meaningful cross-modal attention"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, rnfl_attention, vf_attention):
        """
        Encourage attention weights to be diverse and meaningful
        """
        
        # Attention should not be uniform (entropy regularization)
        rnfl_entropy = -torch.sum(rnfl_attention * torch.log(rnfl_attention + 1e-8), dim=-1)
        vf_entropy = -torch.sum(vf_attention * torch.log(vf_attention + 1e-8), dim=-1)
        
        # Penalize uniform attention (high entropy)
        max_entropy = torch.log(torch.tensor(rnfl_attention.size(-1), dtype=torch.float))
        
        entropy_loss = (torch.relu(rnfl_entropy - max_entropy * 0.8).mean() + 
                       torch.relu(vf_entropy - max_entropy * 0.8).mean()) / 2
        
        return entropy_loss

# ==================== DATA LOADING ====================

def parse_filename(filename):
    basename = os.path.splitext(os.path.basename(filename))[0]
    parts = basename.split('_')
    if len(parts) != 4:
        raise ValueError(f"Invalid filename")
    id_, eye, date, index = parts
    if eye not in ['OD', 'OS']:
        raise ValueError(f"Invalid eye")
    return id_, eye, date, index

def load_matched_data_with_labels(config):
    """Load matched dataset with ensemble progression labels"""
    df = pd.read_csv(config.labels_csv)
    
    print(f"\nLoaded ensemble progression labels: {len(df)} samples")
    
    # Create lookup dictionary
    label_dict = {}
    
    for _, row in df.iterrows():
        key = (str(row['ID']), row['Eye'], row['Date'], str(row['index']))
        
        label_dict[key] = {
            'progression': row['Progression_Ensemble'],
            'age': 0,
            'gender': 0, 
            'md': 0
        }
    
    # Binary label encoding
    unique_labels = sorted(df['Progression_Ensemble'].unique())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    print(f"\nFound {len(unique_labels)} unique classes (binary progression):")
    for label, idx in label_to_idx.items():
        count = (df['Progression_Ensemble'] == label).sum()
        label_name = "Not Worsening" if label == 0 else "Worsening"
        print(f"  {label_name} ({label}): {count} samples (class {idx})")
    
    return label_dict, label_to_idx, len(unique_labels)

def prepare_paired_data(eye_selection='both'):
    """Prepare paired RNFL-VF dataset"""
    print(f"\nPreparing paired RNFL-VF data...")
    print(f"Eye selection: {eye_selection.upper()}")
    
    rnfl_files = (glob(os.path.join(config.rnfl_dir, '*.jpg')) + 
                  glob(os.path.join(config.rnfl_dir, '*.jpeg')))
    rnfl_metadata = []
    
    for f in rnfl_files:
        try:
            id_, eye, date, index = parse_filename(f)
            rnfl_metadata.append({
                'ID': id_, 'Eye': eye, 'Date': date, 
                'Index': index, 'RNFL_path': f
            })
        except:
            continue
    
    vf_files = (glob(os.path.join(config.vf_dir, '*.jpg')) + 
                glob(os.path.join(config.vf_dir, '*.jpeg')))
    vf_metadata = []
    
    for f in vf_files:
        try:
            id_, eye, date, index = parse_filename(f)
            vf_metadata.append({
                'ID': id_, 'Eye': eye, 'Date': date, 
                'Index': index, 'VF_path': f
            })
        except:
            continue
    
    rnfl_df = pd.DataFrame(rnfl_metadata)
    vf_df = pd.DataFrame(vf_metadata)
    
    print(f"  Found {len(rnfl_df):,} total RNFL images")
    print(f"  Found {len(vf_df):,} total VF images")
    
    # Filter by eye
    if eye_selection in ['OD', 'OS']:
        rnfl_df = rnfl_df[rnfl_df['Eye'] == eye_selection]
        vf_df = vf_df[vf_df['Eye'] == eye_selection]
        print(f"  Filtered to {eye_selection} eyes only")
    
    rnfl_df['ID'] = rnfl_df['ID'].astype(str)
    vf_df['ID'] = vf_df['ID'].astype(str)
    
    paired_df = pd.merge(rnfl_df, vf_df, on=['ID', 'Eye', 'Date', 'Index'])
    
    print(f"  Found {len(paired_df):,} matched pairs")
    
    le = LabelEncoder()
    paired_df['PatientID_encoded'] = le.fit_transform(paired_df['ID'])
    
    return paired_df

# ==================== DATASET ====================

class EntangledDataset(Dataset):
    """Dataset for entangled VF-RNFL learning"""
    
    def __init__(self, paired_df, label_dict, label_to_idx, transform=None):
        self.paired_df = paired_df
        self.label_dict = label_dict
        self.label_to_idx = label_to_idx
        self.transform = transform
        
    def __len__(self):
        return len(self.paired_df)
    
    def __getitem__(self, idx):
        row = self.paired_df.iloc[idx]
        
        rnfl_img = Image.open(row['RNFL_path']).convert('RGB')
        vf_img = Image.open(row['VF_path']).convert('RGB')
        
        if self.transform:
            rnfl_img = self.transform(rnfl_img)
            vf_img = self.transform(vf_img)
        else:
            rnfl_img = transforms.ToTensor()(rnfl_img)
            vf_img = transforms.ToTensor()(vf_img)
        
        # Get labels
        key = (str(row['ID']), row['Eye'], row['Date'], str(row['Index']))
        
        if key in self.label_dict:
            labels = self.label_dict[key]
            class_label = self.label_to_idx[labels['progression']]
            
            tabular = torch.tensor([
                labels['age'] / 100.0,
                labels['gender'],
                labels['md'] / 30.0
            ], dtype=torch.float32)
        else:
            class_label = -1
            tabular = torch.zeros(3, dtype=torch.float32)
        
        eye_label = torch.tensor(0 if row['Eye'] == 'OD' else 1, dtype=torch.long)
        
        return {
            'rnfl_img': rnfl_img,
            'vf_img': vf_img,
            'eye_label': eye_label,
            'class_label': torch.tensor(class_label, dtype=torch.long),
            'tabular': tabular,
            'has_label': class_label >= 0
        }

# ==================== TRAINING ====================

def get_medical_transforms(training=True):
    if training:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def attach_scan_labels(paired_df, label_dict, label_to_idx):
    """Add a per-scan class label column (-1 = unlabeled) used for stratification"""
    labels = []
    for _, row in paired_df.iterrows():
        key = (str(row['ID']), row['Eye'], row['Date'], str(row['Index']))
        if key in label_dict:
            labels.append(label_to_idx[label_dict[key]['progression']])
        else:
            labels.append(-1)
    paired_df = paired_df.copy()
    paired_df['class_label'] = labels
    return paired_df


def _patient_level_split(df, test_fraction, seed):
    """Split a dataframe into (rest, held-out) at the patient level, stratified
    by each patient's majority progression label (unlabeled = own stratum)."""
    patients = df.groupby('ID')['class_label'] \
                 .agg(lambda s: int(s.mode().iloc[0])) \
                 .reset_index()
    patients = patients.rename(columns={'class_label': 'patient_label'})
    # Rows of `patients` are patient-level; groups are trivially the patient IDs
    sgkf = StratifiedGroupKFold(n_splits=int(round(1 / test_fraction)),
                                shuffle=True, random_state=seed)
    rest_idx, held_idx = next(sgkf.split(
        patients, patients['patient_label'], groups=patients['ID']))
    rest_ids = set(patients.iloc[rest_idx]['ID'])
    held_ids = set(patients.iloc[held_idx]['ID'])
    return df[df['ID'].isin(rest_ids)], df[df['ID'].isin(held_ids)]


def make_patient_level_folds(paired_df, config):
    """Patient-level, label-stratified K-fold assignment.

    Every scan from a patient (both eyes, all visits) is assigned to exactly one
    fold, so no patient is shared between training/validation and test sets.
    Stratification uses the patient's majority progression label so each fold
    carries a similar worsening / not-worsening proportion.

    Returns a copy of paired_df with a 'fold' column in {0, ..., K-1}.
    """
    patients = paired_df.groupby('ID')['class_label'] \
                        .agg(lambda s: int(s.mode().iloc[0])) \
                        .reset_index() \
                        .rename(columns={'class_label': 'patient_label'})

    sgkf = StratifiedGroupKFold(n_splits=config.n_folds, shuffle=True,
                                random_state=config.seed)
    fold_of_patient = {}
    for fold, (_, held_idx) in enumerate(sgkf.split(
            patients, patients['patient_label'], groups=patients['ID'])):
        for pid in patients.iloc[held_idx]['ID']:
            fold_of_patient[pid] = fold

    paired_df = paired_df.copy()
    paired_df['fold'] = paired_df['ID'].map(fold_of_patient).astype(int)

    # Sanity check: a patient must never appear in more than one fold
    assert (paired_df.groupby('ID')['fold'].nunique() == 1).all(), \
        "Patient-level leakage detected across folds"

    print(f"\nPatient-level {config.n_folds}-fold assignment:")
    for fold in range(config.n_folds):
        sub = paired_df[paired_df['fold'] == fold]
        n_lab = (sub['class_label'] >= 0).sum()
        n_pos = (sub['class_label'] == 1).sum()
        print(f"  Fold {fold}: {sub['ID'].nunique():>5} patients, "
              f"{sub[['ID', 'Eye']].drop_duplicates().shape[0]:>5} eyes, "
              f"{len(sub):>5} scans "
              f"(labeled {n_lab}, worsening {n_pos})")
    return paired_df


def create_fold_loaders(paired_df, fold, label_dict, label_to_idx, config):
    """Build train / val / test loaders for one CV fold.

    test  = patients assigned to `fold`                        (~20% of patients)
    train = remaining patients minus the validation patients   (~70%)
    val   = `config.val_fraction` of the remaining patients    (~10%), patient-level
    """
    test_df = paired_df[paired_df['fold'] == fold]
    trainval_df = paired_df[paired_df['fold'] != fold]
    train_df, val_df = _patient_level_split(
        trainval_df, config.val_fraction, seed=config.seed + fold)

    # No patient may cross the three partitions
    assert not (set(train_df['ID']) & set(val_df['ID']))
    assert not (set(train_df['ID']) & set(test_df['ID']))
    assert not (set(val_df['ID']) & set(test_df['ID']))

    print(f"\n[Fold {fold}] split (patients / eyes / scans):")
    for name, d in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        print(f"  {name:<5}: {d['ID'].nunique():>5} / "
              f"{d[['ID', 'Eye']].drop_duplicates().shape[0]:>5} / {len(d):>5}")

    train_transform = get_medical_transforms(training=True)
    val_transform = get_medical_transforms(training=False)

    train_dataset = EntangledDataset(train_df, label_dict, label_to_idx, train_transform)
    val_dataset = EntangledDataset(val_df, label_dict, label_to_idx, val_transform)
    test_dataset = EntangledDataset(test_df, label_dict, label_to_idx, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                            shuffle=True, num_workers=config.num_workers,
                            pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size * 2,
                          shuffle=False, num_workers=config.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size * 2,
                           shuffle=False, num_workers=config.num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, test_df

def train_entangled_model(model, train_loader, val_loader, config, num_classes, fold=0):
    """Training loop for entangled model (one CV fold)"""

    device = torch.device(config.device)
    model = model.to(device)
    
    # Loss functions
    criterion_moco = nn.CrossEntropyLoss()
    criterion_entanglement = EntanglementLoss()
    criterion_attention = CrossModalAttentionLoss()
    
    # Supervised losses
    class_weights = torch.tensor([1.0, 1.0], dtype=torch.float32).to(device)
    criterion_classification = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    criterion_tabular = nn.MSELoss()
    
    # Optimizer with different learning rates
    param_groups = [
        {'params': model.shared_encoder.parameters(), 'lr': config.base_lr},
        {'params': model.rnfl_encoder.parameters(), 'lr': config.base_lr},
        {'params': model.vf_encoder.parameters(), 'lr': config.base_lr},
        {'params': model.fusion_module.parameters(), 'lr': config.base_lr * 2},
        {'params': model.joint_classifier.parameters(), 'lr': config.base_lr * 2},
        {'params': model.rnfl_classifier.parameters(), 'lr': config.base_lr * 2}
    ]
    
    optimizer = optim.AdamW(param_groups, weight_decay=config.weight_decay)
    
    def lr_lambda(epoch):
        if epoch < config.warmup_epochs:
            return (epoch + 1) / config.warmup_epochs
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - config.warmup_epochs) / 
                                    (config.num_epochs - config.warmup_epochs)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()
    
    best_val_metric = float('inf')
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch': []
    }
    
    print("\n" + "="*70)
    print(f"ENTANGLED VF-RNFL TRAINING — FOLD {fold}")
    print("="*70)

    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        progress_bar = tqdm(train_loader, desc=f"[Fold {fold}] Epoch {epoch+1}/{config.num_epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            rnfl_img = batch['rnfl_img'].to(device)
            vf_img = batch['vf_img'].to(device)
            eye_labels = batch['eye_label'].to(device)
            class_labels = batch['class_label'].to(device)
            tabular = batch['tabular'].to(device)
            has_label = batch['has_label'].to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(rnfl_img, vf_img, eye_labels, return_logits=True)

                # Eq. 2 — MoCo contrastive loss
                loss_moco = criterion_moco(outputs['moco_logits'], outputs['moco_labels'])

                # Eq. 3 / Eq. 4 — cross-modal consistency and diversity
                loss_cross, loss_div = criterion_entanglement(
                    outputs['rnfl_embedding'],
                    outputs['vf_embedding'],
                    outputs['joint_embedding']
                )

                # Eq. 5 — progression classification (joint + RNFL heads)
                loss_joint_cls = criterion_classification(outputs['joint_logits'], class_labels)
                loss_rnfl_cls = criterion_classification(outputs['rnfl_logits'], class_labels)
                loss_prog = (loss_joint_cls + loss_rnfl_cls) / 2

                # Eq. 6 — total loss
                loss = (config.lambda_moco * loss_moco +
                        config.lambda_cross * loss_cross +
                        config.lambda_div * loss_div +
                        config.lambda_prog * loss_prog)

                # Optional auxiliary terms (off by default, see Config)
                if config.lambda_attention > 0:
                    loss_attention = criterion_attention(
                        outputs['attention_weights']['rnfl_attention'],
                        outputs['attention_weights']['vf_attention']
                    )
                    loss = loss + config.lambda_attention * loss_attention
                if config.lambda_tabular > 0:
                    loss_tabular = criterion_tabular(outputs['tabular_pred'], tabular)
                    loss = loss + config.lambda_tabular * loss_tabular
                
                # Track accuracy
                with torch.no_grad():
                    preds = outputs['joint_logits'].argmax(dim=1)
                    correct = (preds == class_labels) & has_label
                    train_correct += correct.sum().item()
                    train_total += has_label.sum().item()
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{train_correct/(train_total+1e-8):.3f}"
            })
        
        n_batches = len(train_loader)
        avg_train_loss = train_loss / n_batches
        avg_train_acc = train_correct / (train_total + 1e-8)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                rnfl_img = batch['rnfl_img'].to(device)
                vf_img = batch['vf_img'].to(device)
                eye_labels = batch['eye_label'].to(device)
                class_labels = batch['class_label'].to(device)
                has_label = batch['has_label'].to(device)

                # inference_mode=True: no momentum update / queue writes on validation data
                outputs = model(rnfl_img, vf_img, eye_labels,
                                return_logits=True, inference_mode=True)

                loss = criterion_classification(outputs['joint_logits'], class_labels)
                
                preds = outputs['joint_logits'].argmax(dim=1)
                correct = (preds == class_labels) & has_label
                val_correct += correct.sum().item()
                val_total += has_label.sum().item()
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_correct / (val_total + 1e-8)
        
        scheduler.step()
        
        # Store history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        
        print(f"\n[Fold {fold}] Epoch {epoch+1}/{config.num_epochs}")
        print(f"  Train - Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.3f}")
        print(f"  Val   - Loss: {avg_val_loss:.4f}, Acc: {avg_val_acc:.3f}")

        combined_metric = avg_val_loss - 0.2 * avg_val_acc

        if combined_metric < best_val_metric:
            best_val_metric = combined_metric
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'fold': fold,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_acc': avg_val_acc,
                'config': {k: v for k, v in vars(Config).items() if not k.startswith('_')},
                'history': history
            }, checkpoint_path(config, fold))

            print(f"  ✓ Saved best model: {os.path.basename(checkpoint_path(config, fold))}")
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            print(f"\n[Fold {fold}] Early stopping at epoch {epoch+1}")
            break

    return model, history


def checkpoint_path(config, fold):
    eye_suffix = config.eye_selection.lower()
    return os.path.join(config.model_dir,
                        f'best_entangled_vf_rnfl_model_{eye_suffix}_fold{fold}.pt')


@torch.no_grad()
def evaluate_fold(model, test_loader, config):
    """Held-out test metrics for one fold (labeled scans only)."""
    device = torch.device(config.device)
    model.eval()
    probs, labels = [], []
    for batch in tqdm(test_loader, desc="Testing", leave=False):
        outputs = model(batch['rnfl_img'].to(device), batch['vf_img'].to(device),
                        batch['eye_label'].to(device),
                        return_logits=True, inference_mode=True)
        p = torch.softmax(outputs['joint_logits'].float(), dim=1)[:, 1]
        probs.append(p.cpu().numpy())
        labels.append(batch['class_label'].numpy())
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    mask = labels >= 0
    metrics = {'n_test_scans': int(len(labels)), 'n_test_labeled': int(mask.sum())}
    if mask.sum() > 0:
        metrics['test_acc'] = float(((probs[mask] > 0.5).astype(int) == labels[mask]).mean())
        if len(np.unique(labels[mask])) == 2:
            metrics['test_auroc'] = float(roc_auc_score(labels[mask], probs[mask]))
    return metrics


@torch.no_grad()
def extract_embeddings(model, loader, config):
    """256-D embeddings for every scan in `loader`.

    Returns the VF-guided embeddings (joint / rnfl / vf) and, in addition, the
    OCT-only embeddings obtained by replacing the VF input with the learned null
    embedding — the path used for external cohorts without perimetry.
    """
    device = torch.device(config.device)
    model.eval()
    out = {k: [] for k in ['joint', 'rnfl', 'vf', 'joint_octonly', 'rnfl_octonly']}
    for batch in tqdm(loader, desc="Extracting embeddings", leave=False):
        rnfl_img = batch['rnfl_img'].to(device)
        vf_img = batch['vf_img'].to(device)
        eye_labels = batch['eye_label'].to(device)

        o = model(rnfl_img, vf_img, eye_labels, inference_mode=True)
        out['joint'].append(o['joint_embedding'].cpu().numpy())
        out['rnfl'].append(o['rnfl_embedding'].cpu().numpy())
        out['vf'].append(o['vf_embedding'].cpu().numpy())

        o = model(rnfl_img, None, eye_labels, inference_mode=True)  # null-VF path
        out['joint_octonly'].append(o['joint_embedding'].cpu().numpy())
        out['rnfl_octonly'].append(o['rnfl_embedding'].cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}

# ==================== MAIN ====================

def main():
    print_banner()
    
    # Load labels
    label_dict, label_to_idx, num_classes = load_matched_data_with_labels(config)
    
    # Prepare data
    paired_df = prepare_paired_data(eye_selection=config.eye_selection)
    
    if len(paired_df) == 0:
        print("ERROR: No paired data found. Exiting.")
        return
    
    # Patient-level, label-stratified K-fold assignment
    paired_df = attach_scan_labels(paired_df, label_dict, label_to_idx)
    paired_df = make_patient_level_folds(paired_df, config)
    paired_df = paired_df.reset_index(drop=True)
    
    eye_suffix = config.eye_selection.lower()
    paired_df[['ID', 'Eye', 'Date', 'Index', 'class_label', 'fold']].to_csv(
        os.path.join(config.output_dir, f'cv_fold_assignment_{eye_suffix}.csv'), index=False)
    
    folds = config.folds_to_run if config.folds_to_run is not None else list(range(config.n_folds))
    
    # Out-of-fold embedding buffers: one row per scan, filled by the fold that
    # holds that scan out. Together they cover every scan exactly once.
    n_scans = len(paired_df)
    oof = {k: np.full((n_scans, config.final_embedding_dim), np.nan, dtype=np.float32)
           for k in ['joint', 'rnfl', 'vf', 'joint_octonly', 'rnfl_octonly']}
    
    cv_summary = {'config': {k: v for k, v in vars(Config).items() if not k.startswith('_')},
                  'folds': {}}
    
    for fold in folds:
        print("\n" + "#"*70)
        print(f"# FOLD {fold + 1} / {config.n_folds}")
        print("#"*70)
        
        # Fresh seed per fold for reproducibility
        torch.manual_seed(config.seed + fold)
        np.random.seed(config.seed + fold)
        random.seed(config.seed + fold)
        
        train_loader, val_loader, test_loader, test_df = create_fold_loaders(
            paired_df, fold, label_dict, label_to_idx, config
        )
        
        # Fresh model for every fold
        model = EntangledVFRNFLModel(config, num_classes)
        if fold == folds[0]:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\nEntangled Model Summary:")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            print(f"  Shared encoder dim: {config.shared_feature_dim}")
            print(f"  Modality-specific dim: {config.modality_feature_dim}")
            print(f"  Final embedding dim: {config.final_embedding_dim}")
            print(f"  Cross-attention heads: {config.num_attention_heads}")
        
        # Train
        model, history = train_entangled_model(
            model, train_loader, val_loader, config, num_classes, fold=fold
        )
        
        # Reload the best (early-stopped) checkpoint of this fold before testing
        ckpt = torch.load(checkpoint_path(config, fold), map_location=config.device,
                          weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(torch.device(config.device))
        
        # Held-out test metrics
        metrics = evaluate_fold(model, test_loader, config)
        metrics.update({
            'best_epoch': int(ckpt['epoch']) + 1,
            'best_val_loss': float(ckpt['val_loss']),
            'best_val_acc': float(ckpt['val_acc']),
            'n_test_patients': int(test_df['ID'].nunique()),
            'n_test_eyes': int(test_df[['ID', 'Eye']].drop_duplicates().shape[0]),
        })
        cv_summary['folds'][fold] = metrics
        print(f"\n[Fold {fold}] test: " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                                   for k, v in metrics.items()))
        
        # Held-out embeddings for this fold (test_loader preserves test_df order)
        emb = extract_embeddings(model, test_loader, config)
        test_idx = test_df.index.values
        for k in oof:
            oof[k][test_idx] = emb[k]
        for k in ['joint', 'rnfl', 'vf', 'joint_octonly', 'rnfl_octonly']:
            np.save(os.path.join(config.output_dir,
                                 f'entangled_{k}_embeddings_{eye_suffix}_fold{fold}.npy'), emb[k])
        test_df[['ID', 'Eye', 'Date', 'Index', 'class_label', 'fold']].to_csv(
            os.path.join(config.output_dir, f'test_metadata_{eye_suffix}_fold{fold}.csv'), index=False)
        
        with open(os.path.join(config.output_dir, f'cv_summary_{eye_suffix}.json'), 'w') as f:
            json.dump(cv_summary, f, indent=2, default=str)
        
        del model, train_loader, val_loader, test_loader
        torch.cuda.empty_cache()
    
    # ==================== AGGREGATE ====================
    print("\n" + "="*70)
    print("CROSS-VALIDATION SUMMARY")
    print("="*70)
    for metric in ['test_acc', 'test_auroc']:
        vals = [m[metric] for m in cv_summary['folds'].values() if metric in m]
        if vals:
            print(f"  {metric}: {np.mean(vals):.4f} ± {np.std(vals):.4f}  "
                  f"(folds: {', '.join(f'{v:.3f}' for v in vals)})")
            cv_summary[f'{metric}_mean'] = float(np.mean(vals))
            cv_summary[f'{metric}_std'] = float(np.std(vals))
    
    with open(os.path.join(config.output_dir, f'cv_summary_{eye_suffix}.json'), 'w') as f:
        json.dump(cv_summary, f, indent=2, default=str)
    
    if len(folds) == config.n_folds:
        # Every scan was held out exactly once -> complete out-of-fold embedding set
        assert not np.isnan(oof['rnfl']).any(), "Out-of-fold embeddings incomplete"
        for k, arr in oof.items():
            np.save(os.path.join(config.output_dir,
                                 f'entangled_{k}_embeddings_{eye_suffix}_oof.npy'), arr)
        paired_df[['ID', 'Eye', 'Date', 'Index', 'class_label', 'fold']].to_csv(
            os.path.join(config.output_dir, f'oof_metadata_{eye_suffix}.csv'), index=False)
        print(f"\n✓ Out-of-fold embeddings saved ({n_scans} scans x {config.final_embedding_dim}D):")
        print(f"  entangled_{{joint,rnfl,vf}}_embeddings_{eye_suffix}_oof.npy        (VF-guided path)")
        print(f"  entangled_{{joint,rnfl}}_octonly_embeddings_{eye_suffix}_oof.npy   (null-VF / OCT-only path)")
        print(f"  oof_metadata_{eye_suffix}.csv                              (row order of the .npy files)")
    
    print("\n" + "="*70)
    print("ENTANGLED TRAINING COMPLETE!")
    print("="*70)
    print(f"\nKey features:")
    print(f"  ✓ Shared encoder backbone")
    print(f"  ✓ Cross-modal attention (RNFL ↔ VF)")
    print(f"  ✓ FiLM feature modulation")
    print(f"  ✓ Joint latent bottleneck")
    print(f"  ✓ Entangled contrastive learning (Eq. 6: λ = "
          f"{config.lambda_moco}, {config.lambda_cross}, {config.lambda_div}, {config.lambda_prog})")
    print(f"  ✓ Patient-level {config.n_folds}-fold cross-validation")
    
    return cv_summary

if __name__ == "__main__":
    main()
