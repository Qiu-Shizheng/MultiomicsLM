import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import logging
import random
import math
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.preprocessing import RobustScaler
from torch.cuda.amp import GradScaler, autocast


warnings.filterwarnings("ignore")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
RESULTS_DIR = 'your/path/results_multiomics_bert'
os.makedirs(RESULTS_DIR, exist_ok=True)
logging.info(f"All results will be saved to directory: {RESULTS_DIR}")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


set_seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda")
    gpu_count = torch.cuda.device_count()
    logging.info(f"Using device: {device}, GPU count: {gpu_count}")
else:
    device = torch.device("cpu")
    gpu_count = 0
    logging.info("Training on CPU")

#############################
# Data Preprocessing and Loading
#############################
logging.info("Loading list of healthy eids...")
healthy_eids = pd.read_csv('your/path/healthy_eids.csv')['eid'].tolist()

logging.info("Loading protein expression data...")
expression_data = pd.read_csv('your/path/proteomic.csv')
logging.info("Loading metabolite expression data...")
metabolite_data = pd.read_csv('your/path/metabolomic.csv')

logging.info("Filtering healthy population data...")
expression_data = expression_data[expression_data['eid'].isin(healthy_eids)]
metabolite_data = metabolite_data[metabolite_data['eid'].isin(healthy_eids)]

# Select samples with both protein and metabolite data
common_eids = sorted(list(set(expression_data['eid']) & set(metabolite_data['eid'])))
logging.info(f"Total number of participants: {len(common_eids)}")
expression_data = expression_data[expression_data['eid'].isin(common_eids)]
metabolite_data = metabolite_data[metabolite_data['eid'].isin(common_eids)]

if expression_data['eid'].duplicated().any() or metabolite_data['eid'].duplicated().any():
    logging.error("Duplicate eids found. Please check the dataset.")
    raise ValueError("Duplicate eids exist!")

expression_data.set_index('eid', inplace=True)
metabolite_data.set_index('eid', inplace=True)
logging.info(f"Number of protein samples: {expression_data.shape[0]}")
logging.info(f"Number of metabolite samples: {metabolite_data.shape[0]}")

# -------------------------------
# Load Global Protein Representations
# -------------------------------
logging.info("Loading protein sequence and functional features data...")
global_data = np.load('your/path/global_representations.npz')
global_keys = list(global_data.keys())

common_proteins = sorted(list(set(expression_data.columns) & set(global_keys)))
expression_data = expression_data[common_proteins]
logging.info(f"Number of proteins in filtered protein data: {len(expression_data.columns)}")

common_metabolites = sorted(list(metabolite_data.columns))
logging.info(f"Total number of metabolites: {len(common_metabolites)}")
metabolite_data = metabolite_data[common_metabolites]
logging.info(f"Number of metabolites in filtered metabolite data: {len(metabolite_data.columns)}")

logging.info("Normalizing protein and metabolite data (RobustScaler)...")
expression_data_scaled = expression_data.copy()
metabolite_data_scaled = metabolite_data.copy()
scaler_protein = RobustScaler()
scaler_metabolite = RobustScaler()
expression_data_scaled[common_proteins] = scaler_protein.fit_transform(expression_data_scaled[common_proteins])
metabolite_data_scaled[common_metabolites] = scaler_metabolite.fit_transform(metabolite_data_scaled[common_metabolites])

filtered_global_features = np.stack([global_data[p] for p in common_proteins], axis=0)


#############################
# Masking Function (80/10/10 strategy)
#############################
def mask_input(x, mask_prob=0.15):
    # x: [B, seq_len]
    mask = (torch.rand(x.shape, device=x.device) < mask_prob)
    rp = torch.rand(x.shape, device=x.device)
    mask_80 = mask & (rp < 0.8)  # 80% set to 0
    mask_10 = mask & (rp >= 0.8) & (rp < 0.9)  # 10% replaced with random
    x_masked = x.clone()
    x_masked[mask_80] = 0.0
    if mask_10.sum() > 0:
        x_masked[mask_10] = torch.randn_like(x_masked[mask_10])
    return x_masked, mask


#############################
# Transformer Encoder
#############################
class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, activation="gelu"):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None, return_attn=False):
        attn_output, attn_weights = self.self_attn(src, src, src, need_weights=True)
        src2 = self.dropout1(attn_output)
        src = self.norm1(src + src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src2 = self.dropout2(src2)
        src = self.norm2(src + src2)
        if return_attn:
            return src, attn_weights
        return src


class CustomTransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dropout=0.1, activation="gelu"):
        super(CustomTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model, nhead, dropout, activation) for _ in range(num_layers)
        ])

    def forward(self, src, return_all_attn=False):
        attn_all = []
        for layer in self.layers:
            if return_all_attn:
                src, attn = layer(src, return_attn=True)
                attn_all.append(attn)
            else:
                src = layer(src)
        if return_all_attn:
            return src, attn_all
        return src


#############################
# BERT Encoder
#############################
class ProteinBERTEncoder(nn.Module):
    def __init__(self, seq_len, hidden_size, num_layers=4, num_heads=8, dropout=0.1):
        super(ProteinBERTEncoder, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, hidden_size))
        self.encoder = CustomTransformerEncoder(num_layers, hidden_size, num_heads, dropout, activation="gelu")

    def forward(self, x, return_attn=False):
        x = x + self.pos_embedding
        if return_attn:
            out, attn_all = self.encoder(x, return_all_attn=True)
            return out, attn_all
        return self.encoder(x)


class MetaboliteBERTEncoder(nn.Module):
    def __init__(self, seq_len, hidden_size, num_layers=4, num_heads=8, dropout=0.1):
        super(MetaboliteBERTEncoder, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, hidden_size))
        self.encoder = CustomTransformerEncoder(num_layers, hidden_size, num_heads, dropout, activation="gelu")

    def forward(self, x, return_attn=False):
        x = x + self.pos_embedding
        if return_attn:
            out, attn_all = self.encoder(x, return_all_attn=True)
            return out, attn_all
        return self.encoder(x)


#############################
# Gated Fusion Module (Fusing two global vectors)
#############################
class GatedFusion(nn.Module):
    def __init__(self, hidden_size):
        super(GatedFusion, self).__init__()
        self.gate_linear = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x, y):
        concat = torch.cat([x, y], dim=-1)
        gate = torch.sigmoid(self.gate_linear(concat))
        return gate * x + (1 - gate) * y


#############################
# Multi-Modal BERT Model
#############################
class MultiModalBERTModel_Modified(nn.Module):
    def __init__(self, hidden_size, num_proteins, num_metabolites, dropout_prob=0.3, global_features=None):
        super(MultiModalBERTModel_Modified, self).__init__()
        self.hidden_size = hidden_size
        self.num_proteins = num_proteins
        self.num_metabolites = num_metabolites

        # Protein branch: using BERT encoder
        self.protein_encoder = ProteinBERTEncoder(seq_len=num_proteins, hidden_size=hidden_size,
                                                  num_layers=4, num_heads=8, dropout=dropout_prob)
        # Metabolite branch: first project 1 dimension to hidden_size, then use BERT encoder
        self.metab_proj = nn.Linear(1, hidden_size)
        self.metabolite_encoder = MetaboliteBERTEncoder(seq_len=num_metabolites, hidden_size=hidden_size,
                                                        num_layers=4, num_heads=8, dropout=dropout_prob)
        # Global protein representation projection
        if global_features is None:
            raise ValueError("Global protein representations (global_features) must be provided")
        self.register_buffer('global_features', torch.tensor(global_features, dtype=torch.float32))
        self.global_proj = nn.Linear(self.global_features.shape[1], hidden_size)

        # Fusion module: gated fusion of protein and metabolite global features (mean pooling used)
        self.gated_fusion = GatedFusion(hidden_size)
        # ITM classification head
        self.itm_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size // 2, 1)
        )
        # MLM and reconstruction heads (token-level prediction)
        self.mlm_head_protein = nn.Linear(hidden_size, 1)
        self.mlm_head_metabolite = nn.Linear(hidden_size, 1)
        self.output_layer_protein = nn.Linear(hidden_size, 1)
        self.output_layer_metabolite = nn.Linear(hidden_size, 1)

    def forward(self, proteins, metabolites, itm_label=None, mask_proteins=None, mask_metabolites=None,
                return_attn=False, return_metab_features=False):
        B = proteins.size(0)
        # Protein branch: fuse with global representation
        projected_global = self.global_proj(self.global_features)  # [num_proteins, hidden_size]
        projected_global = projected_global.unsqueeze(0)  # [1, num_proteins, hidden_size]
        prot_inputs = proteins.unsqueeze(-1) * projected_global  # [B, num_proteins, hidden_size]
        if return_attn:
            prot_encoded, attn_all = self.protein_encoder(prot_inputs, return_attn=True)
            attn_weights = attn_all[-1]  # Last layer attention weights, shape [B, num_proteins, num_proteins]
        else:
            prot_encoded = self.protein_encoder(prot_inputs)
            attn_weights = None
        protein_global = prot_encoded.mean(dim=1)  # [B, hidden_size]

        # Metabolite branch
        metab_inputs = metabolites.unsqueeze(-1)  # [B, num_metabolites, 1]
        metab_inputs = self.metab_proj(metab_inputs)  # [B, num_metabolites, hidden_size]
        metab_encoded = self.metabolite_encoder(metab_inputs)
        metabolite_global = metab_encoded.mean(dim=1)  # [B, hidden_size]

        # Fuse global information
        final_rep = self.gated_fusion(protein_global, metabolite_global)  # [B, hidden_size]
        itm_logits = self.itm_classifier(final_rep).squeeze(-1)
        mlm_pred_protein = self.mlm_head_protein(prot_encoded).squeeze(-1)  # [B, num_proteins]
        mlm_pred_metabolite = self.mlm_head_metabolite(metab_encoded).squeeze(-1)  # [B, num_metabolites]
        recon_protein = self.output_layer_protein(prot_encoded).squeeze(-1)
        recon_metabolite = self.output_layer_metabolite(metab_encoded).squeeze(-1)

        outputs = {
            'prediction_protein': recon_protein,
            'prediction_metabolite': recon_metabolite,
            'itm_logits': itm_logits,
            'mlm_pred_protein': mlm_pred_protein,
            'mlm_pred_metabolite': mlm_pred_metabolite,
            'prot_global': protein_global,
            'metab_global': metabolite_global,
        }
        if itm_label is not None:
            outputs['itm_label'] = itm_label
        if mask_proteins is not None:
            outputs['mask_proteins'] = mask_proteins
        if mask_metabolites is not None:
            outputs['mask_metabolites'] = mask_metabolites
        if return_attn:
            outputs['attn_weights'] = attn_weights
        if return_metab_features:
            outputs['metab_features'] = metab_encoded
        return outputs

#############################
# Contrastive Loss Function (cosine similarity + margin)
#############################
def compute_contrastive_loss(prot, metab, labels, margin=0.5):
    prot_norm = F.normalize(prot, dim=1)
    metab_norm = F.normalize(metab, dim=1)
    cos_sim = (prot_norm * metab_norm).sum(dim=1)
    pos_loss = labels * (1 - cos_sim) ** 2
    neg_loss = (1 - labels) * F.relu(cos_sim - margin) ** 2
    loss = pos_loss + neg_loss
    return loss.mean()


#############################
# Dataset Class
#############################
class MultiOmicsDataset(Dataset):
    def __init__(self, protein_data, metabolite_data, negative_sampling_ratio=0.5):
        self.protein_data = protein_data.values
        self.metabolite_data = metabolite_data.values
        self.num_samples = protein_data.shape[0]
        self.negative_sampling_ratio = negative_sampling_ratio
        self.pairs = []
        self.labels = []
        for idx in range(self.num_samples):
            # Positive sample
            self.pairs.append((idx, idx))
            self.labels.append(1)
            if self.negative_sampling_ratio > 0:
                # Random negative sample
                neg_idx = random.randint(0, self.num_samples - 1)
                while neg_idx == idx:
                    neg_idx = random.randint(0, self.num_samples - 1)
                self.pairs.append((idx, neg_idx))
                self.labels.append(0)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p_idx, m_idx = self.pairs[idx]
        proteins = torch.tensor(self.protein_data[p_idx], dtype=torch.float32)
        metabolites = torch.tensor(self.metabolite_data[m_idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return proteins, metabolites, label


#############################
# Parameter Settings
#############################
num_epochs = 50
batch_size = 52
hidden_size = 768
learning_rate = 2e-5
weight_decay = 1e-3
dropout_prob = 0.3
lambda_contrast = 0.3
negative_sampling_ratio = 0.5

# Build full dataset and split into training and validation sets (80/20)
full_dataset = MultiOmicsDataset(expression_data_scaled, metabolite_data_scaled,
                                 negative_sampling_ratio=negative_sampling_ratio)
dataset_size = len(full_dataset)
val_size = int(0.2 * dataset_size)
train_size = dataset_size - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
logging.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

num_workers = min(4, os.cpu_count() // 2 if os.cpu_count() else 1)
logging.info(f"DataLoader using num_workers: {num_workers}")

# Loss weights (based on the ratio of modality feature counts)
num_proteins = len(common_proteins)
num_metabolites = len(common_metabolites)
total_features = num_proteins + num_metabolites
weight_mlm_protein = 2 * num_proteins / total_features
weight_mlm_metabolite = 2 * num_metabolites / total_features
weight_itm = 1.0
weight_reconstruction_protein = 2 * num_proteins / total_features
weight_reconstruction_metabolite = 2 * num_metabolites / total_features

logging.info(f"Loss weights: MLM Protein={weight_mlm_protein:.4f}, MLM Metabolite={weight_mlm_metabolite:.4f}, "
             f"ITM={weight_itm:.4f}, Recon Protein={weight_reconstruction_protein:.4f}, "
             f"Recon Metabolite={weight_reconstruction_metabolite:.4f}")

#############################
# Create DataLoader, Model, Optimizer, and Scheduler
#############################
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

model = MultiModalBERTModel_Modified(
    hidden_size=hidden_size,
    num_proteins=num_proteins,
    num_metabolites=num_metabolites,
    dropout_prob=dropout_prob,
    global_features=filtered_global_features
)
if gpu_count > 1:
    from torch.nn.parallel import DataParallel
    model = DataParallel(model)
    logging.info("Model using DataParallel")
else:
    logging.info("Single GPU mode")
model.to(device)

optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
total_steps = len(train_loader) * num_epochs
scheduler = get_cosine_schedule_with_warmup(optimizer,
                                            num_warmup_steps=int(0.05 * total_steps),
                                            num_training_steps=total_steps)
scaler = GradScaler()

#############################
# Training / Validation Loop
#############################
metrics_history = {
    'train_loss': [],
    'val_loss': [],
    'train_mse_protein': [],
    'train_mae_protein': [],
    'train_r2_protein': [],
    'train_mse_metabolite': [],
    'train_mae_metabolite': [],
    'train_r2_metabolite': [],
    'val_mse_protein': [],
    'val_mae_protein': [],
    'val_r2_protein': [],
    'val_mse_metabolite': [],
    'val_mae_metabolite': [],
    'val_r2_metabolite': [],
    'train_itm_accuracy': [],
    'val_itm_accuracy': [],
    'train_mask_accuracy_protein': [],
    'train_mask_accuracy_metabolite': [],
    'train_reconstruction_loss_protein': [],
    'train_reconstruction_loss_metabolite': [],
    'train_lr': [],
}

best_val_loss = float('inf')
early_stop_patience = 10
no_improve_epochs = 0
best_epoch = 0

logging.info("Starting pre-training model training...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    all_pred_protein = []
    all_true_protein = []
    all_pred_metabolite = []
    all_true_metabolite = []
    mask_correct_prot = 0
    mask_total_prot = 0
    mask_correct_metab = 0
    mask_total_metab = 0
    train_itm_correct = 0
    train_itm_total = 0
    recon_loss_prot_epoch = []
    recon_loss_metab_epoch = []

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Train")
    for proteins, metabolites, itm_labels in progress_bar:
        proteins = proteins.to(device)
        metabolites = metabolites.to(device)
        itm_labels = itm_labels.to(device)

        # Mask protein and metabolite data respectively
        proteins_masked, mask_prot = mask_input(proteins, mask_prob=0.15)
        metabolites_masked, mask_metab = mask_input(metabolites, mask_prob=0.15)

        optimizer.zero_grad()
        with autocast():
            outputs = model(proteins_masked, metabolites_masked, itm_label=itm_labels,
                            mask_proteins=mask_prot, mask_metabolites=mask_metab,
                            return_attn=True)
            pred_prot = outputs['prediction_protein']  # Reconstruction output [B, num_proteins]
            pred_metab = outputs['prediction_metabolite']  # [B, num_metabolites]
            itm_logits = outputs['itm_logits']  # [B]
            mlm_pred_prot = outputs['mlm_pred_protein']  # [B, num_proteins]
            mlm_pred_metab = outputs['mlm_pred_metabolite']  # [B, num_metabolites]

            # MLM loss calculated only for masked parts
            if mask_prot.sum() > 0:
                loss_mlm_prot = nn.MSELoss()(mlm_pred_prot[mask_prot], proteins[mask_prot])
                mask_correct_prot += (torch.abs(mlm_pred_prot[mask_prot] - proteins[mask_prot]) <= 0.5).float().sum().item()
                mask_total_prot += mask_prot.sum().item()
            else:
                loss_mlm_prot = torch.tensor(0.0, device=device)
            if mask_metab.sum() > 0:
                loss_mlm_metab = nn.MSELoss()(mlm_pred_metab[mask_metab], metabolites[mask_metab])
                mask_correct_metab += (torch.abs(mlm_pred_metab[mask_metab] - metabolites[mask_metab]) <= 0.5).float().sum().item()
                mask_total_metab += mask_metab.sum().item()
            else:
                loss_mlm_metab = torch.tensor(0.0, device=device)
            loss_mlm = weight_mlm_protein * loss_mlm_prot + weight_mlm_metabolite * loss_mlm_metab

            loss_itm = weight_itm * nn.BCEWithLogitsLoss()(itm_logits, itm_labels)
            loss_recon_prot = weight_reconstruction_protein * nn.MSELoss()(pred_prot, proteins)
            loss_recon_metab = weight_reconstruction_metabolite * nn.MSELoss()(pred_metab, metabolites)
            loss_recon = loss_recon_prot + loss_recon_metab

            # Contrastive loss (based on global representations)
            prot_global = outputs['prot_global']
            metab_global = outputs['metab_global']
            loss_contrast = compute_contrastive_loss(prot_global, metab_global, itm_labels)

            loss = loss_mlm + loss_itm + loss_recon + lambda_contrast * loss_contrast
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        cur_lr = scheduler.get_last_lr()[0]
        metrics_history['train_lr'].append(cur_lr)
        total_loss += loss.item()

        # ITM accuracy
        itm_pred = (itm_logits >= 0).float()
        train_itm_correct += (itm_pred == itm_labels).sum().item()
        train_itm_total += itm_labels.numel()

        recon_loss_prot_epoch.append(loss_recon_prot.item())
        recon_loss_metab_epoch.append(loss_recon_metab.item())

        all_pred_protein.append(pred_prot.detach().cpu().numpy())
        all_true_protein.append(proteins.detach().cpu().numpy())
        all_pred_metabolite.append(pred_metab.detach().cpu().numpy())
        all_true_metabolite.append(metabolites.detach().cpu().numpy())

        progress_bar.set_postfix({
            'Loss': loss.item(),
            'ITM Acc': train_itm_correct / train_itm_total if train_itm_total > 0 else 0,
        })
    avg_train_loss = total_loss / len(train_loader)
    metrics_history['train_loss'].append(avg_train_loss)
    train_itm_acc = train_itm_correct / train_itm_total if train_itm_total > 0 else 0
    mask_acc_prot = mask_correct_prot / mask_total_prot if mask_total_prot > 0 else 0
    mask_acc_metab = mask_correct_metab / mask_total_metab if mask_total_metab > 0 else 0

    # Calculate reconstruction metrics on the training set
    if len(all_pred_protein) > 0:
        all_pred_protein_np = np.concatenate(all_pred_protein, axis=0)
        all_true_protein_np = np.concatenate(all_true_protein, axis=0)
        mse_prot = mean_squared_error(all_true_protein_np, all_pred_protein_np)
        mae_prot = mean_absolute_error(all_true_protein_np, all_pred_protein_np)
        r2_prot = r2_score(all_true_protein_np, all_pred_protein_np)
        metrics_history['train_mse_protein'].append(mse_prot)
        metrics_history['train_mae_protein'].append(mae_prot)
        metrics_history['train_r2_protein'].append(r2_prot)
        logging.info(f"Train Protein Metrics: MSE={mse_prot:.4f}, MAE={mae_prot:.4f}, R²={r2_prot:.4f}")
    else:
        metrics_history['train_mse_protein'].append(0)
        metrics_history['train_mae_protein'].append(0)
        metrics_history['train_r2_protein'].append(0)

    if len(all_pred_metabolite) > 0:
        all_pred_metabolite_np = np.concatenate(all_pred_metabolite, axis=0)
        all_true_metabolite_np = np.concatenate(all_true_metabolite, axis=0)
        mse_metab = mean_squared_error(all_true_metabolite_np, all_pred_metabolite_np)
        mae_metab = mean_absolute_error(all_true_metabolite_np, all_pred_metabolite_np)
        r2_metab = r2_score(all_true_metabolite_np, all_pred_metabolite_np)
        metrics_history['train_mse_metabolite'].append(mse_metab)
        metrics_history['train_mae_metabolite'].append(mae_metab)
        metrics_history['train_r2_metabolite'].append(r2_metab)
        logging.info(f"Train Metabolite Metrics: MSE={mse_metab:.4f}, MAE={mae_metab:.4f}, R²={r2_metab:.4f}")
    else:
        metrics_history['train_mse_metabolite'].append(0)
        metrics_history['train_mae_metabolite'].append(0)
        metrics_history['train_r2_metabolite'].append(0)

    metrics_history['train_itm_accuracy'].append(train_itm_acc)
    metrics_history['train_mask_accuracy_protein'].append(mask_acc_prot)
    metrics_history['train_mask_accuracy_metabolite'].append(mask_acc_metab)
    metrics_history['train_reconstruction_loss_protein'].append(np.mean(recon_loss_prot_epoch))
    metrics_history['train_reconstruction_loss_metabolite'].append(np.mean(recon_loss_metab_epoch))

    # ------- Validation Phase -------
    model.eval()
    val_loss = 0
    val_itm_correct = 0
    val_itm_total = 0
    val_all_pred_protein = []
    val_all_true_protein = []
    val_all_pred_metabolite = []
    val_all_true_metabolite = []
    with torch.no_grad():
        for proteins, metabolites, itm_labels in tqdm(val_loader, desc="Epoch {} - Validation".format(epoch + 1)):
            proteins = proteins.to(device)
            metabolites = metabolites.to(device)
            itm_labels = itm_labels.to(device)

            proteins_masked, mask_prot = mask_input(proteins, mask_prob=0.15)
            metabolites_masked, mask_metab = mask_input(metabolites, mask_prob=0.15)

            outputs = model(proteins_masked, metabolites_masked, itm_label=itm_labels,
                            mask_proteins=mask_prot, mask_metabolites=mask_metab,
                            return_attn=False)
            pred_prot = outputs['prediction_protein']
            pred_metab = outputs['prediction_metabolite']
            itm_logits = outputs['itm_logits']
            mlm_pred_prot = outputs['mlm_pred_protein']
            mlm_pred_metab = outputs['mlm_pred_metabolite']

            loss_mlm_prot = nn.MSELoss()(mlm_pred_prot[mask_prot], proteins[mask_prot]) if mask_prot.sum() > 0 else torch.tensor(0.0, device=device)
            loss_mlm_metab = nn.MSELoss()(mlm_pred_metab[mask_metab], metabolites[mask_metab]) if mask_metab.sum() > 0 else torch.tensor(0.0, device=device)
            loss_mlm = weight_mlm_protein * loss_mlm_prot + weight_mlm_metabolite * loss_mlm_metab
            loss_itm = weight_itm * nn.BCEWithLogitsLoss()(itm_logits, itm_labels)
            loss_recon_prot = weight_reconstruction_protein * nn.MSELoss()(pred_prot, proteins)
            loss_recon_metab = weight_reconstruction_metabolite * nn.MSELoss()(pred_metab, metabolites)
            loss_recon = loss_recon_prot + loss_recon_metab
            prot_global = outputs['prot_global']
            metab_global = outputs['metab_global']
            loss_contrast = compute_contrastive_loss(prot_global, metab_global, itm_labels)
            loss = loss_mlm + loss_itm + loss_recon + lambda_contrast * loss_contrast

            val_loss += loss.item()

            itm_pred = (itm_logits >= 0).float()
            val_itm_correct += (itm_pred == itm_labels).sum().item()
            val_itm_total += itm_labels.numel()

            val_all_pred_protein.append(pred_prot.detach().cpu().numpy())
            val_all_true_protein.append(proteins.detach().cpu().numpy())
            val_all_pred_metabolite.append(pred_metab.detach().cpu().numpy())
            val_all_true_metabolite.append(metabolites.detach().cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    metrics_history['val_loss'].append(avg_val_loss)
    val_itm_acc = val_itm_correct / val_itm_total if val_itm_total > 0 else 0
    metrics_history['val_itm_accuracy'].append(val_itm_acc)
    logging.info(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f}, ITM Acc: {val_itm_acc:.4f}")

    # Validation reconstruction metrics
    if len(val_all_pred_protein) > 0:
        val_all_pred_protein_np = np.concatenate(val_all_pred_protein, axis=0)
        val_all_true_protein_np = np.concatenate(val_all_true_protein, axis=0)
        mse_prot_val = mean_squared_error(val_all_true_protein_np, val_all_pred_protein_np)
        mae_prot_val = mean_absolute_error(val_all_true_protein_np, val_all_pred_protein_np)
        r2_prot_val = r2_score(val_all_true_protein_np, val_all_pred_protein_np)
        metrics_history['val_mse_protein'].append(mse_prot_val)
        metrics_history['val_mae_protein'].append(mae_prot_val)
        metrics_history['val_r2_protein'].append(r2_prot_val)
        logging.info(f"Validation Protein Metrics: MSE={mse_prot_val:.4f}, MAE={mae_prot_val:.4f}, R²={r2_prot_val:.4f}")
    if len(val_all_pred_metabolite) > 0:
        val_all_pred_metabolite_np = np.concatenate(val_all_pred_metabolite, axis=0)
        val_all_true_metabolite_np = np.concatenate(val_all_true_metabolite, axis=0)
        mse_metab_val = mean_squared_error(val_all_true_metabolite_np, val_all_pred_metabolite_np)
        mae_metab_val = mean_absolute_error(val_all_true_metabolite_np, val_all_pred_metabolite_np)
        r2_metab_val = r2_score(val_all_true_metabolite_np, val_all_pred_metabolite_np)
        metrics_history['val_mse_metabolite'].append(mse_metab_val)
        metrics_history['val_mae_metabolite'].append(mae_metab_val)
        metrics_history['val_r2_metabolite'].append(r2_metab_val)
        logging.info(f"Validation Metabolite Metrics: MSE={mse_metab_val:.4f}, MAE={mae_metab_val:.4f}, R²={r2_metab_val:.4f}")

    # Early stopping and saving best model (based on validation loss)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch + 1
        no_improve_epochs = 0
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.module.state_dict() if gpu_count > 1 else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': best_val_loss,
            'metrics_history': metrics_history,
        }, os.path.join(RESULTS_DIR, 'best_multi_omics_model.pt'))
        logging.info(f"Epoch {epoch + 1}: Saved best model.")
    else:
        no_improve_epochs += 1

    logging.info(f"Epoch {epoch + 1} completed.")
    if no_improve_epochs >= early_stop_patience:
        logging.info(f"Early stopping triggered: No improvement for {early_stop_patience} consecutive epochs, stopping training early.")
        break

#############################
# Plot training and validation curves along with metrics
#############################
logging.info("Plotting loss and metrics curves...")
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(metrics_history['train_loss']) + 1), metrics_history['train_loss'], marker='o', label='Train Loss')
plt.plot(range(1, len(metrics_history['val_loss']) + 1), metrics_history['val_loss'], marker='x', label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, 'training_validation_loss.pdf'))
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(metrics_history['train_itm_accuracy']) + 1), metrics_history['train_itm_accuracy'], marker='o', label='Train ITM Acc')
plt.plot(range(1, len(metrics_history['val_itm_accuracy']) + 1), metrics_history['val_itm_accuracy'], marker='x', label='Val ITM Acc')
plt.xlabel('Epoch')
plt.ylabel('ITM Accuracy')
plt.title('ITM Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, 'itm_accuracy.pdf'))
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(metrics_history['train_mask_accuracy_protein']) + 1), metrics_history['train_mask_accuracy_protein'], marker='o', label='Protein Mask Acc')
plt.plot(range(1, len(metrics_history['train_mask_accuracy_metabolite']) + 1), metrics_history['train_mask_accuracy_metabolite'], marker='x', label='Metabolite Mask Acc')
plt.xlabel('Epoch')
plt.ylabel('Mask Accuracy')
plt.title('MLM Mask Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, 'mask_accuracy.pdf'))
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(metrics_history['train_reconstruction_loss_protein']) + 1), metrics_history['train_reconstruction_loss_protein'], marker='o', label='Protein Recon Loss')
plt.plot(range(1, len(metrics_history['train_reconstruction_loss_metabolite']) + 1), metrics_history['train_reconstruction_loss_metabolite'], marker='x', label='Metabolite Recon Loss')
plt.xlabel('Epoch')
plt.ylabel('Reconstruction Loss')
plt.title('Reconstruction Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, 'reconstruction_loss.pdf'))
plt.show()

logging.info("Training process completed.")