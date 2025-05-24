import os
import random
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import logging
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import umap
import seaborn as sns
from scipy import stats

sns.set(style="whitegrid", font_scale=1.2)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# Path settings
PRETRAINED_MODEL_PATH = "your/path/results_multiomics_bert/best_multi_omics_model.pt"
LABEL_DIR = "your/path/labels"              # Directory for disease labels
HEALTHY_FILE = "your/path/healthy_eids.csv"
IMPUTED_DATA_PATH = "your/path/proteomic.csv"
METABOLITE_DATA_PATH = "your/path/metabolomic.csv"
GLOBAL_REP_PATH = "your/path/global_representations.npz"
OUTPUT_ROOT = "your/path/results_finetune"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Check device
if torch.cuda.is_available():
    device = torch.device("cuda")
    gpu_count = torch.cuda.device_count()
    logging.info(f"Using device: {device}, GPU count: {gpu_count}")
else:
    device = torch.device("cpu")
    gpu_count = 0
    logging.info("Training on CPU")


###############################################
# Fine-Tuning Dataset Class
###############################################
class FineTuneDataset(Dataset):
    def __init__(self, protein_array, metabolite_array, labels, eids):
        """
        protein_array: np.array, shape [N, num_proteins]
        metabolite_array: np.array, shape [N, num_metabolites]
        labels: list/array with binary labels (0 or 1)
        eids: list of participant IDs
        """
        self.protein_array = protein_array
        self.metabolite_array = metabolite_array
        self.labels = labels
        self.eids = eids
        self.n = len(labels)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        prot = torch.tensor(self.protein_array[idx], dtype=torch.float32)
        metab = torch.tensor(self.metabolite_array[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        eid = self.eids[idx]
        return prot, metab, label, eid


###############################################
# Transformer Model Components
###############################################
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
            CustomTransformerEncoderLayer(d_model, nhead, dropout, activation)
            for _ in range(num_layers)
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

# GatedFusion Module with optional gate extraction
class GatedFusion(nn.Module):
    def __init__(self, hidden_size):
        super(GatedFusion, self).__init__()
        self.gate_linear = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x, y, return_gate=False):
        concat = torch.cat([x, y], dim=-1)
        gate = torch.sigmoid(self.gate_linear(concat))
        fusion = gate * x + (1 - gate) * y
        if return_gate:
            return fusion, gate
        return fusion

###############################################
# Multi-Modal BERT Model
###############################################
class MultiModalBERTModel_Modified(nn.Module):
    def __init__(self, hidden_size, num_proteins, num_metabolites, dropout_prob=0.3, global_features=None):
        """
        global_features: np.array, shape [num_proteins, global_dim]
        """
        super(MultiModalBERTModel_Modified, self).__init__()
        self.hidden_size = hidden_size
        self.num_proteins = num_proteins
        self.num_metabolites = num_metabolites

        # Protein branch
        self.protein_encoder = ProteinBERTEncoder(seq_len=num_proteins, hidden_size=hidden_size,
                                                   num_layers=4, num_heads=8, dropout=dropout_prob)
        # Metabolite branch: project 1-dim to hidden_size then BERT encoding
        self.metab_proj = nn.Linear(1, hidden_size)
        self.metabolite_encoder = MetaboliteBERTEncoder(seq_len=num_metabolites, hidden_size=hidden_size,
                                                        num_layers=4, num_heads=8, dropout=dropout_prob)
        # Global protein representation projection
        if global_features is None:
            raise ValueError("global_features must be provided")
        self.register_buffer('global_features', torch.tensor(global_features, dtype=torch.float32))
        self.global_proj = nn.Linear(self.global_features.shape[1], hidden_size)

        # Fusion module with gated fusion
        self.gated_fusion = GatedFusion(hidden_size)
        # Classification head for disease classification
        self.itm_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size // 2, 1)
        )
        # Additional heads (not used in fine-tuning loss)
        self.mlm_head_protein = nn.Linear(hidden_size, 1)
        self.mlm_head_metabolite = nn.Linear(hidden_size, 1)
        self.output_layer_protein = nn.Linear(hidden_size, 1)
        self.output_layer_metabolite = nn.Linear(hidden_size, 1)

    def forward(self, *args, **kwargs):
        """
        Redesigned forward method to support various input types and to provide the following optional outputs:
        - return_attn: returns attention from protein branch
        - return_metab_attn: returns attention from metabolite branch
        - return_fusion_gate: returns gate weights from the fusion layer
        - return_cosine_similarity: computes token-level cosine similarity using token outputs from protein and metabolite encoders
        """
        proteins = None
        metabolites = None
        itm_label = None
        mask_proteins = None
        mask_metabolites = None

        # Support positional arguments, tuple packing, and keyword arguments
        if len(args) >= 2:
            proteins, metabolites = args[0], args[1]
            if len(args) > 2:
                itm_label = args[2]
            if len(args) > 3:
                mask_proteins = args[3]
            if len(args) > 4:
                mask_metabolites = args[4]
        elif len(args) == 1:
            if isinstance(args[0], tuple) and len(args[0]) >= 2:
                proteins, metabolites = args[0][0], args[0][1]
                if len(args[0]) > 2:
                    itm_label = args[0][2]
                if len(args[0]) > 3:
                    mask_proteins = args[0][3]
                if len(args[0]) > 4:
                    mask_metabolites = args[0][4]
            elif torch.is_tensor(args[0]) and 'metabolites' in kwargs:
                proteins = args[0]
                metabolites = kwargs['metabolites']
        if proteins is None and 'proteins' in kwargs:
            proteins = kwargs['proteins']
        if metabolites is None and 'metabolites' in kwargs:
            metabolites = kwargs['metabolites']
        if itm_label is None and 'itm_label' in kwargs:
            itm_label = kwargs['itm_label']
        if mask_proteins is None and 'mask_proteins' in kwargs:
            mask_proteins = kwargs['mask_proteins']
        if mask_metabolites is None and 'mask_metabolites' in kwargs:
            mask_metabolites = kwargs['mask_metabolites']

        # Obtain optional return flags
        return_attn = kwargs.get('return_attn', False)
        return_metab_attn = kwargs.get('return_metab_attn', False)
        return_fusion = kwargs.get('return_fusion', False)
        return_fusion_gate = kwargs.get('return_fusion_gate', False)
        return_cosine_similarity = kwargs.get('return_cosine_similarity', False)

        if proteins is None or metabolites is None:
            raise ValueError(f"Missing required inputs: proteins and metabolites. Args: {args}, Kwargs: {list(kwargs.keys())}")

        B = proteins.size(0)
        # Process protein branch
        projected_global = self.global_proj(self.global_features)  # [num_proteins, hidden_size]
        projected_global = projected_global.unsqueeze(0)  # [1, num_proteins, hidden_size]
        prot_inputs = proteins.unsqueeze(-1) * projected_global  # [B, num_proteins, hidden_size]
        if return_attn:
            prot_encoded, attn_all = self.protein_encoder(prot_inputs, return_attn=True)
            attn_weights = attn_all[-1]
        else:
            prot_encoded = self.protein_encoder(prot_inputs, return_attn=False)
            attn_weights = None

        protein_global = prot_encoded.mean(dim=1)  # [B, hidden_size]

        # Process metabolite branch
        metab_inputs = metabolites.unsqueeze(-1)  # [B, num_metabolites, 1]
        metab_inputs = self.metab_proj(metab_inputs)  # [B, num_metabolites, hidden_size]
        if return_metab_attn:
            metab_encoded, metab_attn_all = self.metabolite_encoder(metab_inputs, return_attn=True)
            metab_attn_weights = metab_attn_all[-1]
        else:
            metab_encoded = self.metabolite_encoder(metab_inputs, return_attn=False)
            metab_attn_weights = None
        metabolite_global = metab_encoded.mean(dim=1)  # [B, hidden_size]

        # Fusion of protein and metabolite global features
        if return_fusion_gate:
            fusion, fusion_gate = self.gated_fusion(protein_global, metabolite_global, return_gate=True)
        else:
            fusion = self.gated_fusion(protein_global, metabolite_global)

        itm_logits = self.itm_classifier(fusion).squeeze(-1)

        mlm_pred_protein = self.mlm_head_protein(prot_encoded).squeeze(-1)
        mlm_pred_metabolite = self.mlm_head_metabolite(metab_encoded).squeeze(-1)
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
        if return_metab_attn:
            outputs['metab_attn_weights'] = metab_attn_weights
        if return_fusion:
            outputs['fusion'] = fusion
        if return_fusion_gate:
            outputs['fusion_gate'] = fusion_gate
        return outputs


###############################################
# Helper Function
###############################################
def load_global_features(global_rep_path, protein_columns):
    global_data = np.load(global_rep_path)
    common_proteins = sorted(list(set(protein_columns) & set(global_data.keys())))
    logging.info(f"Number of proteins after filtering global representations: {len(common_proteins)}")
    filtered_global_features = np.stack([global_data[p] for p in common_proteins], axis=0)
    return common_proteins, filtered_global_features


###############################################
# Fine-Tuning Function: Train and Evaluate for a Single Disease
###############################################
def fine_tune_disease(disease_file, protein_full_df, metabolite_full_df, global_data):
    try:
        # Create output directory for the disease
        disease_out_dir = os.path.join(OUTPUT_ROOT, os.path.splitext(disease_file)[0])
        os.makedirs(disease_out_dir, exist_ok=True)

        disease_log_path = os.path.join(disease_out_dir, "fine_tuning.log")
        file_handler = logging.FileHandler(disease_log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)

        # Read case and healthy control IDs
        disease_path = os.path.join(LABEL_DIR, disease_file)
        disease_df = pd.read_csv(disease_path)
        disease_eids = disease_df['eid'].astype(str).tolist()
        healthy_df = pd.read_csv(HEALTHY_FILE)
        healthy_eids = healthy_df['eid'].astype(str).tolist()
        all_eids = list(set(disease_eids + healthy_eids))
        logging.info(f"Fine-tuning on [{disease_file}]: total {len(all_eids)} participants (Disease: {len(disease_eids)}, Healthy: {len(healthy_eids)})")
        disease_name = disease_file.replace("_labels_filtered.csv", "").title()

        protein_sub = protein_full_df[protein_full_df['eid'].astype(str).isin(all_eids)].copy()
        metabolite_sub = metabolite_full_df[metabolite_full_df['eid'].astype(str).isin(all_eids)].copy()
        protein_sub.set_index('eid', inplace=True)
        metabolite_sub.set_index('eid', inplace=True)
        common_eids = sorted(list(set(protein_sub.index) & set(metabolite_sub.index)))
        logging.info(f"Number of participants after intersection: {len(common_eids)}")
        protein_sub = protein_sub.loc[common_eids]
        metabolite_sub = metabolite_sub.loc[common_eids]

        common_prot = sorted(list(set(protein_sub.columns) &
                                    set(global_data.files if hasattr(global_data, 'files') else global_data.keys())))
        protein_sub = protein_sub[common_prot]
        logging.info(f"Number of proteins for fine-tuning: {len(common_prot)}")

        common_metab = sorted(list(metabolite_sub.columns))
        metabolite_sub = metabolite_sub[common_metab]
        logging.info(f"Number of metabolites for fine-tuning: {len(common_metab)}")

        # Data normalization (using RobustScaler)
        scaler_prot = RobustScaler()
        scaler_metab = RobustScaler()
        protein_scaled = pd.DataFrame(scaler_prot.fit_transform(protein_sub), index=protein_sub.index, columns=protein_sub.columns)
        metabolite_scaled = pd.DataFrame(scaler_metab.fit_transform(metabolite_sub), index=metabolite_sub.index, columns=metabolite_sub.columns)

        # Construct labels
        labels = [1 if eid in disease_eids else 0 for eid in common_eids]

        # Convert to numpy arrays
        protein_array = protein_scaled.values  # [N, num_proteins]
        metabolite_array = metabolite_scaled.values  # [N, num_metabolites]

        common_proteins, filtered_global_features = load_global_features(GLOBAL_REP_PATH, common_prot)
        logging.info(f"During fine-tuning, number of proteins corresponding to global representations: {len(common_proteins)}")

        # Split the data into training and validation sets (80/20)
        train_idx, val_idx = train_test_split(range(len(common_eids)), test_size=0.2, random_state=42, stratify=labels)
        train_dataset = FineTuneDataset(protein_array[train_idx], metabolite_array[train_idx],
                                        np.array(labels)[train_idx], np.array(common_eids)[train_idx])
        val_dataset = FineTuneDataset(protein_array[val_idx], metabolite_array[val_idx],
                                      np.array(labels)[val_idx], np.array(common_eids)[val_idx])
        logging.info(f"Number of training samples: {len(train_dataset)}, validation samples: {len(val_dataset)}")

        # DataLoader settings
        batch_size = 16
        if gpu_count > 1 and batch_size < gpu_count:
            logging.warning(f"Increasing batch size from {batch_size} to {gpu_count} to match GPU count")
            batch_size = gpu_count

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

        ###############################################
        # Build model, load pretrained parameters, and fine-tune
        ###############################################
        hidden_size = 768
        model = MultiModalBERTModel_Modified(
            hidden_size=hidden_size,
            num_proteins=len(common_prot),
            num_metabolites=len(common_metab),
            dropout_prob=0.3,
            global_features=filtered_global_features
        )
        if os.path.exists(PRETRAINED_MODEL_PATH):
            state_dict = torch.load(PRETRAINED_MODEL_PATH, map_location=device)
            if 'model_state_dict' in state_dict:
                pretrained_dict = state_dict['model_state_dict']
            else:
                pretrained_dict = state_dict
            model.load_state_dict(pretrained_dict, strict=False)
            logging.info("Successfully loaded pretrained model parameters")
        else:
            logging.error(f"Pretrained model file does not exist: {PRETRAINED_MODEL_PATH}")

        model.to(device)
        if gpu_count > 1:
            from torch.nn.parallel import DataParallel
            model = DataParallel(model)
            logging.info("Using DataParallel for fine-tuning")

        learning_rate = 2e-5
        weight_decay = 1e-3
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        num_epochs = 30
        total_steps = len(train_loader) * num_epochs
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(0.05 * total_steps),
                                                    num_training_steps=total_steps)
        scaler = GradScaler()
        bce_loss_fn = nn.BCEWithLogitsLoss()

        best_auc = 0
        best_epoch = 0
        metrics_history = []

        logging.info(f"Starting fine-tuning for [{disease_file}] ...")
        for epoch in range(num_epochs):
            model.train()
            train_losses = []
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training")
            for prot, metab, labels_batch, _ in pbar:
                if prot.size(0) == 0 or metab.size(0) == 0:
                    logging.warning("Skipping empty batch")
                    continue
                prot = prot.to(device)
                metab = metab.to(device)
                labels_batch = labels_batch.to(device)
                optimizer.zero_grad()
                with autocast():
                    try:
                        outputs = model(prot, metab)
                        logits = outputs['itm_logits']
                        loss = bce_loss_fn(logits, labels_batch)
                    except Exception as e:
                        logging.error(f"Forward pass error: {e}")
                        raise
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                train_losses.append(loss.item())
                pbar.set_postfix({'loss': loss.item()})
            avg_train_loss = np.mean(train_losses)

            model.eval()
            all_labels = []
            all_preds = []
            all_probs = []
            with torch.no_grad():
                for prot, metab, labels_batch, _ in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"):
                    prot = prot.to(device)
                    metab = metab.to(device)
                    labels_batch = labels_batch.to(device)
                    try:
                        outputs = model(prot, metab)
                        logits = outputs['itm_logits']
                        probs = torch.sigmoid(logits).detach().cpu().numpy()
                        preds = (probs >= 0.5).astype(int)
                        all_probs.extend(probs.tolist())
                        all_preds.extend(preds.tolist())
                        all_labels.extend(labels_batch.detach().cpu().numpy().tolist())
                    except Exception as e:
                        logging.error(f"Validation error: {e}")
                        continue

            try:
                acc = accuracy_score(all_labels, all_preds)
                prec = precision_score(all_labels, all_preds, zero_division=0)
                rec = recall_score(all_labels, all_preds, zero_division=0)
                f1 = f1_score(all_labels, all_preds, zero_division=0)
                auc = roc_auc_score(all_labels, all_probs)
            except Exception as e:
                logging.error(f"Error computing metrics: {e}")
                acc = prec = rec = f1 = auc = 0.0

            metrics_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'auc': auc
            })
            logging.info(f"Epoch {epoch + 1}: Loss={avg_train_loss:.4f}, Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

            if auc > best_auc:
                best_auc = auc
                best_epoch = epoch + 1
                save_dict = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict() if gpu_count > 1 else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'metrics': metrics_history,
                }
                best_model_path = os.path.join(disease_out_dir, "best_model.pt")
                torch.save(save_dict, best_model_path)
                logging.info(f"Epoch {epoch + 1}: Updated model, saved to {best_model_path}")

        # Inference: extract predictions, fusion features, and fusion gate weights
        train_pred_records = []
        train_fusion_features = []
        train_fusion_gate_records = []

        if gpu_count > 1:
            inference_model = model.module
        else:
            inference_model = model

        inference_model.eval()
        for prot, metab, labels_batch, eids in tqdm(train_loader, desc="Infer Train Set"):
            if prot.size(0) == 0 or metab.size(0) == 0:
                continue
            prot = prot.to(device)
            metab = metab.to(device)
            try:
                with torch.no_grad():
                    outputs = inference_model(prot, metab, return_fusion=True, return_fusion_gate=True, return_metab_attn=True)
                logits = outputs['itm_logits']
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                fusion = outputs.get('fusion', None)
                fusion_gate = outputs.get('fusion_gate', None)
                for i, eid in enumerate(eids):
                    train_pred_records.append({
                        'eid': eid,
                        'split': 'train',
                        'true_label': float(labels_batch[i].item()),
                        'predicted_probability': float(probs[i]),
                        'predicted_label': int(preds[i])
                    })
                    if fusion is not None:
                        train_fusion_features.append((eid, fusion[i].detach().cpu().numpy(), labels_batch[i].item(), 'train'))
                    if fusion_gate is not None:
                        train_fusion_gate_records.append((eid, fusion_gate[i].detach().cpu().numpy(), labels_batch[i].item(), 'train'))
            except Exception as e:
                logging.error(f"Train inference error: {e}")
                continue

        val_pred_records = []
        val_fusion_features = []
        val_fusion_gate_records = []

        for prot, metab, labels_batch, eids in tqdm(val_loader, desc="Infer Validation Set"):
            prot = prot.to(device)
            metab = metab.to(device)
            try:
                with torch.no_grad():
                    outputs = inference_model(prot, metab, return_fusion=True, return_fusion_gate=True, return_metab_attn=True)
                logits = outputs['itm_logits']
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                fusion = outputs.get('fusion', None)
                fusion_gate = outputs.get('fusion_gate', None)
                for i, eid in enumerate(eids):
                    val_pred_records.append({
                        'eid': eid,
                        'split': 'val',
                        'true_label': float(labels_batch[i].item()),
                        'predicted_probability': float(probs[i]),
                        'predicted_label': int(preds[i])
                    })
                    if fusion is not None:
                        val_fusion_features.append((eid, fusion[i].detach().cpu().numpy(), labels_batch[i].item(), 'val'))
                    if fusion_gate is not None:
                        val_fusion_gate_records.append((eid, fusion_gate[i].detach().cpu().numpy(), labels_batch[i].item(), 'val'))
            except Exception as e:
                logging.error(f"Validation inference error: {e}")
                continue

        combined_records = train_pred_records + val_pred_records
        pred_df = pd.DataFrame(combined_records)
        pred_csv_path = os.path.join(disease_out_dir, "predictions_train_val.csv")
        pred_df.to_csv(pred_csv_path, index=False)
        logging.info(f"Combined prediction results saved to {pred_csv_path}")

        # Visualization: t-SNE and UMAP clustering based on fusion features
        combined_fusion_features = train_fusion_features + val_fusion_features
        if len(combined_fusion_features) > 0:
            eids_all = [item[0] for item in combined_fusion_features]
            fusion_all = np.array([item[1] for item in combined_fusion_features])
            labels_all = np.array([item[2] for item in combined_fusion_features])
            split_all = [item[3] for item in combined_fusion_features]
            tsne = TSNE(n_components=2, random_state=42, init='pca')
            tsne_result = tsne.fit_transform(fusion_all)
            tsne_df = pd.DataFrame({
                'tsne1': tsne_result[:, 0],
                'tsne2': tsne_result[:, 1],
                'label': labels_all,
                'split': split_all
            })
            plt.figure(figsize=(8, 6))
            for lab, color in zip([0, 1], ['lightblue', 'lightcoral']):
                indices = tsne_df['label'] == lab
                plt.scatter(tsne_df.loc[indices, 'tsne1'],
                            tsne_df.loc[indices, 'tsne2'],
                            marker='o', color=color, s=50, alpha=0.7,
                            label='Healthy' if lab == 0 else disease_name)
            plt.xlabel('t-SNE 1', fontsize=14)
            plt.ylabel('t-SNE 2', fontsize=14)
            plt.title('t-SNE Clustering', fontsize=16)
            plt.legend(fontsize=13)
            tsne_path = os.path.join(disease_out_dir, "tsne_clustering_train_val.pdf")
            plt.tight_layout()
            plt.savefig(tsne_path)
            plt.close()
            logging.info(f"t-SNE clustering plot saved to {tsne_path}")

            reducer = umap.UMAP(n_components=2, random_state=42)
            umap_result = reducer.fit_transform(fusion_all)
            umap_df = pd.DataFrame({
                'umap1': umap_result[:, 0],
                'umap2': umap_result[:, 1],
                'label': labels_all,
                'split': split_all
            })
            plt.figure(figsize=(8, 6))
            for lab, color in zip([0, 1], ['lightblue', 'lightcoral']):
                indices = umap_df['label'] == lab
                plt.scatter(umap_df.loc[indices, 'umap1'],
                            umap_df.loc[indices, 'umap2'],
                            marker='o', color=color, s=50, alpha=0.7,
                            label='Healthy' if lab == 0 else disease_name)
            plt.xlabel('UMAP 1', fontsize=14)
            plt.ylabel('UMAP 2', fontsize=14)
            plt.title('UMAP Clustering', fontsize=16)
            plt.legend(fontsize=13)
            umap_path = os.path.join(disease_out_dir, "umap_clustering_train_val.pdf")
            plt.tight_layout()
            plt.savefig(umap_path)
            plt.close()
            logging.info(f"UMAP clustering plot saved to {umap_path}")

        # Protein attention visualization (protein branch)
        inference_model.eval()
        attn_weights_all = []
        with torch.no_grad():
            for prot, metab, _, _ in tqdm(val_loader, desc="Computing protein attention weights"):
                prot = prot.to(device)
                metab = metab.to(device)
                try:
                    outputs = inference_model(prot, metab, return_attn=True)
                    batch_attn = outputs.get('attn_weights', None)
                    if batch_attn is not None:
                        batch_attn = batch_attn.mean(dim=1)  # [B, num_proteins]
                        attn_weights_all.append(batch_attn.detach().cpu().numpy())
                except Exception as e:
                    logging.error(f"Protein attention computation error: {e}")
                    continue
        if len(attn_weights_all) > 0:
            attn_weights_all = np.concatenate(attn_weights_all, axis=0)  # [N, num_proteins]
            avg_attn = attn_weights_all.mean(axis=0)  # [num_proteins]
            attn_df = pd.DataFrame({"Protein": common_prot, "Average_Attention": avg_attn})
            attn_csv_path = os.path.join(disease_out_dir, "average_attention_weights_protein.csv")
            attn_df.to_csv(attn_csv_path, index=False)
            logging.info(f"Average protein attention saved to {attn_csv_path}")

            plt.figure(figsize=(max(8, len(common_prot) / 5), 6))
            plt.bar(range(len(common_prot)), avg_attn, width=0.5, color="mediumpurple")
            plt.xticks(range(len(common_prot)), common_prot, rotation=90, ha="center", fontsize=8)
            plt.xlabel("Protein", fontsize=12)
            plt.ylabel("Average Attention Weight", fontsize=12)
            plt.title("Overall Protein Attention", fontsize=14)
            plt.tight_layout()
            overall_attn_path = os.path.join(disease_out_dir, "average_attention_weights_protein.pdf")
            plt.savefig(overall_attn_path)
            plt.close()
            logging.info(f"Protein attention plot saved to {overall_attn_path}")

            top10_idx = np.argsort(avg_attn)[-10:][::-1]
            top10_proteins = [common_prot[i] for i in top10_idx]
            top10_weights = avg_attn[top10_idx]
            plt.figure(figsize=(8, 6))
            plt.bar(range(10), top10_weights, tick_label=top10_proteins, color="plum")
            plt.xticks(rotation=45, ha="right")
            plt.xlabel("Protein", fontsize=12)
            plt.ylabel("Average Attention Weight", fontsize=12)
            plt.title("Top 10 Protein Attention", fontsize=14)
            plt.tight_layout()
            top10_attn_path = os.path.join(disease_out_dir, "top10_attention_weights_protein.pdf")
            plt.savefig(top10_attn_path)
            plt.close()
            logging.info(f"Top 10 protein attention plot saved to {top10_attn_path}")

        # Metabolite attention visualization
        metab_attn_weights_all = []
        with torch.no_grad():
            for prot, metab, _, _ in tqdm(val_loader, desc="Computing metabolite attention weights"):
                prot = prot.to(device)
                metab = metab.to(device)
                try:
                    outputs = inference_model(prot, metab, return_metab_attn=True)
                    batch_metab_attn = outputs.get('metab_attn_weights', None)
                    if batch_metab_attn is not None:
                        batch_metab_attn = batch_metab_attn.mean(dim=1)  # [B, num_metabolites]
                        metab_attn_weights_all.append(batch_metab_attn.detach().cpu().numpy())
                except Exception as e:
                    logging.error(f"Metabolite attention computation error: {e}")
                    continue
        if len(metab_attn_weights_all) > 0:
            metab_attn_weights_all = np.concatenate(metab_attn_weights_all, axis=0)  # [N, num_metabolites]
            avg_metab_attn = metab_attn_weights_all.mean(axis=0)  # [num_metabolites]
            metab_attn_df = pd.DataFrame({"Metabolite": common_metab, "Average_Attention": avg_metab_attn})
            metab_attn_csv_path = os.path.join(disease_out_dir, "average_attention_weights_metabolite.csv")
            metab_attn_df.to_csv(metab_attn_csv_path, index=False)
            logging.info(f"Average metabolite attention saved to {metab_attn_csv_path}")

            plt.figure(figsize=(max(8, len(common_metab) / 5), 10))
            plt.bar(range(len(common_metab)), avg_metab_attn, width=0.5, color="teal")
            plt.xticks(range(len(common_metab)), common_metab, rotation=90, ha="center", fontsize=8)
            plt.xlabel("Metabolite", fontsize=12)
            plt.ylabel("Average Attention Weight", fontsize=12)
            plt.title("Overall Metabolite Attention", fontsize=14)
            plt.tight_layout()
            overall_metab_attn_path = os.path.join(disease_out_dir, "average_attention_weights_metabolite.pdf")
            plt.savefig(overall_metab_attn_path)
            plt.close()
            logging.info(f"Metabolite attention plot saved to {overall_metab_attn_path}")

            top10_idx_metab = np.argsort(avg_metab_attn)[-10:][::-1]
            top10_metabolites = [common_metab[i] for i in top10_idx_metab]
            top10_metab_weights = avg_metab_attn[top10_idx_metab]
            plt.figure(figsize=(8, 10))
            plt.bar(range(10), top10_metab_weights, tick_label=top10_metabolites, color="lightseagreen")
            plt.xticks(rotation=45, ha="right")
            plt.xlabel("Metabolite", fontsize=12)
            plt.ylabel("Average Attention Weight", fontsize=12)
            plt.title("Top 10 Metabolite Attention", fontsize=14)
            plt.tight_layout()
            top10_metab_attn_path = os.path.join(disease_out_dir, "top10_attention_weights_metabolite.pdf")
            plt.savefig(top10_metab_attn_path)
            plt.close()
            logging.info(f"Top 10 metabolite attention plot saved to {top10_metab_attn_path}")

        # Fusion gate visualization
        fusion_gate_all = []
        with torch.no_grad():
            for prot, metab, _, _ in tqdm(val_loader, desc="Computing fusion gate weights"):
                prot = prot.to(device)
                metab = metab.to(device)
                try:
                    outputs = inference_model(prot, metab, return_fusion_gate=True)
                    fusion_gate = outputs.get('fusion_gate', None)
                    if fusion_gate is not None:
                        fusion_gate_all.append(fusion_gate.detach().cpu().numpy())
                except Exception as e:
                    logging.error(f"Fusion gate computation error: {e}")
                    continue
        if len(fusion_gate_all) > 0:
            fusion_gate_all = np.concatenate(fusion_gate_all, axis=0)  # [N, hidden_size]
            avg_fusion_gate = fusion_gate_all.mean(axis=0)  # [hidden_size]
            fusion_gate_df = pd.DataFrame({"Fusion_Gate": avg_fusion_gate})
            fusion_gate_csv_path = os.path.join(disease_out_dir, "average_fusion_gate.csv")
            fusion_gate_df.to_csv(fusion_gate_csv_path, index=False)
            logging.info(f"Average fusion gate saved to {fusion_gate_csv_path}")

            plt.figure(figsize=(8, 6))
            plt.bar(range(len(avg_fusion_gate)), avg_fusion_gate, width=0.5, color="darkorange")
            plt.xlabel("Hidden Dimension", fontsize=12)
            plt.ylabel("Average Gate Value (Protein Contribution)", fontsize=12)
            plt.title("Average Fusion Gate Across Hidden Dimensions", fontsize=14)
            plt.tight_layout()
            fusion_gate_plot_path = os.path.join(disease_out_dir, "average_fusion_gate.pdf")
            plt.savefig(fusion_gate_plot_path)
            plt.close()
            logging.info(f"Fusion gate plot saved to {fusion_gate_plot_path}")

            fusion_gate_sample_means = np.array([fg.mean() for fg in fusion_gate_all])
            plt.figure(figsize=(8, 6))
            plt.hist(fusion_gate_sample_means, bins=30, color="salmon", edgecolor="black")
            plt.xlabel("Average Fusion Gate Value", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.title("Distribution of Average Fusion Gate Values", fontsize=14)
            plt.tight_layout()
            fusion_gate_hist_path = os.path.join(disease_out_dir, "fusion_gate_distribution.pdf")
            plt.savefig(fusion_gate_hist_path)
            plt.close()
            logging.info(f"Fusion gate distribution plot saved to {fusion_gate_hist_path}")

        # Save fine-tuning metrics summary
        metrics_df = pd.DataFrame(metrics_history)
        metrics_csv_path = os.path.join(disease_out_dir, "finetune_metrics.csv")
        metrics_df.to_csv(metrics_csv_path, index=False)
        logging.info(f"Fine-tuning metrics saved to {metrics_csv_path}")

        logging.getLogger().removeHandler(file_handler)
        summary = {
            'disease_file': disease_file,
            'num_samples': len(common_eids),
            'best_epoch': best_epoch,
            'best_auc': best_auc,
            'final_accuracy': acc,
            'final_precision': prec,
            'final_recall': rec,
            'final_f1': f1
        }
        return summary

    except Exception as e:
        logging.error(f"Fine-tuning failed for {disease_file}: {e}", exc_info=True)
        return {
            'disease_file': disease_file,
            'error': str(e),
            'status': 'failed'
        }


###############################################
# Main Process
###############################################
def main():
    logging.info("Loading complete protein data...")
    protein_full_df = pd.read_csv(IMPUTED_DATA_PATH, dtype={'eid': str})
    logging.info("Loading complete metabolite data...")
    metabolite_full_df = pd.read_csv(METABOLITE_DATA_PATH, dtype={'eid': str})
    global_data = np.load(GLOBAL_REP_PATH)
    disease_files = [f for f in os.listdir(LABEL_DIR) if f.endswith("_labels.csv")]
    logging.info(f"Found {len(disease_files)} disease label files")
    summary_list = []
    for disease_file in disease_files:
        try:
            summary = fine_tune_disease(disease_file, protein_full_df, metabolite_full_df, global_data)
            summary_list.append(summary)
        except Exception as e:
            logging.error(f"Fine-tuning failed for {disease_file}: {e}", exc_info=True)
            summary_list.append({
                'disease_file': disease_file,
                'error': str(e),
                'status': 'failed'
            })
    summary_df = pd.DataFrame(summary_list)
    summary_csv_path = os.path.join(OUTPUT_ROOT, "finetune_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    logging.info(f"Fine-tuning summary saved to {summary_csv_path}")


if __name__ == "__main__":
    main()
