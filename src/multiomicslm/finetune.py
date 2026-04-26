import os
import json
import math
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, precision_recall_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .core import MultiomicsLM, amp_autocast, torch_load, read_table, write_table


class OmicsDiseaseDataset(Dataset):
    def __init__(self, data: Dict[str, Any], labels: pd.DataFrame, config, eid_col: str = "eid"):
        self.data = data
        self.labels = labels.reset_index(drop=True)
        self.config = config
        self.eid_col = eid_col
        self.index = {str(e): i for i, e in enumerate(data["eids"].astype(str).tolist())}
        self.indices = np.asarray([self.index[str(e)] for e in self.labels[eid_col].astype(str).tolist()], dtype=np.int64)
        self.y = self.labels["label"].astype(float).to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        i = self.indices[idx]
        if self.data["protein_values"] is None:
            pv = np.zeros(self.config.N_PROTEINS, dtype=np.float32)
            pm = np.zeros(self.config.N_PROTEINS, dtype=np.float32)
        else:
            pv = self.data["protein_values"][i]
            pm = self.data["protein_mask"][i]
        if self.data["metabolite_values"] is None:
            mv = np.zeros(self.config.N_METABOLITES, dtype=np.float32)
            mm = np.zeros(self.config.N_METABOLITES, dtype=np.float32)
        else:
            mv = self.data["metabolite_values"][i]
            mm = self.data["metabolite_mask"][i]
        return {
            "eid": str(self.labels.iloc[idx][self.eid_col]),
            "protein_values": torch.from_numpy(pv).float(),
            "protein_mask": torch.from_numpy(pm).float(),
            "metabolite_values": torch.from_numpy(mv).float(),
            "metabolite_mask": torch.from_numpy(mm).float(),
            "has_protein": torch.tensor(float(pm.sum() > 0), dtype=torch.float32),
            "has_metabolite": torch.tensor(float(mm.sum() > 0), dtype=torch.float32),
            "label": torch.tensor(self.y[idx], dtype=torch.float32),
        }


def collate_batch(batch):
    out = {}
    out["eid"] = [b["eid"] for b in batch]
    for k in ["protein_values", "protein_mask", "metabolite_values", "metabolite_mask", "has_protein", "has_metabolite", "label"]:
        out[k] = torch.stack([b[k] for b in batch], dim=0) if batch[0][k].ndim > 0 else torch.tensor([b[k].item() for b in batch], dtype=torch.float32)
    return out


class DiseaseClassifier(nn.Module):
    def __init__(self, backbone, hidden_dim: int, mode: str = "fused", dropout: float = 0.3):
        super().__init__()
        self.backbone = backbone
        self.mode = mode
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, batch):
        feat = self.backbone.extract_features(
            batch["protein_values"],
            batch["protein_mask"],
            batch["metabolite_values"],
            batch["metabolite_mask"],
            batch["has_protein"],
            batch["has_metabolite"],
            mode=self.mode,
        )
        logits = self.head(feat).squeeze(-1)
        return logits, feat


def parse_labels(labels_path: str, eid_col: str = "eid", label_col: Optional[str] = None, baseline_date_col: Optional[str] = None, diagnosis_date_col: Optional[str] = None, task: str = "baseline") -> pd.DataFrame:
    df = read_table(labels_path, eid_col=eid_col)
    df[eid_col] = df[eid_col].astype(str)

    if label_col is not None:
        out = df[[eid_col, label_col]].copy()
        out = out.rename(columns={label_col: "label"})
        out["label"] = pd.to_numeric(out["label"], errors="coerce")
        out = out.dropna(subset=["label"])
        out["label"] = out["label"].astype(int)
        out = out[out["label"].isin([0, 1])].copy()
        return out[[eid_col, "label"]].reset_index(drop=True)

    if baseline_date_col is None or diagnosis_date_col is None:
        raise ValueError("Either label_col or both baseline_date_col and diagnosis_date_col must be provided")

    base = pd.to_datetime(df[baseline_date_col], errors="coerce")
    diag_raw = df[diagnosis_date_col].astype(str).replace({"": np.nan, "nan": np.nan, "NaN": np.nan})
    diag = pd.to_datetime(diag_raw, errors="coerce")
    valid_base = base.notna()
    df = df.loc[valid_base].copy()
    base = base.loc[valid_base]
    diag = diag.loc[valid_base]

    ever = diag.notna()
    prevalent = ever & (diag <= base)
    incident = ever & (diag > base)
    never = ~ever

    if task in ["baseline", "current", "diagnosis"]:
        include = prevalent | never
        labels = prevalent.astype(int)
    elif task in ["future", "future_all", "prediction"]:
        include = incident | never
        labels = incident.astype(int)
    else:
        raise ValueError(f"Unknown task: {task}")

    out = df.loc[include, [eid_col]].copy()
    out["label"] = labels.loc[include].astype(int).values
    return out[[eid_col, "label"]].reset_index(drop=True)


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    if len(np.unique(y_true)) < 2:
        auroc = np.nan
        auprc = np.nan
    else:
        auroc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))
    acc = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    if len(np.unique(y_true)) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = float(tn / (tn + fp + 1e-8))
    else:
        tn = fp = fn = tp = 0
        specificity = np.nan
    return {
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def move_batch(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def run_epoch(model, loader, criterion, optimizer, device, train: bool, use_amp: bool, grad_clip: float):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_eids = []
    all_y = []
    all_prob = []
    for batch in loader:
        batch = move_batch(batch, device)
        y = batch["label"]
        with torch.set_grad_enabled(train):
            with amp_autocast(use_amp and device.type == "cuda"):
                logits, _ = model(batch)
                loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        prob = torch.sigmoid(logits).detach().float().cpu().numpy()
        total_loss += float(loss.detach().cpu().item()) * len(y)
        all_eids.extend(batch["eid"])
        all_y.append(y.detach().float().cpu().numpy())
        all_prob.append(prob)
    y_true = np.concatenate(all_y) if all_y else np.array([])
    y_prob = np.concatenate(all_prob) if all_prob else np.array([])
    avg_loss = total_loss / max(len(loader.dataset), 1)
    return avg_loss, np.asarray(all_eids), y_true, y_prob


@torch.no_grad()
def predict(model, loader, device, use_amp: bool):
    model.eval()
    all_eids = []
    all_y = []
    all_prob = []
    all_feat = []
    for batch in loader:
        batch = move_batch(batch, device)
        with amp_autocast(use_amp and device.type == "cuda"):
            logits, feat = model(batch)
        prob = torch.sigmoid(logits).detach().float().cpu().numpy()
        all_eids.extend(batch["eid"])
        all_y.append(batch["label"].detach().float().cpu().numpy())
        all_prob.append(prob)
        all_feat.append(feat.detach().float().cpu().numpy())
    return np.asarray(all_eids), np.concatenate(all_y), np.concatenate(all_prob), np.concatenate(all_feat)


def plot_training(history: pd.DataFrame, out_prefix: str):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="Train")
    axes[0].plot(history["epoch"], history["val_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["epoch"], history["train_auroc"], label="Train")
    axes[1].plot(history["epoch"], history["val_auroc"], label="Validation")
    axes[1].set_title("AUROC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUROC")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history["epoch"], history["train_auprc"], label="Train")
    axes[2].plot(history["epoch"], history["val_auprc"], label="Validation")
    axes[2].set_title("AUPRC")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("AUPRC")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_prefix + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_prefix + ".pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr(y_true, y_prob, out_prefix: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        auroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
        axes[0].plot(fpr, tpr, label=f"AUROC = {auroc:.3f}")
        axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
        axes[0].set_xlabel("False positive rate")
        axes[0].set_ylabel("True positive rate")
        axes[0].set_title("ROC curve")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(recall, precision, label=f"AUPRC = {auprc:.3f}")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].set_title("Precision-recall curve")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_prefix + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_prefix + ".pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_probability(y_true, y_prob, out_prefix: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, 1, 40)
    ax.hist(y_prob[y_true == 0], bins=bins, density=True, alpha=0.55, label="Control")
    ax.hist(y_prob[y_true == 1], bins=bins, density=True, alpha=0.55, label="Case")
    ax.axvline(0.5, color="black", linestyle="--")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Density")
    ax.set_title("Predicted probability distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_prefix + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_prefix + ".pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def finetune_disease_model(
    pretrained_dir: Optional[str],
    protein_path: Optional[str],
    metabolite_path: Optional[str],
    labels_path: str,
    output_dir: str,
    eid_col: str = "eid",
    label_col: Optional[str] = None,
    baseline_date_col: Optional[str] = None,
    diagnosis_date_col: Optional[str] = None,
    task: str = "baseline",
    modality: str = "fused",
    finetune: str = "frozen",
    device: Optional[str] = None,
    batch_size: int = 64,
    max_epochs: int = 50,
    learning_rate_head: float = 1e-4,
    learning_rate_encoder: float = 1e-6,
    weight_decay: float = 1e-4,
    val_ratio: float = 0.2,
    patience: int = 10,
    threshold: float = 0.5,
    dropout: float = 0.3,
    seed: int = 42,
    use_amp: bool = True,
    grad_clip: float = 1.0,
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    client = MultiomicsLM(pretrained_dir=pretrained_dir, device=device)
    data = client.load_arrays(protein_path=protein_path, metabolite_path=metabolite_path, eid_col=eid_col)
    labels = parse_labels(
        labels_path=labels_path,
        eid_col=eid_col,
        label_col=label_col,
        baseline_date_col=baseline_date_col,
        diagnosis_date_col=diagnosis_date_col,
        task=task,
    )

    omics_eids = set(data["eids"].astype(str).tolist())
    labels = labels[labels[eid_col].astype(str).isin(omics_eids)].copy()
    labels = labels.drop_duplicates(subset=[eid_col]).reset_index(drop=True)
    if len(labels) < 20:
        raise ValueError("Too few labeled samples after intersecting with omics data")
    if labels["label"].nunique() < 2:
        raise ValueError("Labels contain only one class")

    train_df, val_df = train_test_split(
        labels,
        test_size=val_ratio,
        random_state=seed,
        stratify=labels["label"],
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_ds = OmicsDiseaseDataset(data, train_df, client.config, eid_col=eid_col)
    val_ds = OmicsDiseaseDataset(data, val_df, client.config, eid_col=eid_col)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

    model = DiseaseClassifier(client.model, client.config.HIDDEN_DIM, mode=modality, dropout=dropout).to(client.device)

    if finetune == "frozen":
        for p in model.backbone.parameters():
            p.requires_grad = False
        params = [{"params": model.head.parameters(), "lr": learning_rate_head}]
    elif finetune == "full":
        for p in model.parameters():
            p.requires_grad = True
        params = [
            {"params": model.head.parameters(), "lr": learning_rate_head},
            {"params": model.backbone.parameters(), "lr": learning_rate_encoder},
        ]
    else:
        raise ValueError("finetune must be frozen or full")

    optimizer = AdamW(params, weight_decay=weight_decay)
    n_pos = float((train_df["label"] == 1).sum())
    n_neg = float((train_df["label"] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32, device=client.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc = -np.inf
    best_epoch = 0
    best_state = None
    bad = 0
    history = []

    for epoch in range(max_epochs):
        train_loss, train_eids, train_y, train_prob = run_epoch(model, train_loader, criterion, optimizer, client.device, True, use_amp, grad_clip)
        val_loss, val_eids, val_y, val_prob = run_epoch(model, val_loader, criterion, optimizer, client.device, False, use_amp, grad_clip)
        train_metrics = compute_metrics(train_y, train_prob, threshold)
        val_metrics = compute_metrics(val_y, val_prob, threshold)

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_auroc": train_metrics["auroc"],
            "val_auroc": val_metrics["auroc"],
            "train_auprc": train_metrics["auprc"],
            "val_auprc": val_metrics["auprc"],
            "train_f1": train_metrics["f1"],
            "val_f1": val_metrics["f1"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(os.path.join(output_dir, "metrics", "training_history.csv"), index=False)

        current_auc = val_metrics["auroc"]
        improved = np.isfinite(current_auc) and current_auc > best_auc
        if improved:
            best_auc = current_auc
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": best_state,
                    "best_val_auroc": best_auc,
                    "modality": modality,
                    "finetune": finetune,
                    "threshold": threshold,
                },
                os.path.join(output_dir, "checkpoints", "best_model.pt"),
            )
            bad = 0
        else:
            bad += 1

        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_loader_eval = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)
    val_loader_eval = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_batch)

    train_eids, train_y, train_prob, train_feat = predict(model, train_loader_eval, client.device, use_amp)
    val_eids, val_y, val_prob, val_feat = predict(model, val_loader_eval, client.device, use_amp)

    train_metrics = compute_metrics(train_y, train_prob, threshold)
    val_metrics = compute_metrics(val_y, val_prob, threshold)

    pred_train = pd.DataFrame({
        eid_col: train_eids,
        "split": "train",
        "y_true": train_y.astype(int),
        "pred_prob": train_prob.astype(float),
        "pred_label": (train_prob >= threshold).astype(int),
    })
    pred_val = pd.DataFrame({
        eid_col: val_eids,
        "split": "val",
        "y_true": val_y.astype(int),
        "pred_prob": val_prob.astype(float),
        "pred_label": (val_prob >= threshold).astype(int),
    })
    pred_all = pd.concat([pred_train, pred_val], axis=0).reset_index(drop=True)

    pred_train.to_csv(os.path.join(output_dir, "predictions", "predictions_train.csv"), index=False)
    pred_val.to_csv(os.path.join(output_dir, "predictions", "predictions_val.csv"), index=False)
    pred_all.to_csv(os.path.join(output_dir, "predictions", "predictions_all.csv"), index=False)
    np.save(os.path.join(output_dir, "predictions", "train_embeddings.npy"), train_feat)
    np.save(os.path.join(output_dir, "predictions", "val_embeddings.npy"), val_feat)

    summary = {
        "best_epoch": best_epoch,
        "best_val_auroc": float(best_auc),
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "train_cases": int((train_df["label"] == 1).sum()),
        "val_cases": int((val_df["label"] == 1).sum()),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "task": task,
        "modality": modality,
        "finetune": finetune,
        "threshold": threshold,
    }
    with open(os.path.join(output_dir, "metrics", "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    hist = pd.DataFrame(history)
    plot_training(hist, os.path.join(output_dir, "figures", "training_curves"))
    plot_roc_pr(val_y, val_prob, os.path.join(output_dir, "figures", "roc_pr_val"))
    plot_probability(val_y, val_prob, os.path.join(output_dir, "figures", "probability_val"))

    return summary