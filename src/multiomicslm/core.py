import os
import json
import math
import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def torch_load(path: str, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def amp_autocast(enabled=True):
    if not enabled:
        return contextlib.nullcontext()
    try:
        return torch.amp.autocast("cuda", enabled=True)
    except Exception:
        return torch.cuda.amp.autocast(enabled=True)


def no_amp():
    try:
        return torch.amp.autocast("cuda", enabled=False)
    except Exception:
        return torch.cuda.amp.autocast(enabled=False)


@dataclass
class MultiomicsLMConfig:
    N_PROTEINS: int = 2923
    N_METABOLITES: int = 249
    SEMANTIC_DIM: int = 768
    HIDDEN_DIM: int = 512
    NUM_HEADS: int = 8
    DROPOUT: float = 0.12
    PROTEIN_WITHIN_LAYERS: int = 4
    PROTEIN_CROSS_LAYERS: int = 2
    METABOLITE_ENCODER_LAYERS: int = 4
    FUSION_LAYERS: int = 2
    MODALITY_DROPOUT: float = 0.15

    @classmethod
    def from_json(cls, path: str):
        cfg = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    def to_dict(self):
        return self.__dict__.copy()


def package_pretrained_dir() -> Path:
    return Path(__file__).resolve().parent / "assets" / "pretrained"


def resolve_pretrained_dir(pretrained_dir: Optional[str] = None) -> Path:
    if pretrained_dir is None:
        pretrained_dir = os.environ.get("MULTIOMICSLM_PRETRAINED_DIR", None)
    if pretrained_dir is None:
        pretrained_dir = package_pretrained_dir()
    p = Path(pretrained_dir).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Pretrained directory does not exist: {p}")
    return p


def find_file(root: Path, names: List[str], subdirs: List[str] = ["", "exported_models", "checkpoints"]) -> Path:
    for sub in subdirs:
        for name in names:
            p = root / sub / name if sub else root / name
            if p.exists():
                return p
    raise FileNotFoundError(f"Cannot find any of {names} under {root}")


def read_feature_names(path: Optional[Path], n: int, prefix: str) -> List[str]:
    if path is not None and path.exists():
        with open(path, "r", encoding="utf-8") as f:
            names = [x.strip() for x in f if x.strip()]
        if len(names) == n:
            return names
    return [f"{prefix}_{i + 1:04d}" for i in range(n)]


def read_table(path: str, eid_col: str = "eid") -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext in [".tsv", ".tab"]:
        df = pd.read_csv(path, sep="\t")
    elif ext in [".txt"]:
        df = pd.read_csv(path, sep=None, engine="python")
    else:
        df = pd.read_csv(path)
    if eid_col not in df.columns:
        df.insert(0, eid_col, np.arange(len(df)).astype(str))
    df[eid_col] = df[eid_col].astype(str)
    return df


def write_table(df: pd.DataFrame, path: str):
    ext = Path(path).suffix.lower()
    if ext in [".tsv", ".tab", ".txt"]:
        df.to_csv(path, sep="\t", index=False)
    else:
        df.to_csv(path, index=False)


def numeric_matrix(df: pd.DataFrame, eid_col: str) -> Tuple[np.ndarray, List[str]]:
    cols = [c for c in df.columns if c != eid_col]
    x = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    return x, cols


class SemanticGuidedEmbedding(nn.Module):
    def __init__(self, n_features, semantic_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.value_embed = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        self.semantic_proj = nn.Linear(semantic_dim, hidden_dim)
        self.missing_embed = nn.Embedding(2, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, mask, semantic_emb):
        b = values.shape[0]
        v = self.value_embed(values.unsqueeze(-1))
        s = self.semantic_proj(semantic_emb).unsqueeze(0).expand(b, -1, -1)
        m = self.missing_embed(mask.long())
        return self.dropout(self.layer_norm(v + s + m))


class ProteinEncoder(nn.Module):
    def __init__(self, n_proteins, n_clusters, hidden_dim, n_heads, n_within_layers, n_cross_layers, dropout):
        super().__init__()
        self.n_proteins = n_proteins
        self.n_clusters = n_clusters
        self.hidden_dim = hidden_dim

        wl = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.within_encoder = nn.TransformerEncoder(wl, num_layers=n_within_layers)

        self.cluster_queries = nn.Parameter(torch.randn(n_clusters, 1, hidden_dim) * 0.02)
        self.cluster_pool = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.cluster_norm = nn.LayerNorm(hidden_dim)

        cl = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cross_encoder = nn.TransformerEncoder(cl, num_layers=n_cross_layers)

        self.context_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.protein_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

        self.cls_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.cls_pool = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.cls_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    @staticmethod
    def safe_mask(m):
        return m.masked_fill(m.all(dim=1, keepdim=True), False)

    def forward(self, protein_emb, protein_mask, cluster_labels, cidx_list, has_protein):
        b = protein_emb.shape[0]
        device = protein_emb.device
        cluster_outputs = []
        reorder_indices = []
        cluster_summaries = []

        for c in range(self.n_clusters):
            ci = cidx_list[c]
            if len(ci) == 0:
                cluster_summaries.append(torch.zeros(b, self.hidden_dim, device=device))
                continue
            cf = protein_emb[:, ci, :]
            cm = protein_mask[:, ci]
            am = self.safe_mask(cm == 0)
            co = self.within_encoder(cf, src_key_padding_mask=am)
            cluster_outputs.append(co)
            reorder_indices.append(ci)
            q = self.cluster_queries[c].unsqueeze(0).expand(b, -1, -1)
            cs, _ = self.cluster_pool(q, co, co, key_padding_mask=am)
            cluster_summaries.append(self.cluster_norm(cs.squeeze(1)))

        all_f = torch.cat(cluster_outputs, dim=1)
        all_i = torch.cat(reorder_indices, dim=0)
        sort_order = torch.argsort(all_i)
        enhanced = all_f[:, sort_order, :]

        cr = torch.stack(cluster_summaries, dim=1)
        cc = self.cross_encoder(cr)

        ctx = self.context_proj(cc[:, cluster_labels, :])
        enhanced = enhanced + ctx
        enhanced = enhanced + self.protein_ffn(enhanced)

        cq = self.cls_query.expand(b, -1, -1)
        pm = self.safe_mask(protein_mask == 0)
        no_p = (has_protein == 0).unsqueeze(1).expand(-1, self.n_proteins)
        pm = self.safe_mask(pm | no_p)
        cls_out, _ = self.cls_pool(cq, enhanced, enhanced, key_padding_mask=pm)
        sample_repr = self.out_proj(self.cls_norm(cls_out.squeeze(1)))
        return enhanced, sample_repr


class MetaboliteEncoder(nn.Module):
    def __init__(self, n_metabolites, hidden_dim, n_heads, n_layers, dropout):
        super().__init__()
        self.n_metabolites = n_metabolites
        el = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(el, num_layers=n_layers)
        self.cls_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.cls_pool = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.cls_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, metabolite_emb, metabolite_mask, has_metabolite):
        b = metabolite_emb.shape[0]
        am = metabolite_mask == 0
        no_m = (has_metabolite == 0).unsqueeze(1).expand(-1, self.n_metabolites)
        am = am | no_m
        am = am.masked_fill(am.all(dim=1, keepdim=True), False)
        feat = self.encoder(metabolite_emb, src_key_padding_mask=am)
        cq = self.cls_query.expand(b, -1, -1)
        cls_out, _ = self.cls_pool(cq, feat, feat, key_padding_mask=am)
        sample_repr = self.out_proj(self.cls_norm(cls_out.squeeze(1)))
        return feat, sample_repr


class CrossModalFusion(nn.Module):
    def __init__(self, hidden_dim, n_heads, n_layers, dropout, modality_dropout=0.15):
        super().__init__()
        self.modality_dropout = modality_dropout
        self.n_layers = n_layers

        self.p2m_attn = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        self.p2m_norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.p2m_ffn = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        ])

        self.m2p_attn = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        self.m2p_norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.m2p_ffn = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        ])

        self.protein_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.metabolite_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        nn.init.constant_(self.protein_gate[0].bias, -1.5)
        nn.init.constant_(self.metabolite_gate[0].bias, -1.5)

        self.fuse_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, protein_repr, metabolite_repr, has_protein, has_metabolite):
        b = protein_repr.shape[0]
        device = protein_repr.device
        has_both = has_protein * has_metabolite

        if self.training and self.modality_dropout > 0:
            drop = torch.rand(b, device=device) < self.modality_dropout
            eff_both = has_both.clone()
            eff_both[drop & (has_both > 0)] = 0.0
        else:
            eff_both = has_both

        mask3 = eff_both.unsqueeze(1).unsqueeze(2)
        p = protein_repr.unsqueeze(1)
        m = metabolite_repr.unsqueeze(1)

        for i in range(self.n_layers):
            pc, _ = self.p2m_attn[i](p, m, m)
            p = self.p2m_norm[i](p + pc * mask3)
            p = p + self.p2m_ffn[i](p) * mask3
            mc, _ = self.m2p_attn[i](m, p, p)
            m = self.m2p_norm[i](m + mc * mask3)
            m = m + self.m2p_ffn[i](m) * mask3

        p = p.squeeze(1)
        m = m.squeeze(1)

        cross_p = p - protein_repr
        cross_m = m - metabolite_repr
        protein_enriched = protein_repr + self.protein_gate(cross_p) * cross_p * eff_both.unsqueeze(1)
        metabolite_enriched = metabolite_repr + self.metabolite_gate(cross_m) * cross_m * eff_both.unsqueeze(1)

        fused = self.fuse_proj(torch.cat([protein_enriched, metabolite_enriched], dim=-1))
        only_p = has_protein * (1 - has_metabolite)
        only_m = (1 - has_protein) * has_metabolite
        fused = fused * has_both.unsqueeze(1) + protein_enriched * only_p.unsqueeze(1) + metabolite_enriched * only_m.unsqueeze(1)
        return protein_enriched, metabolite_enriched, fused


class ContrastiveHead(nn.Module):
    def __init__(self, hidden_dim, proj_dim=256, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x):
        return F.normalize(self.proj(x), dim=-1)


class MultiModalPretrainModel(nn.Module):
    def __init__(self, config, protein_sem_emb, metabolite_sem_emb, cluster_labels, n_clusters):
        super().__init__()
        self.config = config
        self.n_clusters = n_clusters
        h = config.HIDDEN_DIM

        self.register_buffer("protein_sem_emb", protein_sem_emb)
        self.register_buffer("metabolite_sem_emb", metabolite_sem_emb)
        self.register_buffer("cluster_labels", cluster_labels)
        self._n_cidx = n_clusters

        cl_np = cluster_labels.cpu().numpy() if torch.is_tensor(cluster_labels) else np.asarray(cluster_labels)
        for c in range(n_clusters):
            idx = torch.from_numpy((cl_np == c).nonzero()[0]).long()
            self.register_buffer(f"cidx_{c}", idx)

        self.protein_embed = SemanticGuidedEmbedding(config.N_PROTEINS, config.SEMANTIC_DIM, h, config.DROPOUT)
        self.metabolite_embed = SemanticGuidedEmbedding(config.N_METABOLITES, config.SEMANTIC_DIM, h, config.DROPOUT)
        self.modality_type_embed = nn.Embedding(2, h)

        self.protein_encoder = ProteinEncoder(
            config.N_PROTEINS,
            n_clusters,
            h,
            config.NUM_HEADS,
            config.PROTEIN_WITHIN_LAYERS,
            config.PROTEIN_CROSS_LAYERS,
            config.DROPOUT,
        )
        self.metabolite_encoder = MetaboliteEncoder(
            config.N_METABOLITES,
            h,
            config.NUM_HEADS,
            config.METABOLITE_ENCODER_LAYERS,
            config.DROPOUT,
        )
        self.fusion = CrossModalFusion(
            h,
            config.NUM_HEADS,
            config.FUSION_LAYERS,
            config.DROPOUT,
            config.MODALITY_DROPOUT,
        )

        self.protein_recon_head = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(h, 1),
        )
        self.metabolite_recon_head = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(h, 1),
        )
        self.cross_metabolite_head = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(h, config.N_METABOLITES),
        )
        self.protein_contrast = ContrastiveHead(h, dropout=config.DROPOUT)
        self.metabolite_contrast = ContrastiveHead(h, dropout=config.DROPOUT)

    def get_cidx_list(self):
        return [getattr(self, f"cidx_{c}") for c in range(self._n_cidx)]

    def forward(self, protein_values, protein_mask, metabolite_values, metabolite_mask, has_protein, has_metabolite):
        b = protein_values.shape[0]
        device = protein_values.device
        cfg = self.config

        p_emb = self.protein_embed(protein_values, protein_mask, self.protein_sem_emb)
        m_emb = self.metabolite_embed(metabolite_values, metabolite_mask, self.metabolite_sem_emb)
        p_emb = p_emb + self.modality_type_embed(torch.zeros(b, cfg.N_PROTEINS, dtype=torch.long, device=device))
        m_emb = m_emb + self.modality_type_embed(torch.ones(b, cfg.N_METABOLITES, dtype=torch.long, device=device))

        enhanced_p, p_sample = self.protein_encoder(
            p_emb,
            protein_mask,
            self.cluster_labels,
            self.get_cidx_list(),
            has_protein,
        )
        m_features, m_sample = self.metabolite_encoder(m_emb, metabolite_mask, has_metabolite)
        p_enriched, m_enriched, fused = self.fusion(p_sample, m_sample, has_protein, has_metabolite)

        with no_amp():
            ep = enhanced_p.float() + p_enriched.float().unsqueeze(1)
            protein_pred = self.protein_recon_head(ep).squeeze(-1)
            mf = m_features.float() + m_enriched.float().unsqueeze(1)
            metabolite_pred = self.metabolite_recon_head(mf).squeeze(-1)
            cross_metabolite_pred = self.cross_metabolite_head(p_sample.float())

        p_proj = self.protein_contrast(p_sample)
        m_proj = self.metabolite_contrast(m_sample)

        return {
            "protein_pred": protein_pred,
            "metabolite_pred": metabolite_pred,
            "cross_metabolite_pred": cross_metabolite_pred,
            "protein_sample_repr": p_sample,
            "metabolite_sample_repr": m_sample,
            "protein_enriched": p_enriched,
            "metabolite_enriched": m_enriched,
            "fused_repr": fused,
            "protein_proj": p_proj,
            "metabolite_proj": m_proj,
        }

    def extract_features(self, protein_values, protein_mask, metabolite_values, metabolite_mask, has_protein, has_metabolite, mode="fused"):
        out = self.forward(protein_values, protein_mask, metabolite_values, metabolite_mask, has_protein, has_metabolite)
        mapping = {
            "fused": "fused_repr",
            "protein": "protein_sample_repr",
            "metabolite": "metabolite_sample_repr",
            "protein_only": "protein_sample_repr",
            "metabolite_only": "metabolite_sample_repr",
            "protein_enriched": "protein_enriched",
            "metabolite_enriched": "metabolite_enriched",
        }
        return out[mapping.get(mode, "fused_repr")]


class MultiomicsLM:
    def __init__(self, pretrained_dir: Optional[str] = None, device: Optional[str] = None):
        self.pretrained_dir = resolve_pretrained_dir(pretrained_dir)
        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

        config_path = find_file(self.pretrained_dir, ["config.json"])
        self.config = MultiomicsLMConfig.from_json(str(config_path))

        cluster_path = find_file(self.pretrained_dir, ["cluster_labels.npy"])
        protein_sem_path = find_file(
            self.pretrained_dir,
            ["protein_semantic_embeddings.npy", "protein_pubmedbert_embeddings.npy", "protein_embeddings.npy"],
        )
        metabolite_sem_path = find_file(
            self.pretrained_dir,
            ["metabolite_semantic_embeddings.npy", "metabolite_pubmedbert_embeddings.npy", "metabolite_embeddings.npy"],
        )

        protein_sem = torch.from_numpy(np.load(protein_sem_path).astype(np.float32))
        metabolite_sem = torch.from_numpy(np.load(metabolite_sem_path).astype(np.float32))
        cluster_labels = torch.from_numpy(np.load(cluster_path).astype(np.int64)).long()
        n_clusters = int(len(np.unique(cluster_labels.cpu().numpy())))

        self.model = MultiModalPretrainModel(self.config, protein_sem, metabolite_sem, cluster_labels, n_clusters)

        weights_path = find_file(
            self.pretrained_dir,
            ["full_model_weights.pt", "model_weights_best.pt", "model_weights_latest.pt", "checkpoint_best.pt"],
            ["", "exported_models", "checkpoints"],
        )
        state = torch_load(str(weights_path), map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device).eval()

        self.protein_means = np.load(find_file(self.pretrained_dir, ["protein_means.npy"])).astype(np.float32)
        self.protein_stds = np.load(find_file(self.pretrained_dir, ["protein_stds.npy"])).astype(np.float32)
        self.metabolite_means = np.load(find_file(self.pretrained_dir, ["metabolite_means.npy"])).astype(np.float32)
        self.metabolite_stds = np.load(find_file(self.pretrained_dir, ["metabolite_stds.npy"])).astype(np.float32)

        protein_names_path = None
        metabolite_names_path = None
        try:
            protein_names_path = find_file(self.pretrained_dir, ["protein_columns.txt", "protein_features.txt"])
        except Exception:
            protein_names_path = None
        try:
            metabolite_names_path = find_file(self.pretrained_dir, ["metabolite_columns.txt", "metabolite_features.txt"])
        except Exception:
            metabolite_names_path = None

        self.protein_columns = read_feature_names(protein_names_path, self.config.N_PROTEINS, "protein")
        self.metabolite_columns = read_feature_names(metabolite_names_path, self.config.N_METABOLITES, "metabolite")

    def to(self, device: str):
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        return self

    def preprocess_proteins(self, raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = raw.astype(np.float32).copy()
        mask = np.isfinite(x).astype(np.float32)
        x[~np.isfinite(x)] = 0.0
        x = (x - self.protein_means) / self.protein_stds
        x = x * mask
        return x.astype(np.float32), mask.astype(np.float32)

    def preprocess_metabolites(self, raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = raw.astype(np.float32).copy()
        x[(x < 0) & np.isfinite(x)] = np.nan
        mask = np.isfinite(x).astype(np.float32)
        x[~np.isfinite(x)] = 0.0
        x = np.log1p(np.maximum(x, 0.0))
        x = (x - self.metabolite_means) / self.metabolite_stds
        x = x * mask
        return x.astype(np.float32), mask.astype(np.float32)

    def inverse_proteins(self, z: np.ndarray) -> np.ndarray:
        return (z * self.protein_stds + self.protein_means).astype(np.float32)

    def inverse_metabolites(self, z: np.ndarray) -> np.ndarray:
        y = z * self.metabolite_stds + self.metabolite_means
        y = np.expm1(np.maximum(y, 0.0))
        y = np.maximum(y, 0.0)
        return y.astype(np.float32)

    def load_arrays(self, protein_path: Optional[str] = None, metabolite_path: Optional[str] = None, eid_col: str = "eid") -> Dict[str, Any]:
        protein_df = read_table(protein_path, eid_col) if protein_path else None
        metabolite_df = read_table(metabolite_path, eid_col) if metabolite_path else None

        if protein_df is not None and metabolite_df is not None:
            p_eids = protein_df[eid_col].astype(str).tolist()
            m_set = set(metabolite_df[eid_col].astype(str).tolist())
            common = [x for x in p_eids if x in m_set]
            if len(common) == 0:
                raise ValueError("No overlapping EIDs between protein and metabolite files")
            protein_df = protein_df.set_index(eid_col).loc[common].reset_index()
            metabolite_df = metabolite_df.set_index(eid_col).loc[common].reset_index()
            eids = common
        elif protein_df is not None:
            eids = protein_df[eid_col].astype(str).tolist()
        elif metabolite_df is not None:
            eids = metabolite_df[eid_col].astype(str).tolist()
        else:
            raise ValueError("At least one of protein_path or metabolite_path must be provided")

        out = {"eids": np.asarray(eids, dtype=str)}

        if protein_df is not None:
            raw_p, p_cols = numeric_matrix(protein_df, eid_col)
            if raw_p.shape[1] != self.config.N_PROTEINS:
                raise ValueError(f"Protein feature count mismatch: got {raw_p.shape[1]}, expected {self.config.N_PROTEINS}")
            p_norm, p_mask = self.preprocess_proteins(raw_p)
            out.update({
                "protein_raw": raw_p,
                "protein_values": p_norm,
                "protein_mask": p_mask,
                "protein_columns": p_cols,
            })
        else:
            out.update({
                "protein_raw": None,
                "protein_values": None,
                "protein_mask": None,
                "protein_columns": self.protein_columns,
            })

        if metabolite_df is not None:
            raw_m, m_cols = numeric_matrix(metabolite_df, eid_col)
            if raw_m.shape[1] != self.config.N_METABOLITES:
                raise ValueError(f"Metabolite feature count mismatch: got {raw_m.shape[1]}, expected {self.config.N_METABOLITES}")
            m_norm, m_mask = self.preprocess_metabolites(raw_m)
            out.update({
                "metabolite_raw": raw_m,
                "metabolite_values": m_norm,
                "metabolite_mask": m_mask,
                "metabolite_columns": m_cols,
            })
        else:
            out.update({
                "metabolite_raw": None,
                "metabolite_values": None,
                "metabolite_mask": None,
                "metabolite_columns": self.metabolite_columns,
            })

        return out

    def _batch_forward(self, protein_values, protein_mask, metabolite_values, metabolite_mask, batch_size=128, use_amp=True):
        n = None
        if protein_values is not None:
            n = len(protein_values)
        if metabolite_values is not None:
            n = len(metabolite_values)
        if n is None:
            raise ValueError("No input arrays")

        if protein_values is None:
            protein_values = np.zeros((n, self.config.N_PROTEINS), dtype=np.float32)
            protein_mask = np.zeros((n, self.config.N_PROTEINS), dtype=np.float32)
        if metabolite_values is None:
            metabolite_values = np.zeros((n, self.config.N_METABOLITES), dtype=np.float32)
            metabolite_mask = np.zeros((n, self.config.N_METABOLITES), dtype=np.float32)

        outputs = []
        self.model.eval()
        with torch.no_grad():
            for s in range(0, n, batch_size):
                e = min(n, s + batch_size)
                pv = torch.from_numpy(protein_values[s:e]).float().to(self.device)
                pm = torch.from_numpy(protein_mask[s:e]).float().to(self.device)
                mv = torch.from_numpy(metabolite_values[s:e]).float().to(self.device)
                mm = torch.from_numpy(metabolite_mask[s:e]).float().to(self.device)
                hp = (pm.sum(dim=1) > 0).float()
                hm = (mm.sum(dim=1) > 0).float()
                with amp_autocast(use_amp and self.device.type == "cuda"):
                    out = self.model(pv, pm, mv, mm, hp, hm)
                outputs.append({k: v.detach().float().cpu().numpy() for k, v in out.items()})
        merged = {}
        for k in outputs[0].keys():
            merged[k] = np.concatenate([o[k] for o in outputs], axis=0)
        return merged

    def impute_metabolites_array(self, metabolite_values: np.ndarray, metabolite_mask: np.ndarray, batch_size=128, n_iters=3) -> np.ndarray:
        current = metabolite_values.copy()
        for _ in range(max(1, n_iters)):
            out = self._batch_forward(None, None, current, metabolite_mask, batch_size=batch_size)
            pred = out["metabolite_pred"]
            current = metabolite_values * metabolite_mask + pred * (1.0 - metabolite_mask)
        return current.astype(np.float32)

    def impute_proteins_array(self, protein_values: np.ndarray, protein_mask: np.ndarray, batch_size=128, n_iters=3) -> np.ndarray:
        current = protein_values.copy()
        for _ in range(max(1, n_iters)):
            out = self._batch_forward(current, protein_mask, None, None, batch_size=batch_size)
            pred = out["protein_pred"]
            current = protein_values * protein_mask + pred * (1.0 - protein_mask)
        return current.astype(np.float32)

    def predict_metabolites_from_proteins_array(self, protein_values: np.ndarray, protein_mask: np.ndarray, batch_size=128) -> np.ndarray:
        out = self._batch_forward(protein_values, protein_mask, None, None, batch_size=batch_size)
        return out["cross_metabolite_pred"].astype(np.float32)

    def extract_embeddings_array(self, protein_values=None, protein_mask=None, metabolite_values=None, metabolite_mask=None, mode="fused", batch_size=128) -> np.ndarray:
        n = None
        if protein_values is not None:
            n = len(protein_values)
        if metabolite_values is not None:
            n = len(metabolite_values)
        if n is None:
            raise ValueError("No input arrays")

        if protein_values is None:
            protein_values = np.zeros((n, self.config.N_PROTEINS), dtype=np.float32)
            protein_mask = np.zeros((n, self.config.N_PROTEINS), dtype=np.float32)
        if metabolite_values is None:
            metabolite_values = np.zeros((n, self.config.N_METABOLITES), dtype=np.float32)
            metabolite_mask = np.zeros((n, self.config.N_METABOLITES), dtype=np.float32)

        feats = []
        self.model.eval()
        with torch.no_grad():
            for s in range(0, n, batch_size):
                e = min(n, s + batch_size)
                pv = torch.from_numpy(protein_values[s:e]).float().to(self.device)
                pm = torch.from_numpy(protein_mask[s:e]).float().to(self.device)
                mv = torch.from_numpy(metabolite_values[s:e]).float().to(self.device)
                mm = torch.from_numpy(metabolite_mask[s:e]).float().to(self.device)
                hp = (pm.sum(dim=1) > 0).float()
                hm = (mm.sum(dim=1) > 0).float()
                with amp_autocast(self.device.type == "cuda"):
                    z = self.model.extract_features(pv, pm, mv, mm, hp, hm, mode=mode)
                feats.append(z.detach().float().cpu().numpy())
        return np.concatenate(feats, axis=0)

    def impute_file(self, protein_path: Optional[str] = None, metabolite_path: Optional[str] = None, output_path: str = "imputed.csv", mode: str = "auto", eid_col: str = "eid", batch_size: int = 128, n_iters: int = 3) -> pd.DataFrame:
        data = self.load_arrays(protein_path, metabolite_path, eid_col=eid_col)
        eids = data["eids"]

        if mode == "auto":
            if metabolite_path is not None:
                mode = "metabolite"
            elif protein_path is not None:
                mode = "p2m"
            else:
                raise ValueError("Cannot infer imputation mode")

        if mode in ["metabolite", "metabolites"]:
            if data["metabolite_values"] is None:
                raise ValueError("Metabolite file is required for metabolite imputation")
            pred_norm = self.impute_metabolites_array(data["metabolite_values"], data["metabolite_mask"], batch_size=batch_size, n_iters=n_iters)
            pred_raw = self.inverse_metabolites(pred_norm)
            raw = data["metabolite_raw"].copy()
            mask = data["metabolite_mask"].astype(bool)
            out_raw = pred_raw
            out_raw[mask] = raw[mask]
            df = pd.DataFrame(out_raw, columns=data["metabolite_columns"])
            df.insert(0, eid_col, eids)
            write_table(df, output_path)
            return df

        if mode in ["protein", "proteins"]:
            if data["protein_values"] is None:
                raise ValueError("Protein file is required for protein imputation")
            pred_norm = self.impute_proteins_array(data["protein_values"], data["protein_mask"], batch_size=batch_size, n_iters=n_iters)
            pred_raw = self.inverse_proteins(pred_norm)
            raw = data["protein_raw"].copy()
            mask = data["protein_mask"].astype(bool)
            out_raw = pred_raw
            out_raw[mask] = raw[mask]
            df = pd.DataFrame(out_raw, columns=data["protein_columns"])
            df.insert(0, eid_col, eids)
            write_table(df, output_path)
            return df

        if mode in ["p2m", "protein_to_metabolite", "protein-to-metabolite"]:
            if data["protein_values"] is None:
                raise ValueError("Protein file is required for protein-to-metabolite imputation")
            pred_norm = self.predict_metabolites_from_proteins_array(data["protein_values"], data["protein_mask"], batch_size=batch_size)
            pred_raw = self.inverse_metabolites(pred_norm)
            if data["metabolite_raw"] is not None:
                raw = data["metabolite_raw"].copy()
                mask = data["metabolite_mask"].astype(bool)
                pred_raw[mask] = raw[mask]
                cols = data["metabolite_columns"]
            else:
                cols = self.metabolite_columns
            df = pd.DataFrame(pred_raw, columns=cols)
            df.insert(0, eid_col, eids)
            write_table(df, output_path)
            return df

        raise ValueError(f"Unknown imputation mode: {mode}")

    def embed_file(self, protein_path: Optional[str] = None, metabolite_path: Optional[str] = None, output_path: str = "embeddings.csv", mode: str = "fused", eid_col: str = "eid", batch_size: int = 128) -> pd.DataFrame:
        data = self.load_arrays(protein_path, metabolite_path, eid_col=eid_col)
        modes = ["protein", "metabolite", "fused"] if mode == "all" else [mode]
        result = pd.DataFrame({eid_col: data["eids"]})
        for m in modes:
            z = self.extract_embeddings_array(
                protein_values=data["protein_values"],
                protein_mask=data["protein_mask"],
                metabolite_values=data["metabolite_values"],
                metabolite_mask=data["metabolite_mask"],
                mode=m,
                batch_size=batch_size,
            )
            cols = [f"{m}_embedding_{i + 1:04d}" for i in range(z.shape[1])]
            zdf = pd.DataFrame(z, columns=cols)
            result = pd.concat([result, zdf], axis=1)
        write_table(result, output_path)
        return result


def load_model(pretrained_dir: Optional[str] = None, device: Optional[str] = None) -> MultiomicsLM:
    return MultiomicsLM(pretrained_dir=pretrained_dir, device=device)