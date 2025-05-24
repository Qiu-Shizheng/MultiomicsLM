import torch
import torch.nn as nn
import torch.nn.functional as F


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


class MultiModalBERTModel_Modified(nn.Module):
    def __init__(self, hidden_size, num_proteins, num_metabolites, dropout_prob=0.2, global_features=None):
        """
        Initializes the multi-modal BERT model.

        Parameters:
          hidden_size: Hidden dimension size.
          num_proteins: Number of protein features.
          num_metabolites: Number of metabolite features.
          dropout_prob: Dropout probability.
          global_features: Numpy array of global protein features with shape [num_proteins, global_dim].
        """
        super(MultiModalBERTModel_Modified, self).__init__()
        self.hidden_size = hidden_size
        self.num_proteins = num_proteins
        self.num_metabolites = num_metabolites

        # Protein branch encoder
        self.protein_encoder = ProteinBERTEncoder(seq_len=num_proteins, hidden_size=hidden_size,
                                                  num_layers=4, num_heads=8, dropout=dropout_prob)
        # Metabolite branch: project 1-dim to hidden_size then encode
        self.metab_proj = nn.Linear(1, hidden_size)
        self.metabolite_encoder = MetaboliteBERTEncoder(seq_len=num_metabolites, hidden_size=hidden_size,
                                                        num_layers=4, num_heads=8, dropout=dropout_prob)
        # Global protein representation projection
        if global_features is None:
            raise ValueError("global_features must be provided")
        self.register_buffer('global_features', torch.tensor(global_features, dtype=torch.float32))
        self.global_proj = nn.Linear(self.global_features.shape[1], hidden_size)

        # Gated fusion module
        self.gated_fusion = GatedFusion(hidden_size)
        # Classification head for disease classification (binary logit)
        self.itm_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size // 2, 1)
        )

        # Pretraining task heads (not used during inference)
        self.mlm_head_protein = nn.Linear(hidden_size, 1)
        self.mlm_head_metabolite = nn.Linear(hidden_size, 1)
        self.output_layer_protein = nn.Linear(hidden_size, 1)
        self.output_layer_metabolite = nn.Linear(hidden_size, 1)

    def forward(self, *args, **kwargs):
        """
        Forward pass expects two inputs: proteins and metabolites.

        Parameters:
          proteins: Tensor of shape [B, num_proteins]
          metabolites: Tensor of shape [B, num_metabolites]

        Returns:
          A dictionary containing at least:
            - 'itm_logits': classification logit (before sigmoid)
        """
        proteins = None
        metabolites = None

        if len(args) >= 2:
            proteins, metabolites = args[0], args[1]
        if proteins is None and 'proteins' in kwargs:
            proteins = kwargs['proteins']
        if metabolites is None and 'metabolites' in kwargs:
            metabolites = kwargs['metabolites']
        if proteins is None or metabolites is None:
            raise ValueError("Missing required inputs: proteins and metabolites.")

        B = proteins.size(0)
        # Protein branch
        projected_global = self.global_proj(self.global_features)  # [num_proteins, hidden_size]
        projected_global = projected_global.unsqueeze(0)  # [1, num_proteins, hidden_size]
        prot_inputs = proteins.unsqueeze(-1) * projected_global  # [B, num_proteins, hidden_size]
        prot_encoded = self.protein_encoder(prot_inputs)  # [B, num_proteins, hidden_size]
        protein_global = prot_encoded.mean(dim=1)  # [B, hidden_size]

        # Metabolite branch
        metab_inputs = metabolites.unsqueeze(-1)  # [B, num_metabolites, 1]
        metab_inputs = self.metab_proj(metab_inputs)  # [B, num_metabolites, hidden_size]
        metab_encoded = self.metabolite_encoder(metab_inputs)  # [B, num_metabolites, hidden_size]
        metabolite_global = metab_encoded.mean(dim=1)  # [B, hidden_size]

        # Fusion using gated fusion
        fusion = self.gated_fusion(protein_global, metabolite_global)

        itm_logits = self.itm_classifier(fusion).squeeze(-1)
        outputs = {
            'itm_logits': itm_logits,
            'prot_global': protein_global,
            'metab_global': metabolite_global,
        }
        return outputs