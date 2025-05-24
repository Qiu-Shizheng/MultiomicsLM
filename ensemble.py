import os
import glob
import torch
from multiomics_ensemble import models

# Mapping of disease codes to full disease names.
DISEASE_NAME_MAPPING = {
    "dementia": "Dementia",
    "parkinson": "Parkinson's disease",
    "copd": "COPD",
    "asthma": "Asthma",
    "RA": "Rheumatoid arthritis",
    "obesity": "Obesity",
    "diabetes": "T2D",
    "gout": "Gout",
    "hypertension": "Hypertension",
    "heart_failure": "Heart failure",
    "ischaemic_heart_disease": "Ischaemic heart disease",
    "atrial_fibrillation": "Atrial fibrillation",
    "stroke": "Stroke",
    "cerebral_infarction": "Cerebral infarction",
    "Breast_cancer": "Breast cancer",
    "Colorectal_cancer": "Colon cancer",
    "Lung_cancer": "Lung cancer",
    "Prostate_cancer": "Prostate cancer",
    "skin_cancer": "Skin cancer",
    "glaucoma": "Glaucoma"
}


def load_ensemble_models(model_pattern, device, hidden_size, num_proteins, num_metabolites, global_features):
    """
    Loads multiple fine-tuned models from files found using the provided glob pattern.

    Parameters:
      model_pattern: Glob pattern for model files.
      device: Torch device.
      hidden_size: Model hidden dimension.
      num_proteins: Number of protein features.
      num_metabolites: Number of metabolite features.
      global_features: Numpy array of global features.

    Returns:
      A dictionary whose keys are disease full names and values are the corresponding models.
    """
    model_files = glob.glob(model_pattern)
    if not model_files:
        raise ValueError(f"No model files found with pattern: {model_pattern}")
    ensemble_models = {}
    for model_file in model_files:

        disease_dir = os.path.dirname(model_file)
        disease_code = os.path.basename(disease_dir).replace("_filtered", "")
        # Map the disease code to a full disease name if available.
        disease_full_name = DISEASE_NAME_MAPPING.get(disease_code, disease_code)
        # Initialize the model instance.
        model = models.MultiModalBERTModel_Modified(
            hidden_size=hidden_size,
            num_proteins=num_proteins,
            num_metabolites=num_metabolites,
            dropout_prob=0.2,
            global_features=global_features
        )
        state = torch.load(model_file, map_location=device)
        # If state contains "model_state_dict", use it; otherwise, assume state is the state dict.
        if isinstance(state, dict) and "model_state_dict" in state:
            state_dict = state["model_state_dict"]
        else:
            state_dict = state
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        ensemble_models[disease_full_name] = model
        print(f"Loaded model for disease '{disease_full_name}' from {model_file}")
    return ensemble_models


def ensemble_predict(ensemble_models, protein_data, metabolite_data, device):
    """
    Given a new sample's multiomics data, computes prediction probabilities for each disease.

    Parameters:
      ensemble_models: Dictionary of models keyed by disease full name.
      protein_data: Numpy array with shape (num_proteins,).
      metabolite_data: Numpy array with shape (num_metabolites,).
      device: Torch device.

    Returns:
      A dictionary whose keys are disease full names and values are the prediction probabilities.
    """
    predictions = {}
    protein_tensor = torch.tensor(protein_data, dtype=torch.float32).unsqueeze(0).to(device)
    metabolite_tensor = torch.tensor(metabolite_data, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        for disease, model in ensemble_models.items():
            outputs = model(protein_tensor, metabolite_tensor)
            logits = outputs["itm_logits"]
            prob = torch.sigmoid(logits).item()  # scalar value
            predictions[disease] = prob
    return predictions