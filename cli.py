import argparse
import torch
import numpy as np
from multiomics_ensemble.ensemble import load_ensemble_models, ensemble_predict

def main():
    parser = argparse.ArgumentParser(description="Multiomics Ensemble Inference CLI")
    parser.add_argument('--global-rep', type=str, required=True,
                        help="Path to the global representations npz file.")
    parser.add_argument('--model-pattern', type=str, required=True,
                        help='Glob pattern for ensemble model files. Example: "/path/to/*_filtered/best_model.pt"')
    parser.add_argument('--num-metabolites', type=int, default=100,
                        help="Number of metabolite features expected (default: 100).")
    parser.add_argument('--protein-data', type=str, default=None,
                        help="Path to a .npy file containing protein data for a new sample.")
    parser.add_argument('--metabolite-data', type=str, default=None,
                        help="Path to a .npy file containing metabolite data for a new sample.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load global protein representations.
    global_data = np.load(args.global_rep)
    keys = sorted(list(global_data.files))
    if not keys:
        raise ValueError("No global features found in the provided npz file.")
    filtered_global_features = np.stack([global_data[k] for k in keys], axis=0)
    num_proteins = filtered_global_features.shape[0]
    print(f"Global features: {num_proteins} proteins with global_dim={filtered_global_features.shape[1]}")

    hidden_size = 768

    ensemble_models = load_ensemble_models(
        args.model_pattern,
        device,
        hidden_size,
        num_proteins,
        args.num_metabolites,
        filtered_global_features
    )

    # Load new sample data.
    if args.protein_data and args.metabolite_data:
        protein_data = np.load(args.protein_data)
        metabolite_data = np.load(args.metabolite_data)
        if protein_data.shape[0] != num_proteins:
            raise ValueError("Protein data dimension does not match the global features.")
        if metabolite_data.shape[0] != args.num_metabolites:
            raise ValueError("Metabolite data dimension does not match expected num_metabolites.")
    else:
        print("No sample data provided; generating random sample data.")
        protein_data = np.random.rand(num_proteins)
        metabolite_data = np.random.rand(args.num_metabolites)

    predictions = ensemble_predict(ensemble_models, protein_data, metabolite_data, device)
    print("\nEnsemble predictions for the new sample:")
    for disease, prob in predictions.items():
        print(f"{disease}: {prob:.4f}")

if __name__ == '__main__':
    main()