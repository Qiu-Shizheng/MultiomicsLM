import argparse
import json

from .core import MultiomicsLM
from .finetune import finetune_disease_model


def add_common_model_args(p):
    p.add_argument("--pretrained-dir", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eid-col", type=str, default="eid")


def main():
    parser = argparse.ArgumentParser(prog="multiomicslm", description="MultiomicsLM command line interface")
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info")
    p_info.add_argument("--pretrained-dir", type=str, default=None)
    p_info.add_argument("--device", type=str, default=None)

    p_impute = sub.add_parser("impute")
    add_common_model_args(p_impute)
    p_impute.add_argument("--protein", type=str, default=None)
    p_impute.add_argument("--metabolite", type=str, default=None)
    p_impute.add_argument("--output", type=str, required=True)
    p_impute.add_argument("--mode", type=str, default="auto", choices=["auto", "metabolite", "protein", "p2m"])
    p_impute.add_argument("--n-iters", type=int, default=3)

    p_embed = sub.add_parser("embed")
    add_common_model_args(p_embed)
    p_embed.add_argument("--protein", type=str, default=None)
    p_embed.add_argument("--metabolite", type=str, default=None)
    p_embed.add_argument("--output", type=str, required=True)
    p_embed.add_argument("--mode", type=str, default="fused", choices=["protein", "metabolite", "fused", "protein_enriched", "metabolite_enriched", "all"])

    p_finetune = sub.add_parser("finetune")
    add_common_model_args(p_finetune)
    p_finetune.add_argument("--protein", type=str, default=None)
    p_finetune.add_argument("--metabolite", type=str, default=None)
    p_finetune.add_argument("--labels", type=str, required=True)
    p_finetune.add_argument("--output-dir", type=str, required=True)
    p_finetune.add_argument("--label-col", type=str, default=None)
    p_finetune.add_argument("--baseline-date-col", type=str, default=None)
    p_finetune.add_argument("--diagnosis-date-col", type=str, default=None)
    p_finetune.add_argument("--task", type=str, default="baseline", choices=["baseline", "current", "diagnosis", "future", "future_all", "prediction"])
    p_finetune.add_argument("--modality", type=str, default="fused", choices=["protein", "metabolite", "fused"])
    p_finetune.add_argument("--finetune", type=str, default="frozen", choices=["frozen", "full"])
    p_finetune.add_argument("--epochs", type=int, default=50)
    p_finetune.add_argument("--lr-head", type=float, default=1e-4)
    p_finetune.add_argument("--lr-encoder", type=float, default=1e-6)
    p_finetune.add_argument("--weight-decay", type=float, default=1e-4)
    p_finetune.add_argument("--val-ratio", type=float, default=0.2)
    p_finetune.add_argument("--patience", type=int, default=10)
    p_finetune.add_argument("--threshold", type=float, default=0.5)
    p_finetune.add_argument("--dropout", type=float, default=0.3)
    p_finetune.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.command == "info":
        model = MultiomicsLM(pretrained_dir=args.pretrained_dir, device=args.device)
        info = {
            "pretrained_dir": str(model.pretrained_dir),
            "device": str(model.device),
            "config": model.config.to_dict(),
            "n_protein_columns": len(model.protein_columns),
            "n_metabolite_columns": len(model.metabolite_columns),
        }
        print(json.dumps(info, indent=2))
        return

    if args.command == "impute":
        model = MultiomicsLM(pretrained_dir=args.pretrained_dir, device=args.device)
        model.impute_file(
            protein_path=args.protein,
            metabolite_path=args.metabolite,
            output_path=args.output,
            mode=args.mode,
            eid_col=args.eid_col,
            batch_size=args.batch_size,
            n_iters=args.n_iters,
        )
        print(f"Saved imputed table to {args.output}")
        return

    if args.command == "embed":
        model = MultiomicsLM(pretrained_dir=args.pretrained_dir, device=args.device)
        model.embed_file(
            protein_path=args.protein,
            metabolite_path=args.metabolite,
            output_path=args.output,
            mode=args.mode,
            eid_col=args.eid_col,
            batch_size=args.batch_size,
        )
        print(f"Saved embeddings to {args.output}")
        return

    if args.command == "finetune":
        summary = finetune_disease_model(
            pretrained_dir=args.pretrained_dir,
            protein_path=args.protein,
            metabolite_path=args.metabolite,
            labels_path=args.labels,
            output_dir=args.output_dir,
            eid_col=args.eid_col,
            label_col=args.label_col,
            baseline_date_col=args.baseline_date_col,
            diagnosis_date_col=args.diagnosis_date_col,
            task=args.task,
            modality=args.modality,
            finetune=args.finetune,
            device=args.device,
            batch_size=args.batch_size,
            max_epochs=args.epochs,
            learning_rate_head=args.lr_head,
            learning_rate_encoder=args.lr_encoder,
            weight_decay=args.weight_decay,
            val_ratio=args.val_ratio,
            patience=args.patience,
            threshold=args.threshold,
            dropout=args.dropout,
            seed=args.seed,
        )
        print(json.dumps(summary, indent=2))
        return


if __name__ == "__main__":
    main()