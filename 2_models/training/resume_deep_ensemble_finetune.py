"""
Riprende un training di fine-tuning interrotto, dallo stato completo salvato in checkpoint_latest.pth.

Uso: python resume_deep_ensemble_finetune.py --dataset <id> --fold <k> --output_folder_suffix <suffisso>
"""
import argparse
import os

import torch
from torch.backends import cudnn


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--configuration", default="3d_fullres")
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--num_epochs", type=int, default=1000)
    p.add_argument("--plans_identifier", default="nnUNetPlans")
    p.add_argument("--output_folder_suffix", required=True,
                    help="Stesso suffisso usato nel run originale, es. '__finetune_pretrained291'.")
    p.add_argument("--export_validation_probabilities", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()

    from nnunetv2.run.run_training import get_trainer_from_args

    device = torch.device("cuda")
    nnunet_trainer = get_trainer_from_args(
        args.dataset, args.configuration, args.fold,
        trainer_name="nnUNetTrainer", plans_identifier=args.plans_identifier, device=device)

    nnunet_trainer.num_epochs = args.num_epochs

    # Stesso redirect di output_folder di train_deep_ensemble_finetune.py, per
    # puntare alla cartella del run interrotto (non quella from-scratch).
    dataset_root = os.path.dirname(nnunet_trainer.output_folder_base)
    new_folder_name = f"nnUNetTrainer__{args.plans_identifier}__{args.configuration}{args.output_folder_suffix}"
    nnunet_trainer.output_folder_base = os.path.join(dataset_root, new_folder_name)
    nnunet_trainer.output_folder = os.path.join(nnunet_trainer.output_folder_base, f"fold_{args.fold}")

    checkpoint_path = os.path.join(nnunet_trainer.output_folder, "checkpoint_latest.pth")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Nessun checkpoint_latest.pth trovato in {nnunet_trainer.output_folder}")

    from datetime import datetime
    timestamp = datetime.now()
    nnunet_trainer.log_file = os.path.join(
        nnunet_trainer.output_folder,
        "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" % (
            timestamp.year, timestamp.month, timestamp.day,
            timestamp.hour, timestamp.minute, timestamp.second))
    print(f"[resume_deep_ensemble_finetune] output_folder: {nnunet_trainer.output_folder}")
    print(f"[resume_deep_ensemble_finetune] riprendo da: {checkpoint_path}")

    nnunet_trainer.load_checkpoint(checkpoint_path)
    print(f"[resume_deep_ensemble_finetune] current_epoch dopo il resume: {nnunet_trainer.current_epoch}")

    cudnn.deterministic = False
    cudnn.benchmark = True

    nnunet_trainer.run_training()
    nnunet_trainer.perform_actual_validation(args.export_validation_probabilities)


if __name__ == "__main__":
    main()
