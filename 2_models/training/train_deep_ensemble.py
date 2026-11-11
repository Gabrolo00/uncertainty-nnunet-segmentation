"""
Training di un fold del deep ensemble con nnU-Net (3d_fullres), con il numero di epoche impostabile.

Uso: python train_deep_ensemble.py --dataset <id> --fold <k> [--num_epochs 1000]
"""
import argparse

import torch
from torch.backends import cudnn


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="Dataset id o nome, es. 904 o Dataset904_KidneyTSMechanical")
    p.add_argument("--configuration", default="3d_fullres")
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--num_epochs", type=int, default=1000)
    p.add_argument("--plans_identifier", default="nnUNetPlans")
    p.add_argument("--export_validation_probabilities", action="store_true", default=True,
                    help="Salva le softmax della validazione finale (.npz), utile per il "
                         "deep ensemble (equivalente al flag --npz della CLI).")
    p.add_argument("--continue_training", action="store_true",
                    help="Riprende un fold interrotto (es. riavvio del server) da checkpoint_latest.pth, "
                         "come il flag --c di nnUNetv2_train.")
    return p.parse_args()


def main():
    args = parse_args()

    from nnunetv2.run.run_training import get_trainer_from_args, maybe_load_checkpoint

    device = torch.device("cuda")
    nnunet_trainer = get_trainer_from_args(
        args.dataset, args.configuration, args.fold,
        trainer_name="nnUNetTrainer", plans_identifier=args.plans_identifier, device=device)

    nnunet_trainer.num_epochs = args.num_epochs
    print(f"[train_deep_ensemble] num_epochs forzato a {nnunet_trainer.num_epochs} "
          f"(default libreria: 250)")

    maybe_load_checkpoint(nnunet_trainer, continue_training=args.continue_training, validation_only=False,
                          pretrained_weights_file=None)

    cudnn.deterministic = False
    cudnn.benchmark = True

    nnunet_trainer.run_training()
    nnunet_trainer.perform_actual_validation(args.export_validation_probabilities)


if __name__ == "__main__":
    main()
