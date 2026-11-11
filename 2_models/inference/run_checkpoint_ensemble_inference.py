"""
Inferenza del checkpoint ensemble: i membri sono gli snapshot di un solo training con learning rate ciclico.
Output come il deep ensemble: segmentazione, logit medi e logit dei singoli membri.

Uso: python run_checkpoint_ensemble_inference.py --model_folder <cartella del training> -i <imagesTs> -o <out>
"""
import argparse
import json
import os
import sys
from os.path import join

import torch

sys.path[:0] = ['/home/grolo/thesis_pipeline/2_models/inference']  # cartelle dei moduli del progetto importati qui
from predict_from_raw_data import nnUNetPredictor  # noqa: E402  (versione patchata: fold_logits + T)

REITANI_SNAPSHOTS = [
    "checkpoint_cycle0_ep396.pth", "checkpoint_cycle0_ep399.pth",
    "checkpoint_cycle1_ep796.pth", "checkpoint_cycle1_ep799.pth",
    "checkpoint_cycle2_ep1196.pth", "checkpoint_cycle2_ep1199.pth",
]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_folder", required=True,
                   help="Cartella del training (quella che contiene fold_N/)")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--snapshots", nargs="+", default=REITANI_SNAPSHOTS)
    p.add_argument("-i", required=True, help="Cartella di input ({case}_0000.nii.gz)")
    p.add_argument("-o", required=True, help="Cartella di output")
    p.add_argument("--npp", type=int, default=3, help="processi di preprocessing")
    p.add_argument("--nps", type=int, default=3, help="processi di export")
    args = p.parse_args()

    fold_dir = join(args.model_folder, f"fold_{args.fold}")
    for s in args.snapshots:
        if not os.path.isfile(join(fold_dir, s)):
            raise FileNotFoundError(join(fold_dir, s))

    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                perform_everything_on_device=True, device=torch.device("cuda"),
                                verbose=False, verbose_preprocessing=False, allow_tqdm=True)
    # Inizializza con il primo snapshot (rete, plans, dataset.json), poi sostituisce la lista dei pesi
    predictor.initialize_from_trained_model_folder(args.model_folder, (args.fold,), checkpoint_name=args.snapshots[0])
    params = []
    for s in args.snapshots:
        ck = torch.load(join(fold_dir, s), map_location="cpu", weights_only=False)
        params.append(ck["network_weights"])
        print(f"[checkpoint_ensemble] membro {len(params) - 1}: {s} (epoch {ck['current_epoch'] - 1})")
    predictor.list_of_parameters = params
    predictor.use_folds = list(range(len(params)))  # fold_logits/{case}_fold_{k}: k = indice dello snapshot
    predictor.save_logits = True
    predictor.temperature = 1.0

    os.makedirs(args.o, exist_ok=True)
    with open(join(args.o, "checkpoint_ensemble_members.json"), "w") as f:
        json.dump({"model_folder": args.model_folder, "fold": args.fold,
                   "members": {str(k): s for k, s in enumerate(args.snapshots)},
                   "aggregation": "mean of logits", "temperature": 1.0}, f, indent=2)

    predictor.predict_from_files(args.i, args.o, save_probabilities=False, overwrite=True,
                                 num_processes_preprocessing=args.npp,
                                 num_processes_segmentation_export=args.nps)


if __name__ == "__main__":
    main()
