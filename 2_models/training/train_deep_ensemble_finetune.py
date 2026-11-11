"""
Training di un fold del deep ensemble a partire da pesi pretrained di TotalSegmentator
(caricamento permissivo: si saltano le teste di segmentazione).

Uso: python train_deep_ensemble_finetune.py --dataset <id> --fold <k> --pretrained_weights <checkpoint> --output_folder_suffix <suffisso>
"""
import argparse
import os

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
    p.add_argument("--pretrained_weights", required=True,
                    help="Path al checkpoint_final.pth da cui inizializzare (es. quello del "
                         "TotalSegmentator task 291, fold_0). Caricamento permissivo: le chiavi "
                         "con shape incompatibile vengono saltate (loggate) invece di causare crash.")
    p.add_argument("--output_folder_suffix", required=True,
                    help="Suffisso per la cartella di output, per NON scrivere sopra il run "
                         "from-scratch gia' completato (che usa "
                         "'{trainer}__{plans_identifier}__{configuration}/fold_N'). La cartella "
                         "finale sara' '{trainer}__{plans_identifier}__{configuration}{suffix}/fold_N'. "
                         "Esempio: '__finetune_pretrained291'.")
    p.add_argument("--export_validation_probabilities", action="store_true", default=True,
                    help="Salva le softmax della validazione finale (.npz), utile per il "
                         "deep ensemble (equivalente al flag --npz della CLI).")
    return p.parse_args()


def load_pretrained_weights_lenient(network, fname, verbose=True):
    """Come nnunetv2.run.load_pretrained_weights.load_pretrained_weights, ma salta
    QUALUNQUE chiave con shape incompatibile (non solo '.seg_layers.') invece di
    sollevare AssertionError. Vedi docstring del modulo per il perche' e la verifica.

    IMPORTANTE: su GPU nnU-Net avvolge la rete con torch.compile (OptimizedModule),
    che aggiunge un prefisso '_orig_mod.' a TUTTE le chiavi di state_dict(). Il
    checkpoint pretrained non ha questo prefisso (salvato da un mod gia' "srotolato").
    Senza questo unwrap, 0/292 chiavi
    combaciano (falso "successo" silenzioso: nessun crash, ma nessun peso pretrained
    caricato). Replichiamo lo stesso smontaggio della funzione ufficiale nnU-Net."""
    from torch._dynamo import OptimizedModule
    mod = network._orig_mod if isinstance(network, OptimizedModule) else network

    saved_model = torch.load(fname, map_location="cpu", weights_only=False)
    pretrained_dict = saved_model['network_weights']
    model_dict = mod.state_dict()

    new_state = {}
    transferred, skipped_missing, skipped_shape = [], [], []
    for key, tensor in model_dict.items():
        if key not in pretrained_dict:
            skipped_missing.append(key)
            new_state[key] = tensor
            continue
        if pretrained_dict[key].shape != tensor.shape:
            skipped_shape.append((key, tuple(pretrained_dict[key].shape), tuple(tensor.shape)))
            new_state[key] = tensor
            continue
        new_state[key] = pretrained_dict[key]
        transferred.append(key)

    mod.load_state_dict(new_state)

    print(f"[load_pretrained_weights_lenient] Trasferiti: {len(transferred)}/{len(model_dict)} tensori da {fname}")
    if skipped_missing:
        print(f"[load_pretrained_weights_lenient] Saltati (assenti nel pretrained): {len(skipped_missing)}")
        if verbose:
            for k in skipped_missing:
                print("    missing:", k)
    if skipped_shape:
        print(f"[load_pretrained_weights_lenient] Saltati (shape incompatibile): {len(skipped_shape)}")
        if verbose:
            for k, ps, ms in skipped_shape:
                print(f"    shape mismatch: {k}  pretrained={ps}  target={ms}")


def main():
    args = parse_args()

    from nnunetv2.run.run_training import get_trainer_from_args

    device = torch.device("cuda")
    nnunet_trainer = get_trainer_from_args(
        args.dataset, args.configuration, args.fold,
        trainer_name="nnUNetTrainer", plans_identifier=args.plans_identifier, device=device)

    nnunet_trainer.num_epochs = args.num_epochs
    print(f"[train_deep_ensemble_finetune] num_epochs forzato a {nnunet_trainer.num_epochs} "
          f"(default libreria: 250)")

    # IMPORTANTE: get_trainer_from_args costruisce di default output_folder come
    # '{trainer}__{plans_identifier}__{configuration}/fold_N', IDENTICO a quello del run
    # from-scratch gia' completato (stesso trainer_name/plans_identifier/configuration/dataset).
    # Senza questo redirect, questo script sovrascriverebbe i checkpoint del from-scratch.
    # dataset_root = nnUNet_results/{dataset_name} (un livello sopra output_folder_base attuale).
    dataset_root = os.path.dirname(nnunet_trainer.output_folder_base)
    new_folder_name = f"nnUNetTrainer__{args.plans_identifier}__{args.configuration}{args.output_folder_suffix}"
    nnunet_trainer.output_folder_base = os.path.join(dataset_root, new_folder_name)
    nnunet_trainer.output_folder = os.path.join(nnunet_trainer.output_folder_base, f"fold_{args.fold}")
    os.makedirs(nnunet_trainer.output_folder, exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now()
    nnunet_trainer.log_file = os.path.join(
        nnunet_trainer.output_folder,
        "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" % (
            timestamp.year, timestamp.month, timestamp.day,
            timestamp.hour, timestamp.minute, timestamp.second))
    print(f"[train_deep_ensemble_finetune] output_folder rediretto a: {nnunet_trainer.output_folder}")

    # Costruisce la rete (necessario prima di poterci caricare i pesi pretrained dentro).
    # run_training() vedra' was_initialized=True e non la ricostruira' da capo (verificato
    # leggendo il sorgente di nnUNetTrainer: on_train_start() chiama initialize() solo
    # "if not self.was_initialized").
    nnunet_trainer.initialize()
    load_pretrained_weights_lenient(nnunet_trainer.network, args.pretrained_weights)

    cudnn.deterministic = False
    cudnn.benchmark = True

    nnunet_trainer.run_training()
    nnunet_trainer.perform_actual_validation(args.export_validation_probabilities)


if __name__ == "__main__":
    main()
