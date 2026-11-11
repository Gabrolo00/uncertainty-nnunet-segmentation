"""
Training del checkpoint ensemble: un solo training nnU-Net con learning rate ciclico; gli snapshot salvati
alla fine di ogni ciclo sono i membri (Zhao et al. 2022, "Efficient Bayesian Uncertainty Estimation for nnU-Net",
MICCAI 2022, arXiv:2212.06278). Senza --pretrained_weights la rete parte da pesi casuali.

Uso: python train_checkpoint_ensemble.py --dataset <id> --fold 0 --output_folder_suffix cyclr_scratch
"""
import argparse
import os
from datetime import datetime
from os.path import join

import numpy as np
import torch
from torch.backends import cudnn

from train_deep_ensemble_finetune import load_pretrained_weights_lenient


def cyclic_lr(epoch, cycle_length, alpha_r, alpha0, gamma, epsilon):
    """LR dell'epoca `epoch` (0-based), eq. 4 di Zhao et al. 2022."""
    tc = epoch % cycle_length
    if tc == 0:
        return alpha_r
    return alpha0 * (1.0 - min(tc, gamma * cycle_length) / cycle_length) ** epsilon


def check_lr_formula():
    """Confronta la formula con valori di LR di riferimento (controllo all'avvio)."""
    logged = {0: 0.1, 40: 0.00910, 80: 0.00818, 200: 0.00536, 320: 0.00235, 360: 0.00235, 400: 0.1}
    for epoch, expected in logged.items():
        got = cyclic_lr(epoch, 400, 0.1, 0.01, 0.8, 0.9)
        assert abs(got - expected) < 5e-6, f"epoch {epoch}: {got:.5f} != {expected:.5f} (valore di riferimento)"
    print(f"[check_lr_formula] OK: la formula riproduce i {len(logged)} valori di LR di riferimento")


def make_cyclr_trainer_class():
    # import qui dentro: nnunetv2 va importato dopo aver letto gli argomenti
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from torch._dynamo import OptimizedModule

    class nnUNetTrainerCycLR(nnUNetTrainer):
        """nnUNetTrainer con LR ciclico e salvataggio degli snapshot di fine ciclo.
        Non ha un __init__ proprio: si ottiene cambiando la classe di un nnUNetTrainer
        gia' costruito (vedi main) e poi chiamando configure_cyclr()."""

        def configure_cyclr(self, num_cycles, cycle_length, checkpoints_per_cycle,
                            alpha_r, alpha0, gamma, epsilon):
            self.num_cycles = num_cycles
            self.cycle_length = cycle_length
            self.checkpoints_per_cycle = checkpoints_per_cycle
            self.alpha_r = alpha_r
            self.alpha0 = alpha0
            self.gamma = gamma
            self.epsilon = epsilon
            self.num_epochs = num_cycles * cycle_length

        def on_train_epoch_start(self):
            # Sostituisce lo step del PolyLRScheduler standard: l'LR e' impostato a mano.
            # (Lo scheduler creato da configure_optimizers esiste ma non viene mai usato.)
            self.network.train()
            lr = cyclic_lr(self.current_epoch, self.cycle_length, self.alpha_r,
                           self.alpha0, self.gamma, self.epsilon)
            for group in self.optimizer.param_groups:
                group['lr'] = lr
            cycle = self.current_epoch // self.cycle_length
            tc = self.current_epoch % self.cycle_length
            self.print_to_log_file('')
            self.print_to_log_file(f'Epoch {self.current_epoch}')
            self.print_to_log_file(f'Cycle={cycle + 1}/{self.num_cycles}, tc={tc}, '
                                   f'Current learning rate: {np.round(lr, decimals=5)}')
            self.logger.log('lrs', lr, self.current_epoch)

        def on_epoch_end(self):
            # Snapshot nelle ultime `checkpoints_per_cycle` epoche del ciclo. Va fatto PRIMA
            # di super().on_epoch_end(), che incrementa current_epoch.
            cycle = self.current_epoch // self.cycle_length
            tc = self.current_epoch % self.cycle_length
            if tc >= self.cycle_length - self.checkpoints_per_cycle:
                fname = join(self.output_folder, f'checkpoint_cycle{cycle}_ep{self.current_epoch}.pth')
                self.print_to_log_file(f'Saving cycle snapshot -> {fname}')
                self.save_checkpoint(fname)
            super().on_epoch_end()

        def save_checkpoint(self, filename: str) -> None:
            # Come nnUNetTrainer.save_checkpoint, con due differenze:
            #  - trainer_name='nnUNetTrainer', cosi' l'inferenza standard ritrova la classe
            #    (architettura identica, vedi docstring del modulo);
            #  - chiave 'cyclr' con i parametri del ciclo, per tracciabilita'.
            if self.local_rank != 0:
                return
            if self.disable_checkpointing:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')
                return
            mod = self.network.module if self.is_ddp else self.network
            if isinstance(mod, OptimizedModule):
                mod = mod._orig_mod
            checkpoint = {
                'network_weights': mod.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                'logging': self.logger.get_checkpoint(),
                '_best_ema': self._best_ema,
                'current_epoch': self.current_epoch + 1,
                'init_args': self.my_init_kwargs,
                'trainer_name': 'nnUNetTrainer',
                'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                'cyclr': {'num_cycles': self.num_cycles, 'cycle_length': self.cycle_length,
                          'checkpoints_per_cycle': self.checkpoints_per_cycle,
                          'alpha_r': self.alpha_r, 'alpha0': self.alpha0,
                          'gamma': self.gamma, 'epsilon': self.epsilon},
            }
            torch.save(checkpoint, filename)

    return nnUNetTrainerCycLR


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="Dataset id o nome, es. 903")
    p.add_argument("--configuration", default="3d_fullres")
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--plans_identifier", default="nnUNetPlans")
    p.add_argument("--pretrained_weights", default=None,
                   help="checkpoint_final.pth da cui inizializzare (task 291), caricamento permissivo. "
                        "Se omesso: pesi casuali (from scratch).")
    p.add_argument("--output_folder_suffix", required=True,
                   help="Suffisso della cartella di output, es. '__cyclr_pretrained291'")
    # Parametri del ciclo (default: 3 cicli da 400 epoche)
    p.add_argument("--num_cycles", type=int, default=3)
    p.add_argument("--cycle_length", type=int, default=400)
    p.add_argument("--checkpoints_per_cycle", type=int, default=10)
    p.add_argument("--alpha_r", type=float, default=0.1, help="LR di restart (tc=0)")
    p.add_argument("--alpha0", type=float, default=0.01, help="LR base")
    p.add_argument("--gamma", type=float, default=0.8, help="frazione del ciclo dopo cui l'LR resta costante")
    p.add_argument("--epsilon", type=float, default=0.9, help="esponente del decadimento")
    # Solo per il test breve
    p.add_argument("--num_iterations_per_epoch", type=int, default=None)
    p.add_argument("--num_val_iterations_per_epoch", type=int, default=None)
    p.add_argument("--save_every", type=int, default=None,
                   help="Ogni quante epoche salvare checkpoint_latest.pth (default nnU-Net: 50)")
    p.add_argument("--skip_final_validation", action="store_true",
                   help="Non eseguire la validazione finale sui casi del fold (solo test)")
    p.add_argument("--resume", action="store_true",
                   help="Riprende da checkpoint_latest.pth (niente pesi pretrained)")
    return p.parse_args()


def main():
    args = parse_args()
    check_lr_formula()

    from nnunetv2.run.run_training import get_trainer_from_args

    device = torch.device("cuda")
    trainer = get_trainer_from_args(args.dataset, args.configuration, args.fold,
                                    trainer_name="nnUNetTrainer",
                                    plans_identifier=args.plans_identifier, device=device)
    trainer.__class__ = make_cyclr_trainer_class()
    trainer.configure_cyclr(args.num_cycles, args.cycle_length, args.checkpoints_per_cycle,
                            args.alpha_r, args.alpha0, args.gamma, args.epsilon)
    if args.num_iterations_per_epoch is not None:
        trainer.num_iterations_per_epoch = args.num_iterations_per_epoch
    if args.num_val_iterations_per_epoch is not None:
        trainer.num_val_iterations_per_epoch = args.num_val_iterations_per_epoch
    if args.save_every is not None:
        trainer.save_every = args.save_every
    print(f"[train_checkpoint_ensemble] {args.num_cycles} cicli x {args.cycle_length} epoche = "
          f"{trainer.num_epochs} epoche, {args.checkpoints_per_cycle} snapshot per ciclo")

    # Redirect dell'output in una cartella con suffisso (stessa logica di train_deep_ensemble_finetune.py)
    dataset_root = os.path.dirname(trainer.output_folder_base)
    trainer.output_folder_base = join(
        dataset_root, f"nnUNetTrainer__{args.plans_identifier}__{args.configuration}{args.output_folder_suffix}")
    trainer.output_folder = join(trainer.output_folder_base, f"fold_{args.fold}")
    os.makedirs(trainer.output_folder, exist_ok=True)
    ts = datetime.now()
    trainer.log_file = join(trainer.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" % (
        ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second))
    print(f"[train_checkpoint_ensemble] output_folder: {trainer.output_folder}")

    trainer.initialize()
    if args.resume:
        latest = join(trainer.output_folder, "checkpoint_latest.pth")
        trainer.load_checkpoint(latest)
        print(f"[train_checkpoint_ensemble] resume da {latest}, current_epoch={trainer.current_epoch}")
    elif args.pretrained_weights:
        load_pretrained_weights_lenient(trainer.network, args.pretrained_weights)
    else:
        print("[train_checkpoint_ensemble] nessun --pretrained_weights: init random (from-scratch)")

    cudnn.deterministic = False
    cudnn.benchmark = True

    trainer.run_training()
    if not args.skip_final_validation:
        trainer.perform_actual_validation(True)


if __name__ == "__main__":
    main()
