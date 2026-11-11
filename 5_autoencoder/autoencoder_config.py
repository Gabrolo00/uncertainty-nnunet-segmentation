"""
Configurazione dell'autoencoder: percorsi, T* e tabella E[U | d] del modello scelto con AE_DATASET (default 904).
Si importa dagli altri script autoencoder_*.
"""
import os

_R = "/mnt/sda2/grolo/nnUNet/raw"
_O = "/mnt/sda2/grolo"
_P = _O
_K = "/mnt/sda2/grolo/humanitas_source"

CONFIGS = {
    "904": dict(
        T_STAR=2.0753725478881155,                                   # mediana di kidney/deep/loo_T_summary.csv
        TS_PRED=f"{_P}/kidney/predictions/deep",
        TS_RAW=f"{_R}/Dataset904_KidneyTSMechanical",
        # test esterno Humanitas con lo stesso modello
        HUM_PRED=f"/mnt/sda2/grolo/kidney/external/pred_deep", HUM_CT=f"/mnt/sda2/grolo/kidney/external/cropped/images", HUM_GT=f"/mnt/sda2/grolo/kidney/external/cropped/labelsTs",
        BASELINE=f"{_O}/kidney/deep/baseline_EU/baseline_EU_given_d_mm_inside_outside.npz",
        ROOT=f"{_O}/kidney/autoencoder",
        AE_TAG="d4_lat16_it30000",                                   # configurazione scelta sul 904
    ),
}

DS = os.environ.get("AE_DATASET", "904")
if DS not in CONFIGS:
    raise ValueError(f"AE_DATASET={DS!r} non valido; scegliere fra {list(CONFIGS)}")
C = CONFIGS[DS]
DATASET_DIR = f"{C['ROOT']}/dataset"
RUNS_DIR = f"{C['ROOT']}/runs"
EVAL_DIR = f"{C['ROOT']}/eval"
