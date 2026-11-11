"""
Costruisce il dataset nnU-Net del rene (Dataset904_KidneyTSMechanical) con soli criteri di esclusione meccanici:
split train/test stratificato, label binarie e link alle CT originali.
Gli input intermedi usati per costruirlo non sono piu' su disco: lo script documenta come e' stato fatto il dataset.

Uso: python build_kidney_mechanical_dataset.py
"""
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn.model_selection import train_test_split

sys.path[:0] = ['/home/grolo/thesis_pipeline/1_data']  # cartelle dei moduli del progetto importati qui
from select_stratified_split import group_study_type

TS = "/mnt/sda2/DATASETS/TOTALSEGMENTATOR"
RAW = "/mnt/sda2/grolo/nnUNet/raw"
OLD = f"{RAW}/Dataset902_KidneyTS"
EXCL = "/mnt/sda2/grolo/nnUNet_data/kidney_excluded_test"
NAME = "Dataset904_KidneyTSMechanical"
OUT = f"{RAW}/{NAME}"
SEED = 42


def ids(d, suffix):
    return sorted(f[: -len(suffix)] for f in os.listdir(d) if f.endswith(suffix))


def main():
    if os.path.exists(OUT):
        sys.exit(f"{OUT} esiste gia': non sovrascrivo")
    old_tr, old_ts = ids(f"{OLD}/labelsTr", ".nii.gz"), ids(f"{OLD}/labelsTs", ".nii.gz")
    ex = pd.read_csv(f"{EXCL}/cases.csv")
    ex = ex[ex.included]
    assert not (set(ex.case_id) & (set(old_tr) | set(old_ts))), "sovrapposizione fra esclusi e Dataset902"
    meta = pd.read_csv(f"{TS}/meta.csv", sep=";", encoding="utf-8-sig").set_index("image_id")
    ex["group"] = [group_study_type(meta.loc[c, "study_type"]) if c in meta.index else "other" for c in ex.case_id]
    ex["strat"] = ex.group + "|" + ex.original_status
    n_add = int(round(len(ex) * len(old_ts) / (len(old_tr) + len(old_ts))))
    # strati con un solo caso non sono stratificabili: vanno in train (come in select_stratified_split.py)
    counts = ex.strat.value_counts()
    strat_ok = ex[ex.strat.map(counts) >= 2]
    singles = ex[ex.strat.map(counts) < 2]
    add_tr, add_ts = train_test_split(strat_ok.case_id, test_size=n_add, stratify=strat_ok.strat, random_state=SEED)  # (train, test)
    add_tr = list(add_tr) + list(singles.case_id)
    split = {"train": sorted(old_tr + list(add_tr)), "test": sorted(old_ts + list(add_ts))}
    assert not set(split["train"]) & set(split["test"])
    assert len(add_ts) == n_add, (len(add_ts), n_add)  # controllo dell'ordine (train, test)
    assert len(split["train"]) + len(split["test"]) == len(old_tr) + len(old_ts) + len(ex)

    for d in ["imagesTr", "labelsTr", "imagesTs", "labelsTs"]:
        os.makedirs(f"{OUT}/{d}")
    rows = []
    for part, tag in [("train", "Tr"), ("test", "Ts")]:
        for c in split[part]:
            if c in old_tr:
                src_lab, origin = f"{OLD}/labelsTr/{c}.nii.gz", "Dataset902_train"
            elif c in old_ts:
                src_lab, origin = f"{OLD}/labelsTs/{c}.nii.gz", "Dataset902_test"
            else:
                src_lab, origin = f"{EXCL}/labelsTs/{c}.nii.gz", "ex_" + ex.set_index("case_id").loc[c, "original_status"]
            ct = f"{TS}/{c}/ct.nii.gz"
            img, lab = sitk.ReadImage(ct), sitk.ReadImage(src_lab)  # controlli meccanici
            assert img.GetSize() == lab.GetSize(), c
            nvox = int(sitk.GetArrayViewFromImage(lab).sum())
            assert nvox > 0, c
            os.symlink(ct, f"{OUT}/images{tag}/{c}_0000.nii.gz")
            shutil.copy2(src_lab, f"{OUT}/labels{tag}/{c}.nii.gz")
            rows.append(dict(case_id=c, split=part, origin=origin, label_voxels=nvox,
                             study_type=meta.loc[c, "study_type"] if c in meta.index else ""))
    json.dump({"channel_names": {"0": "CT"}, "labels": {"background": 0, "kidney": 1},
               "numTraining": len(split["train"]), "file_ending": ".nii.gz", "name": NAME,
               "description": "TotalSegmentator, rene binario (kidney_left OR kidney_right), soli criteri meccanici: "
                              "tutti i casi con rene e CT leggibile (vedi build_kidney_mechanical_dataset.py)."},
              open(f"{OUT}/dataset.json", "w"), indent=2)
    json.dump(split, open(f"{OUT}/split_train_test.json", "w"), indent=1)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/cases.csv", index=False)
    print(df.groupby(["split", "origin"]).size().to_string())
    print("totale train", len(split["train"]), "| test", len(split["test"]))


if __name__ == "__main__":
    main()
