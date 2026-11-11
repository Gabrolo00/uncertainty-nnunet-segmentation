"""
Ritaglia in z le CT Humanitas (whole body) attorno ai reni, usando la predizione veloce di un solo modello
unita alla reference, per ridurre spazio e tempo dell'inferenza con l'ensemble completo.

Uso: python crop_humanitas_kidney.py --work_dir <cartella con pass1/ e immagini>
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import SimpleITK as sitk

BASE = "/mnt/sda2/grolo/humanitas_source"
MARGIN_MM = 50.0


def main():
    # --work_dir: cartella del modello
    # pass1/ si legge e cropped/ + crop_report.csv si scrivono in work_dir; TAC, label, corpo e coorte restano
    # quelli comuni di BASE. Non sovrascrive un ritaglio esistente.
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default=BASE)
    W = ap.parse_args().work_dir
    if any(os.listdir(f"{W}/cropped/{d}") for d in ["images", "labelsTs", "body"] if os.path.isdir(f"{W}/cropped/{d}")):
        sys.exit(f"{W}/cropped contiene gia' file: non sovrascrivo")
    c = pd.read_csv(f"{BASE}/cohort.csv", dtype=str).fillna("")
    c = c[c.excluded == ""]  # esclusioni meccaniche (vedi cohort.csv / excluded.csv)
    for d in ["images", "labelsTs", "body"]:
        os.makedirs(f"{W}/cropped/{d}", exist_ok=True)
    rows = []
    for case in c.case:
        fp = f"{W}/pass1/{case}.nii.gz"
        if not os.path.exists(fp):
            rows.append(dict(case=case, status="manca_pass1"))
            continue
        ct = sitk.ReadImage(f"{BASE}/images/{case}_0000.nii.gz")
        p = sitk.GetArrayFromImage(sitk.ReadImage(fp)) > 0
        g = sitk.GetArrayFromImage(sitk.ReadImage(f"{BASE}/labelsTs/{case}.nii.gz")) > 0
        v = np.prod(ct.GetSpacing()) / 1000
        m = int(np.ceil(MARGIN_MM / ct.GetSpacing()[2]))
        zp = np.where(p.any((1, 2)))[0]
        zg = np.where(g.any((1, 2)))[0]
        ends = [z for z in (zp, zg) if len(z)]
        z0 = max(0, min(z[0] for z in ends) - m)
        z1 = min(p.shape[0] - 1, max(z[-1] for z in ends) + m)
        # quanto la GT ha allargato la finestra rispetto alla sola predizione (trasparenza)
        p0, p1 = (max(0, zp[0] - m), min(p.shape[0] - 1, zp[-1] + m)) if len(zp) else (None, None)
        extended = p0 is None or z0 < p0 or z1 > p1
        inside = np.zeros(p.shape[0], bool)
        inside[z0:z1 + 1] = True
        for name, src in [("images", f"{BASE}/images/{case}_0000.nii.gz"),
                          ("labelsTs", f"{BASE}/labelsTs/{case}.nii.gz"),
                          ("body", f"{BASE}/body/{case}.nii.gz")]:
            img = sitk.ReadImage(src)
            sitk.WriteImage(img[:, :, z0:z1 + 1], f"{W}/cropped/{name}/" + os.path.basename(src))
        rows.append(dict(case=case, status="ok", z0=z0, z1=z1, n_slices_full=p.shape[0], n_slices_crop=z1 - z0 + 1,
                         pass1_pred_ml=round(p.sum() * v, 1), window_extended_by_gt=bool(extended),
                         slices_added_by_gt=int((p0 - z0) + (z1 - p1)) if p0 is not None else int(z1 - z0 + 1),
                         gt_outside_crop_ml=round(g[~inside].sum() * v, 2),
                         pass1_fp_outside_crop_ml=round(p[~inside].sum() * v, 2),
                         pass1_dice_full=round(2 * (p & g).sum() / max(p.sum() + g.sum(), 1), 4)))
        print(rows[-1], flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(f"{W}/crop_report.csv", index=False)
    print(r.status.value_counts().to_string())
    ok = r[r.status == "ok"]
    print("finestre allargate dalla GT:", int(ok.window_extended_by_gt.sum()),
          "| fette aggiunte dalla GT (mediana tra questi):",
          ok.loc[ok.window_extended_by_gt, "slices_added_by_gt"].median())
    print(ok[["n_slices_full", "n_slices_crop", "gt_outside_crop_ml", "pass1_fp_outside_crop_ml", "pass1_dice_full"]]
          .describe().loc[["min", "50%", "max"]].to_string())


if __name__ == "__main__":
    main()
