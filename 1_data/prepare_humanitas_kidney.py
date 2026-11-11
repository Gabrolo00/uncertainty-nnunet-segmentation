"""
Prepara il test esterno Humanitas per il rene: scelta del contorno per paziente, conversione in NIfTI e
controllo qualita'. Tutti i pazienti utilizzabili sono casi di test.

Uso: python prepare_humanitas_kidney.py --stage cohort|convert|qa
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

CHK = "/mnt/sda2/grolo/humanitas_source/inventory"
OUT = "/mnt/sda2/grolo/humanitas_source"


import re

# Contorni clinici "Rene DX/SX" (e varianti di grafia) vs segmentazione automatica RayStation "Kidney_L/R".
# Esclusi i duplicati derivati ("Rene DX3", "Rene_DX_CT 2->CT 1").
RENE = re.compile(r"^rene[ _]?(dx|sx)$", re.I)
KIDNEY_AUTO = re.compile(r"^kidney_(l|r)$", re.I)


def cohort():
    s = pd.read_csv(f"{CHK}/structures.csv", dtype={"patient_id": str})
    s = s[s.roi_name.str.contains(r"rene|kidney", case=False)].copy()
    # RayStation tronca l'ID paziente a 8 caratteri (Varian ne usa 10): chiave = primi 8 caratteri.
    s["pkey"] = s.patient_id.str[:8]
    all_pat = sorted(s.pkey.unique())
    s = s[s.ref_series_in_dir & (s.n_points > 0)]
    rows = []
    for (pkey, d, f), x in s.groupby(["pkey", "dir", "rtstruct"]):
        names = x.roi_name.str.strip()
        for family, pat, rank in [("Rene (clinico)", RENE, 0), ("Kidney_L/R (RayStation auto)", KIDNEY_AUTO, 1)]:
            sel = x[names.str.match(pat)]
            side = sel.roi_name.str.strip().str.lower().map(
                lambda n: "dx" if n.endswith(("dx", "_r")) else ("sx" if n.endswith(("sx", "_l")) else ""))
            if set(side) == {"dx", "sx"} and len(sel) == 2:
                rows.append(dict(pkey=pkey, patient_id=x.patient_id.iloc[0], dir=d, rtstruct=f, gt_source=family,
                                 source_rank=rank, rois=",".join(sel.roi_name), approval=x.approval.iloc[0],
                                 date=x.structure_set_date.iloc[0], n_points=int(sel.n_points.sum()),
                                 contour_system=x.contour_system.iloc[0]))
                break  # se ci sono i contorni clinici si usano quelli
    g = pd.DataFrame(rows)
    g["appr_rank"] = (g.approval != "APPROVED").astype(int)
    g = g.sort_values(["pkey", "source_rank", "appr_rank", "date", "n_points"],
                      ascending=[True, True, True, False, False])
    c = g.drop_duplicates("pkey").copy()
    c["case"] = "hum" + c.pkey
    excluded = sorted(set(all_pat) - set(c.pkey))
    os.makedirs(OUT, exist_ok=True)
    c.drop(columns=["appr_rank", "source_rank"]).to_csv(f"{OUT}/cohort.csv", index=False)
    pd.DataFrame({"pkey": excluded,
                  "reason": "nessun RTSTRUCT con entrambi i reni contornati (stessa famiglia di nomi) e CT di riferimento"}) \
        .to_csv(f"{OUT}/excluded.csv", index=False)
    print(f"Pazienti (chiave 8 caratteri): {len(all_pat)}; coorte: {len(c)}; esclusi: {len(excluded)} {excluded}")
    print(c.groupby(["gt_source", "contour_system", "approval"]).size().to_string())


def convert():
    c = pd.read_csv(f"{OUT}/cohort.csv")
    py = "/home/aiart/anaconda3/envs/nnunet_medcam/bin/python"
    log = open(f"{OUT}/convert.log", "a")
    for i, r in c.iterrows():
        if os.path.exists(f"{OUT}/labelsTs/{r.case}.nii.gz"):
            continue
        tmp = f"{OUT}/_tmp"
        cmd = [py, f"{CHK}/convert_humanitas_case.py", "--dicom_dir", r.dir, "--rtstruct", r.rtstruct,
               "--case", r.case, "--out", tmp, "--roi", f"{r.rois}:kidney"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        ok = res.returncode == 0
        log.write(f"{r.case} rc={res.returncode} {res.stdout.strip()} {'' if ok else res.stderr[-500:]}\n")
        log.flush()
        if ok:
            for sub, dst in [("images", "images"), ("labels_kidney", "labelsTs")]:
                os.makedirs(f"{OUT}/{dst}", exist_ok=True)
                for f in os.listdir(f"{tmp}/{sub}"):
                    os.replace(f"{tmp}/{sub}/{f}", f"{OUT}/{dst}/{f}")
        print(f"[{i + 1}/{len(c)}] {r.case} {'ok' if ok else 'ERRORE'}", flush=True)


def qa():
    import SimpleITK as sitk
    from scipy import ndimage
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = pd.read_csv(f"{OUT}/cohort.csv")
    os.makedirs(f"{OUT}/qa_png", exist_ok=True)
    os.makedirs(f"{OUT}/body", exist_ok=True)
    rows = []
    for _, r in c.iterrows():
        fi, fl = f"{OUT}/images/{r.case}_0000.nii.gz", f"{OUT}/labelsTs/{r.case}.nii.gz"
        if not (os.path.exists(fi) and os.path.exists(fl)):
            rows.append(dict(case=r.case, converted=False))
            continue
        ct, lb = sitk.ReadImage(fi), sitk.ReadImage(fl)
        a, g = sitk.GetArrayFromImage(ct), sitk.GetArrayFromImage(lb) > 0
        sp = np.array(ct.GetSpacing())[::-1]
        v = np.prod(sp) / 1000
        # maschera del corpo dalla CT: HU > -400, componente connessa piu' grande, buchi riempiti per fetta
        b = a > -400
        lab, n = ndimage.label(b)
        if n:
            b = lab == (np.argmax(ndimage.sum(b, lab, range(1, n + 1))) + 1)
        b = np.stack([ndimage.binary_fill_holes(s) for s in b])
        bi = sitk.GetImageFromArray(b.astype(np.uint8))
        bi.CopyInformation(ct)
        sitk.WriteImage(bi, f"{OUT}/body/{r.case}.nii.gz")
        lab, n = ndimage.label(g)
        comps = sorted(ndimage.sum(g, lab, range(1, n + 1)) * v, reverse=True)
        zs = np.where(g.any((1, 2)))[0]
        rows.append(dict(
            case=r.case, converted=True, contour_system=r.contour_system, gt_source=r.gt_source, approval=r.approval,
            same_geometry=ct.GetSize() == lb.GetSize() and np.allclose(ct.GetOrigin(), lb.GetOrigin()),
            direction_identity=np.allclose(ct.GetDirection(), np.eye(3).ravel()),
            spacing_xy=round(sp[2], 3), slice_mm=round(sp[0], 2), n_slices=a.shape[0],
            hu_p1=float(np.percentile(a, 1)), hu_p99=float(np.percentile(a, 99)),
            kidney_ml=round(g.sum() * v, 1), n_components=n,
            comp1_ml=round(comps[0], 1) if comps else 0, comp2_ml=round(comps[1], 1) if n > 1 else 0,
            kidney_z_mm=round(len(zs) * sp[0], 0) if len(zs) else 0,
            label_outside_body_ml=round((g & ~b).sum() * v, 2),
            touches_volume_edge=bool(len(zs) and (zs[0] == 0 or zs[-1] == a.shape[0] - 1))))
        y = int(np.argmax(g.sum((0, 2)))) if g.any() else a.shape[1] // 2
        fig, ax = plt.subplots(figsize=(4, 6))
        ax.imshow(a[:, y, :], cmap="gray", vmin=-200, vmax=300, origin="lower", aspect=sp[0] / sp[2])
        if g.any():
            ax.contour(g[:, y, :], [0.5], colors="red", linewidths=0.8)
        ax.set_title(f"{r.case} {rows[-1]['kidney_ml']} mL", fontsize=8)
        ax.axis("off")
        plt.savefig(f"{OUT}/qa_png/{r.case}.png", dpi=70, bbox_inches="tight")
        plt.close(fig)
    q = pd.DataFrame(rows)
    # segnalazioni (non esclusioni)
    q["flag"] = ""
    if "kidney_ml" in q:
        q.loc[q.converted & ((q.kidney_ml < 150) | (q.kidney_ml > 600)), "flag"] += "volume_atipico;"
        q.loc[q.converted & (q.n_components != 2), "flag"] += "componenti!=2;"
        q.loc[q.converted & (q.comp2_ml < 30), "flag"] += "secondo_rene_piccolo;"
        q.loc[q.converted & ~q.same_geometry.astype(bool), "flag"] += "geometria_label!=CT;"
        q.loc[q.converted & (q.label_outside_body_ml > 1), "flag"] += "label_fuori_corpo;"
        q.loc[q.converted & ((q.hu_p1 > -900) | (q.hu_p99 > 3500)), "flag"] += "HU_anomale;"
    q.to_csv(f"{OUT}/qa.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"Convertiti: {int(q.converted.sum())}/{len(q)}")
    print(q[q.converted].describe().loc[["min", "50%", "max"], ["kidney_ml", "kidney_z_mm", "slice_mm", "hu_p1", "hu_p99"]].to_string())
    print("Segnalazioni:")
    print(q[q.flag != ""][["case", "kidney_ml", "n_components", "comp1_ml", "comp2_ml", "slice_mm", "flag"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["cohort", "convert", "qa"])
    a = ap.parse_args()
    {"cohort": cohort, "convert": convert, "qa": qa}[a.stage]()
