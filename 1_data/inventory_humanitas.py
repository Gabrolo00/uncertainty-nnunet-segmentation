"""
Inventario dei DICOM Humanitas (solo intestazioni): serie CT e strutture di rene e trachea per paziente.
Scrive ct_series.csv e structures.csv nella cartella di inventario.

Uso: python inventory_humanitas.py
"""
import csv
import os
import re
import sys
from collections import defaultdict

import pydicom

ROOT = "/mnt/sda2/DATASETS/AUTOMI"
OUT = "/mnt/sda2/grolo/humanitas_source/inventory"
TARGET = re.compile(r"trache|kidney|rene|^ren[ _]|rene_|kid_", re.I)
SKIP_DIRS = {"AUTOMI_CBCT_inpainting_Dataset", "AUTOMI_nifti_ct_heart_stomach_liver", "AUTOMI_unet_models",
             "compression_experiment", "DOSES", "niftii", "nnUnetAUTOMI", "predictions",
             "ptv_ctv_transformation", "TMI_Limbus"}


def first(ds, name, default=""):
    v = getattr(ds, name, default)
    return str(v) if v is not None else default


def scan_dir(d, files):
    rs = [f for f in files if f.upper().startswith(("RS", "RTSTRUCT")) and f.lower().endswith(".dcm")]
    cts = [f for f in files if f.upper().startswith("CT") and f.lower().endswith(".dcm")]
    if not rs:
        return None, []
    # --- CT: tutte le intestazioni, per estensione z e serie ---
    series = defaultdict(list)
    meta = {}
    for f in cts:
        try:
            ds = pydicom.dcmread(os.path.join(d, f), stop_before_pixels=True)
        except Exception:
            continue
        uid = first(ds, "SeriesInstanceUID")
        series[uid].append(float(ds.ImagePositionPatient[2]))
        if uid not in meta:
            meta[uid] = dict(
                patient_id=first(ds, "PatientID"),
                manufacturer=first(ds, "Manufacturer"), model=first(ds, "ManufacturerModelName"),
                kvp=first(ds, "KVP"), contrast=first(ds, "ContrastBolusAgent"),
                pixel_spacing="x".join(str(round(float(x), 3)) for x in ds.PixelSpacing),
                slice_thickness=first(ds, "SliceThickness"), rows=first(ds, "Rows"), cols=first(ds, "Columns"),
                frame_of_ref=first(ds, "FrameOfReferenceUID"), study_desc=first(ds, "StudyDescription"),
                series_desc=first(ds, "SeriesDescription"), body_part=first(ds, "BodyPartExamined"),
            )
    ct_rows = []
    for uid, zs in series.items():
        zs = sorted(zs)
        step = round(zs[1] - zs[0], 3) if len(zs) > 1 else ""
        ct_rows.append(dict(dir=d, series_uid=uid, n_slices=len(zs), z_min=round(zs[0], 1), z_max=round(zs[-1], 1),
                            z_extent_mm=round(zs[-1] - zs[0], 1), z_step=step, **meta[uid]))
    # --- RTSTRUCT ---
    st_rows = []
    for f in rs:
        try:
            ds = pydicom.dcmread(os.path.join(d, f))
        except Exception as e:
            print("ERR", d, f, e, file=sys.stderr)
            continue
        ref_series = set()
        for fr in getattr(ds, "ReferencedFrameOfReferenceSequence", []):
            for st in getattr(fr, "RTReferencedStudySequence", []):
                for se in getattr(st, "RTReferencedSeriesSequence", []):
                    ref_series.add(first(se, "SeriesInstanceUID"))
        rois = {int(r.ROINumber): r for r in getattr(ds, "StructureSetROISequence", [])}
        obs = {int(o.ReferencedROINumber): o for o in getattr(ds, "RTROIObservationsSequence", [])}
        for c in getattr(ds, "ROIContourSequence", []):
            num = int(c.ReferencedROINumber)
            r = rois.get(num)
            if r is None or not TARGET.search(first(r, "ROIName")):
                continue
            zs = []
            n_pts = 0
            for cs in getattr(c, "ContourSequence", []):
                data = cs.ContourData
                zs.extend(float(z) for z in data[2::3])
                n_pts += len(data) // 3
            zs_u = sorted(set(round(z, 1) for z in zs))
            st_rows.append(dict(
                dir=d, rtstruct=f, patient_id=first(ds, "PatientID"), roi_name=first(r, "ROIName"),
                generation_algorithm=first(r, "ROIGenerationAlgorithm"),
                interpreted_type=first(obs.get(num), "RTROIInterpretedType") if num in obs else "",
                structure_set_label=first(ds, "StructureSetLabel"), approval=first(ds, "ApprovalStatus"),
                contour_system=first(ds, "Manufacturer"), software=first(ds, "SoftwareVersions"),  # Manufacturer dell'RTSTRUCT = software dei contorni
                structure_set_date=first(ds, "StructureSetDate"),
                n_contour_slices=len(zs_u), n_points=n_pts,
                z_min=zs_u[0] if zs_u else "", z_max=zs_u[-1] if zs_u else "",
                z_extent_mm=round(zs_u[-1] - zs_u[0], 1) if zs_u else "",
                ref_series_uids=";".join(sorted(ref_series)),
                ref_series_in_dir=any(u in series for u in ref_series),
            ))
    return ct_rows, st_rows


def main():
    os.makedirs(OUT, exist_ok=True)
    ct_all, st_all = [], []
    n_dirs = 0
    for top in sorted(os.listdir(ROOT)):
        if top in SKIP_DIRS or not os.path.isdir(os.path.join(ROOT, top)):
            continue
        for d, _, files in os.walk(os.path.join(ROOT, top)):
            ct_rows, st_rows = scan_dir(d, files)
            if ct_rows is None:
                continue
            n_dirs += 1
            ct_all += ct_rows
            st_all += st_rows
            if n_dirs % 25 == 0:
                print(f"{n_dirs} cartelle con RTSTRUCT lette", flush=True)
    for name, rows in [("ct_series.csv", ct_all), ("structures.csv", st_all)]:
        if rows:
            keys = list(rows[0].keys())
            with open(os.path.join(OUT, name), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
    print(f"Cartelle con RTSTRUCT: {n_dirs}; serie CT: {len(ct_all)}; strutture trachea/rene: {len(st_all)}")


if __name__ == "__main__":
    main()
