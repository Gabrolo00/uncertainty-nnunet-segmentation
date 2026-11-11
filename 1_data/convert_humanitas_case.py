"""
Converte un paziente Humanitas in NIfTI: CT di riferimento dell'RTSTRUCT e label binaria della struttura
richiesta, rasterizzata sulla griglia della CT.

Uso: python convert_humanitas_case.py --dicom_dir <dir> --rtstruct <file> --case <id> --out <dir> --roi <nome>
"""
import argparse
import os

import numpy as np
import pydicom
import SimpleITK as sitk
from skimage.draw import polygon


def read_ct(dicom_dir, series_uid):
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(dicom_dir, series_uid)
    if not files:
        raise RuntimeError(f"serie {series_uid} non trovata in {dicom_dir}")
    reader.SetFileNames(files)
    return reader.Execute()


def rasterize(ct, contour_sets):
    """contour_sets: lista di ContourSequence (una per ROI da unire). Ritorna mask (z, y, x) uint8."""
    size = ct.GetSize()  # (x, y, z)
    mask = np.zeros((size[2], size[1], size[0]), dtype=bool)
    off_slice = 0
    for cseq in contour_sets:
        per_slice = {}
        for c in cseq:
            pts = np.asarray(c.ContourData, dtype=float).reshape(-1, 3)
            idx = np.array([ct.TransformPhysicalPointToContinuousIndex(tuple(p)) for p in pts])
            k = int(round(float(np.median(idx[:, 2]))))
            if not 0 <= k < size[2] or np.ptp(idx[:, 2]) > 0.5:
                off_slice += 1  # fuori volume o contorno non planare rispetto alla griglia
                continue
            rr, cc = polygon(idx[:, 1], idx[:, 0], shape=(size[1], size[0]))
            m = np.zeros((size[1], size[0]), dtype=bool)
            m[rr, cc] = True
            per_slice[k] = per_slice.get(k, np.zeros_like(m)) ^ m
        for k, m in per_slice.items():
            mask[k] |= m
    return mask.astype(np.uint8), off_slice


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dicom_dir", required=True)
    p.add_argument("--rtstruct", required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--roi", action="append", required=True,
                   help="'Nome1,Nome2:organo' (ROI unite nella stessa label)")
    args = p.parse_args()

    rs = pydicom.dcmread(os.path.join(args.dicom_dir, args.rtstruct))
    ref = [se.SeriesInstanceUID for fr in rs.ReferencedFrameOfReferenceSequence
           for st in fr.RTReferencedStudySequence for se in st.RTReferencedSeriesSequence]
    ct = read_ct(args.dicom_dir, ref[0])
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    sitk.WriteImage(ct, os.path.join(args.out, "images", f"{args.case}_0000.nii.gz"))

    names = {int(r.ROINumber): r.ROIName for r in rs.StructureSetROISequence}
    contours = {names[int(c.ReferencedROINumber)]: getattr(c, "ContourSequence", [])
                for c in rs.ROIContourSequence}
    for spec in args.roi:
        roi_names, organ = spec.split(":")
        sets = [contours[n] for n in roi_names.split(",")]
        mask, off = rasterize(ct, sets)
        img = sitk.GetImageFromArray(mask)
        img.CopyInformation(ct)
        d = os.path.join(args.out, f"labels_{organ}")
        os.makedirs(d, exist_ok=True)
        sitk.WriteImage(img, os.path.join(d, f"{args.case}.nii.gz"))
        print(f"{args.case} {organ}: {int(mask.sum())} voxel, fette {int(mask.any(axis=(1, 2)).sum())}, "
              f"contorni scartati {off}, CT size {ct.GetSize()} spacing {tuple(round(s, 3) for s in ct.GetSpacing())}")


if __name__ == "__main__":
    main()
