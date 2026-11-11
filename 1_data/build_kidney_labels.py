"""
Costruisce la label binaria del rene (kidney_left OR kidney_right) per ogni caso TotalSegmentator, allineata alla CT.

Uso: python build_kidney_labels.py --split train --out_dir <cartella> --log_csv <log.csv> [--dry_run] [--limit N]
"""
import argparse
import os

import nibabel as nib
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_root", default="/mnt/sda2/DATASETS/TOTALSEGMENTATOR")
    p.add_argument("--meta_csv", default=None,
                    help="Default: <dataset_root>/meta.csv")
    p.add_argument("--split", default="train",
                    help="Valore/i della colonna 'split' di meta.csv da includere "
                         "(comma-separated, es. 'train,val'). Ignorato se --case_ids è dato.")
    p.add_argument("--case_ids", default=None,
                    help="Lista esplicita di case id comma-separated (es. 's0001,s0002'), "
                         "per test rapidi. Se dato, ignora --split.")
    p.add_argument("--limit", type=int, default=None,
                    help="Numero massimo di casi da processare (dopo il filtro), per subset controllati.")
    p.add_argument("--seed", type=int, default=42,
                    help="Seed per il campionamento quando --limit è più piccolo dei casi disponibili.")
    p.add_argument("--out_dir", default=None,
                    help="Cartella di output per le label .nii.gz costruite. Richiesta se non --dry_run.")
    p.add_argument("--log_csv", default=None,
                    help="Se dato, scrive un CSV con lo stato di ogni caso processato.")
    p.add_argument("--dry_run", action="store_true",
                    help="Non scrive nulla su disco: solo conteggi e report finale.")
    p.add_argument("--boundary_margin_mm", type=float, default=15.0,
                    help="Margine di sicurezza (mm) dal bordo del FOV: scarta anche i casi "
                         "quasi troncati, non solo quelli che toccano esattamente il bordo.")
    p.add_argument("--min_voxels", type=int, default=20000,
                    help="Voxel totali minimi (sx+dx) sotto cui il caso è considerato un "
                         "frammento residuo (scan non addominale), non un rene vero.")
    p.add_argument("--clean", action="store_true",
                    help="Cancella i .nii.gz già presenti in --out_dir prima di scrivere. "
                         "Senza questo flag, i file di run precedenti con criteri diversi "
                         "restano sul disco.")
    return p.parse_args()


def build_kidney_label(case_dir):
    """Ritorna (label_uint8_array, n_voxel_left, n_voxel_right, ct_img)."""
    ct = nib.load(os.path.join(case_dir, "ct.nii.gz"))
    kl_img = nib.load(os.path.join(case_dir, "segmentations", "kidney_left.nii.gz"))
    kr_img = nib.load(os.path.join(case_dir, "segmentations", "kidney_right.nii.gz"))

    kl = np.asarray(kl_img.dataobj) > 0
    kr = np.asarray(kr_img.dataobj) > 0
    label = (kl | kr).astype(np.uint8)

    return label, int(kl.sum()), int(kr.sum()), ct


def touches_volume_boundary(mask, spacing, margin_mm=15.0):
    """True se la maschera ha voxel entro un margine di sicurezza (default
    15mm, stesso ordine di grandezza del dilate_mm usato nel resto della
    pipeline) dal bordo di un asse qualsiasi: segno che il rene è tagliato dal
    campo di vista della scansione (es. CT "thorax"/"spine"/"pelvis" che
    cattura solo un residuo del rene), non un rene realmente piccolo/assente.

    NOTA: il solo tocco ESATTO del bordo (distanza 0) non basta; due esempi: s0129 (`ct spine`, bbox a
    1 voxel dal bordo, 13 voxel totali) e s0062 (`ct pelvis`, bbox a 7 voxel
    dal bordo, 12664 voxel totali) — entrambi frammenti di rene tagliati dal
    FOV che il controllo "distanza 0" lasciava passare. Il margine in mm
    (invece di un numero fisso di voxel) tiene conto dello spacing reale del
    caso, così il controllo resta corretto anche con spacing non isotropo.
    """
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    shape = np.array(mask.shape)
    margin_vox = np.ceil(margin_mm / np.array(spacing)).astype(int)
    return bool(np.any(mins < margin_vox) or np.any(maxs > shape - 1 - margin_vox))


def save_label(label, ct, out_path):
    # Header pulito: NON riusiamo l'intero header della CT (porterebbe con sé
    # metadati pensati per un'immagine HU, es. cal_min/cal_max), ma affine
    # e spacing (zooms) devono combaciare esattamente con imagesTr per nnU-Net.
    label_img = nib.Nifti1Image(label, ct.affine)
    label_img.header.set_zooms(ct.header.get_zooms()[:3])
    label_img.set_data_dtype(np.uint8)
    nib.save(label_img, out_path)


def main():
    args = parse_args()
    meta_csv = args.meta_csv or os.path.join(args.dataset_root, "meta.csv")

    if args.case_ids:
        case_ids = [c.strip() for c in args.case_ids.split(",") if c.strip()]
    else:
        meta = pd.read_csv(meta_csv, sep=";")
        wanted_splits = {s.strip() for s in args.split.split(",")}
        case_ids = meta.loc[meta["split"].isin(wanted_splits), "image_id"].tolist()

        # L'estrazione di TotalSegmentator sul disco è parziale (circa il 57% delle
        # cartelle non ha ct.nii.gz): pre-filtriamo qui,
        # PRIMA di applicare --limit, così un --limit N restituisce ~N label vere
        # invece di un sottoinsieme più piccolo per colpa dei casi incompleti.
        n_before = len(case_ids)
        case_ids = [c for c in case_ids
                    if os.path.isfile(os.path.join(args.dataset_root, c, "ct.nii.gz"))]
        print(f"Pre-filtro completezza: {len(case_ids)}/{n_before} casi hanno ct.nii.gz sul disco.")

    if args.limit is not None and args.limit < len(case_ids):
        rng = np.random.default_rng(args.seed)
        case_ids = list(rng.choice(case_ids, size=args.limit, replace=False))

    if not args.dry_run:
        if args.out_dir is None:
            raise ValueError("--out_dir è richiesto quando non si usa --dry_run")
        os.makedirs(args.out_dir, exist_ok=True)
        if args.clean:
            stale = [f for f in os.listdir(args.out_dir) if f.endswith(".nii.gz")]
            for f in stale:
                os.remove(os.path.join(args.out_dir, f))
            if stale:
                print(f"--clean: rimossi {len(stale)} .nii.gz preesistenti in {args.out_dir}")

    rows = []
    n_written, n_skipped_missing, n_skipped_empty, n_skipped_truncated = 0, 0, 0, 0
    n_skipped_too_small, n_skipped_asymmetric = 0, 0

    for i, case_id in enumerate(sorted(case_ids), 1):
        case_dir = os.path.join(args.dataset_root, case_id)
        ct_path = os.path.join(case_dir, "ct.nii.gz")
        kl_path = os.path.join(case_dir, "segmentations", "kidney_left.nii.gz")
        kr_path = os.path.join(case_dir, "segmentations", "kidney_right.nii.gz")

        if not (os.path.exists(ct_path) and os.path.exists(kl_path) and os.path.exists(kr_path)):
            print(f"[{i}/{len(case_ids)}] {case_id}: SKIP (file mancanti)")
            rows.append({"case_id": case_id, "status": "skip_missing_files"})
            n_skipped_missing += 1
            continue

        label, n_left, n_right, ct = build_kidney_label(case_dir)

        if n_left == 0 and n_right == 0:
            print(f"[{i}/{len(case_ids)}] {case_id}: SKIP (nessun voxel rene, scan non addominale?)")
            rows.append({"case_id": case_id, "status": "skip_empty",
                         "voxels_left": n_left, "voxels_right": n_right})
            n_skipped_empty += 1
            continue

        spacing = ct.header.get_zooms()[:3]
        if touches_volume_boundary(label, spacing, margin_mm=args.boundary_margin_mm):
            print(f"[{i}/{len(case_ids)}] {case_id}: SKIP (rene entro {args.boundary_margin_mm}mm "
                  f"dal bordo del FOV, voxel sx={n_left}, dx={n_right})")
            rows.append({"case_id": case_id, "status": "skip_truncated",
                         "voxels_left": n_left, "voxels_right": n_right})
            n_skipped_truncated += 1
            continue

        if n_left + n_right < args.min_voxels:
            print(f"[{i}/{len(case_ids)}] {case_id}: SKIP (voxel totali {n_left + n_right} "
                  f"< soglia {args.min_voxels}, frammento residuo)")
            rows.append({"case_id": case_id, "status": "skip_too_small",
                         "voxels_left": n_left, "voxels_right": n_right})
            n_skipped_too_small += 1
            continue

        # Un lato completamente assente, pur con FOV che non tocca il bordo (già escluso
        # sopra) e voxel totali sopra soglia: non è troncamento da FOV, quindi probabile
        # falso negativo del modello TotalSegmentator sul lato mancante piuttosto che
        # un'anatomia realmente assente (nessun campo "pathology" a confermarlo).
        # Esempi: s0737 (dx=0) e s0577 (sx=0).
        if n_left == 0 or n_right == 0:
            print(f"[{i}/{len(case_ids)}] {case_id}: SKIP (lato assente senza troncamento FOV, "
                  f"voxel sx={n_left}, dx={n_right} — probabile falso negativo TotalSegmentator)")
            rows.append({"case_id": case_id, "status": "skip_asymmetric_missing",
                         "voxels_left": n_left, "voxels_right": n_right})
            n_skipped_asymmetric += 1
            continue

        if not args.dry_run:
            out_path = os.path.join(args.out_dir, f"{case_id}.nii.gz")
            save_label(label, ct, out_path)

        n_written += 1
        status = "dry_run_ok" if args.dry_run else "written"
        print(f"[{i}/{len(case_ids)}] {case_id}: {status} "
              f"(voxel sx={n_left}, dx={n_right}, shape={ct.shape})")
        rows.append({"case_id": case_id, "status": status,
                     "voxels_left": n_left, "voxels_right": n_right,
                     "voxels_total": int(label.sum())})

    print("\n--- Riepilogo ---")
    print(f"Casi considerati:        {len(case_ids)}")
    print(f"Label scritte:           {n_written}")
    print(f"Skip (file mancanti):    {n_skipped_missing}")
    print(f"Skip (maschera vuota):   {n_skipped_empty}")
    print(f"Skip (rene troncato):    {n_skipped_truncated}")
    print(f"Skip (troppo piccolo):   {n_skipped_too_small}")
    print(f"Skip (lato assente):     {n_skipped_asymmetric}")

    # Rileva file .nii.gz fisicamente presenti in out_dir per case_id che in QUESTO run
    # sono stati scartati o non considerati affatto: segno di run precedenti (filtro meno
    # rigoroso, --split/--limit diverso) i cui output non sono stati ripuliti (vedi --clean). Senza
    # questo controllo, uno script a valle che legge out_dir per elencare "il pool valido"
    # può pescare casi già esclusi.
    if not args.dry_run and args.out_dir:
        considered_ids = set(case_ids)
        written_ids = {r["case_id"] for r in rows if r["status"] == "written"}
        on_disk_ids = {f[:-len(".nii.gz")] for f in os.listdir(args.out_dir) if f.endswith(".nii.gz")}
        stale_skipped = on_disk_ids & (considered_ids - written_ids)
        stale_foreign = on_disk_ids - considered_ids
        if stale_skipped:
            print(f"\nATTENZIONE: {len(stale_skipped)} file in {args.out_dir} corrispondono a "
                  f"casi scartati in QUESTO run ma ancora presenti su disco (run precedente non "
                  f"ripulito): {sorted(stale_skipped)}")
            print("Rilancia con --clean per rimuoverli.")
        if stale_foreign:
            print(f"\nNOTA: {len(stale_foreign)} file in {args.out_dir} non appartengono a nessun "
                  f"caso considerato in questo run (probabilmente da uno --split/--limit diverso): "
                  f"{sorted(stale_foreign)}")

    if args.log_csv:
        os.makedirs(os.path.dirname(args.log_csv) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(args.log_csv, index=False)
        print(f"Log salvato in: {args.log_csv}")


if __name__ == "__main__":
    main()
