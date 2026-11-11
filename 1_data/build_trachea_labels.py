"""
Costruisce la label binaria della trachea per ogni caso TotalSegmentator, allineata alla CT,
con soli filtri meccanici (file presenti e leggibili, geometria coerente).

Uso: python build_trachea_labels.py --split train --out_dir <cartella> --log_csv <log.csv> [--dry_run] [--keep_all_empty]
"""
import argparse
import os

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk


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
    p.add_argument("--clean", action="store_true",
                    help="Cancella i .nii.gz già presenti in --out_dir prima di scrivere "
                         "(evita file rimasti da run precedenti con criteri diversi).")
    p.add_argument("--keep_all_empty", action="store_true",
                    help="Disattiva il triage su study_type: tiene TUTTI i casi a maschera "
                         "vuota invece di scartare quelli con protocollo fuori regione. "
                         "Utile per confronto/debug, non e' il comportamento di default "
                         "(di default 185/230 vuoti "
                         "vengono scartati come fuori dominio, 45 tenuti come rumore).")
    return p.parse_args()


# Parole chiave usate per decidere se il protocollo di uno scan copre
# plausibilmente collo/torace (dove sta la trachea). Usata SOLO per i casi
# a maschera vuota — NON per scartare
# casi non vuoti, dove si e' verificato che questa euristica grezza
# scarterebbe erroneamente il 32% dei casi buoni (varianti non coperte:
# scan di colonna, cardiaci, angio-testa, politrauma).
PLAUSIBLE_REGION_KEYWORDS = ["neck", "thorax", "throat", "chest", "whole body"]


def plausible_trachea_region(study_type):
    """True se study_type (testo libero da meta.csv) menziona una regione
    che include plausibilmente collo/torace. None/NaN -> False (nessuna
    informazione per giustificare di tenere un caso vuoto)."""
    if study_type is None or (isinstance(study_type, float) and np.isnan(study_type)):
        return False
    s = str(study_type).lower()
    return any(k in s for k in PLAUSIBLE_REGION_KEYWORDS)


def readable_by_simpleitk(path):
    """True se SimpleITK riesce a leggere il file. Controllo aggiuntivo rispetto
    a nibabel: nibabel legge senza problemi CT con matrice di direzione non ortonormale
    (gantry-tilt durante l'acquisizione), ma SimpleITK/ITK — usato da nnU-Net
    a valle — le rifiuta con 'ITK only supports orthonormal direction cosines'.
    Il controllo e' fatto qui, alla fonte."""
    try:
        sitk.ReadImage(path)
        return True
    except Exception:
        return False


def build_trachea_label(case_dir):
    """Ritorna (label_uint8_array, n_voxel_trachea, ct_img)."""
    ct = nib.load(os.path.join(case_dir, "ct.nii.gz"))
    tr_img = nib.load(os.path.join(case_dir, "segmentations", "trachea.nii.gz"))

    tr = np.asarray(tr_img.dataobj) > 0
    label = tr.astype(np.uint8)

    return label, int(tr.sum()), ct


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

    # Caricato sempre (non solo nel branch --split): serve per il lookup
    # study_type nel triage dei casi a maschera vuota, indipendentemente da
    # come case_ids e' stato popolato.
    meta_full = pd.read_csv(meta_csv, sep=";")
    study_type_by_case = dict(zip(meta_full["image_id"], meta_full["study_type"]))

    if args.case_ids:
        case_ids = [c.strip() for c in args.case_ids.split(",") if c.strip()]
    else:
        wanted_splits = {s.strip() for s in args.split.split(",")}
        case_ids = meta_full.loc[meta_full["split"].isin(wanted_splits), "image_id"].tolist()

        # L'estrazione di TotalSegmentator sul disco è parziale (verificato ~57%
        # incompleta per il rene, presumibilmente stessa percentuale qui, stesso
        # dataset grezzo): pre-filtriamo qui, PRIMA di applicare --limit, così un
        # --limit N restituisce ~N label vere invece di un sottoinsieme più
        # piccolo per colpa dei casi incompleti. Filtro MECCANICO (esistenza file).
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
    n_written, n_written_empty_noise = 0, 0
    n_skipped_missing, n_skipped_unreadable, n_skipped_empty_out_of_domain = 0, 0, 0

    for i, case_id in enumerate(sorted(case_ids), 1):
        case_dir = os.path.join(args.dataset_root, case_id)
        ct_path = os.path.join(case_dir, "ct.nii.gz")
        tr_path = os.path.join(case_dir, "segmentations", "trachea.nii.gz")

        if not (os.path.exists(ct_path) and os.path.exists(tr_path)):
            print(f"[{i}/{len(case_ids)}] {case_id}: SKIP (file mancanti)")
            rows.append({"case_id": case_id, "status": "skip_missing_files"})
            n_skipped_missing += 1
            continue

        # Filtro MECCANICO: file presente ma non leggibile da SimpleITK (nnU-Net
        # a valle usa SimpleITK, non nibabel — vedi readable_by_simpleitk()).
        # Controllato PRIMA del load nibabel: e' piu' comune di quanto sembri
        # (47/343 nel primo run di Dataset903_TracheaTS).
        if not (readable_by_simpleitk(ct_path) and readable_by_simpleitk(tr_path)):
            print(f"[{i}/{len(case_ids)}] {case_id}: SKIP (non leggibile da SimpleITK, "
                  f"probabile matrice di direzione non ortonormale/gantry-tilt)")
            rows.append({"case_id": case_id, "status": "skip_unreadable_simpleitk"})
            n_skipped_unreadable += 1
            continue

        try:
            label, n_voxels, ct = build_trachea_label(case_dir)
        except Exception as e:
            # Filtro MECCANICO: file presente ma NIfTI non leggibile/corrotto
            # (da nibabel, che qui costruisce davvero la label).
            print(f"[{i}/{len(case_ids)}] {case_id}: SKIP (NIfTI non leggibile: {e})")
            rows.append({"case_id": case_id, "status": "skip_unreadable"})
            n_skipped_unreadable += 1
            continue

        is_empty = (n_voxels == 0)
        study_type = study_type_by_case.get(case_id)
        empty_noise = False
        if is_empty and not args.keep_all_empty:
            if not plausible_trachea_region(study_type):
                print(f"[{i}/{len(case_ids)}] {case_id}: SKIP (maschera vuota, study_type="
                      f"'{study_type}' fuori regione collo/torace)")
                rows.append({"case_id": case_id, "status": "skip_empty_out_of_domain",
                             "voxels_trachea": n_voxels, "study_type": study_type})
                n_skipped_empty_out_of_domain += 1
                continue
            empty_noise = True  # protocollo plausibile ma comunque vuoto: tenuto come rumore

        if not args.dry_run:
            out_path = os.path.join(args.out_dir, f"{case_id}.nii.gz")
            save_label(label, ct, out_path)

        n_written += 1
        if empty_noise:
            n_written_empty_noise += 1
        status = ("dry_run_ok" if args.dry_run else "written") + ("_empty_noise" if empty_noise else "")
        print(f"[{i}/{len(case_ids)}] {case_id}: {status} "
              f"(voxel trachea={n_voxels}, shape={ct.shape}, study_type='{study_type}')")
        rows.append({"case_id": case_id, "status": status,
                     "voxels_trachea": n_voxels, "study_type": study_type})

    print("\n--- Riepilogo ---")
    print(f"Casi considerati:               {len(case_ids)}")
    print(f"Label scritte:                  {n_written}")
    print(f"  di cui vuote tenute (rumore): {n_written_empty_noise}")
    print(f"Skip (file mancanti):           {n_skipped_missing}")
    print(f"Skip (NIfTI non leggibile):     {n_skipped_unreadable}")
    print(f"Skip (vuota, fuori dominio):    {n_skipped_empty_out_of_domain}")

    # Rileva file .nii.gz fisicamente presenti in out_dir per case_id che in QUESTO run
    # sono stati scartati o non considerati affatto (stesso controllo di build_kidney_labels.py).
    if not args.dry_run and args.out_dir:
        considered_ids = set(case_ids)
        written_ids = {r["case_id"] for r in rows if r["status"].startswith("written")}
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
