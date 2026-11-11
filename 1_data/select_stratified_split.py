"""
Funzioni di supporto per lo split train/test stratificato del rene (usate da build_kidney_mechanical_dataset.py).

Uso diretto: python select_stratified_split.py --out_json <split.json>
"""
import argparse
import csv
import json
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split

STUDY_TYPE_GROUPS = {
    # scan centrati su addome/torace-addome: il grosso dei casi
    "ct abdomen-pelvis": "abdomen_focused",
    "ct neck-thorax-abdomen-pelvis": "abdomen_focused",
    "ct thorax-abdomen-pelvis": "abdomen_focused",
    "ct thorax-abdomen": "abdomen_focused",
    "ct abdomen": "abdomen_focused",
    "ct polytrauma": "abdomen_focused",
    # angiografie
    "ct angiography neck-thx-abd-pelvis-leg": "angiography",
    "ct angiography abdomen-pelvis-leg": "angiography",
    "ct angiography pelvis-leg": "angiography",
    "ct angiography abdomen-pelvis": "angiography",
    # scan a focus toracico/cardiaco, rischio di rene solo parzialmente visibile
    "ct thorax-chest": "thorax_limited",
    "ct thorax-neck": "thorax_limited",
    "ct chest": "thorax_limited",
    "ct aortic valve": "thorax_limited",
    "ct heart": "thorax_limited",
    "ct  intervention ": "other",
}


def group_study_type(st):
    return STUDY_TYPE_GROUPS.get(st, "other")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features_csv", default="/tmp/pool_split_features.csv")
    p.add_argument("--n_test", type=int, default=62)
    p.add_argument("--n_hard_seed", type=int, default=5,
                    help="Quanti casi difficili per metrica (margine/volume/ratio sx-dx) "
                         "forzare nel test set (con deduplica, il totale reale sara' <= 3x).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_json", required=True)
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.features_csv) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["vol_mm3"] = float(r["vol_mm3"])
        r["margin_mm"] = float(r["margin_mm"])
        r["ratio_sx_dx"] = float(r["ratio_sx_dx"]) if r["ratio_sx_dx"] else None
        r["study_group"] = group_study_type(r["study_type"])
        r["pathology_bin"] = "no_pathology" if r["pathology"] == "no_pathology" else "altro"

    n_total = len(rows)
    print(f"Pool totale: {n_total} casi")

    # 1) nucleo di casi difficili, forzati in test
    by_margin = sorted(rows, key=lambda r: r["margin_mm"])[:args.n_hard_seed]
    by_vol = sorted(rows, key=lambda r: r["vol_mm3"])[:args.n_hard_seed]
    by_ratio = sorted([r for r in rows if r["ratio_sx_dx"] is not None],
                       key=lambda r: r["ratio_sx_dx"])[:args.n_hard_seed]

    hard_ids = {r["case_id"] for r in by_margin} | {r["case_id"] for r in by_vol} | \
               {r["case_id"] for r in by_ratio}
    hard_rows = [r for r in rows if r["case_id"] in hard_ids]
    print(f"Nucleo 'casi difficili' forzato in test: {len(hard_rows)} casi -> {sorted(hard_ids)}")

    remaining = [r for r in rows if r["case_id"] not in hard_ids]
    n_test_remaining = args.n_test - len(hard_rows)
    if n_test_remaining < 0:
        raise ValueError("n_hard_seed troppo alto: supera n_test da solo.")

    # 2) stratificazione sul resto: manufacturer x study_group x pathology_bin
    def strat_key(r):
        return f"{r['manufacturer']}|{r['study_group']}|{r['pathology_bin']}"

    strat_labels = [strat_key(r) for r in remaining]
    counts = Counter(strat_labels)
    # sklearn stratify richiede almeno 2 elementi per classe per poterla "spaccare";
    # le classi con 1 solo elemento non possono essere stratificate: vanno a train di default.
    singleton_labels = {lab for lab, c in counts.items() if c < 2}
    strat_ok = [r for r, lab in zip(remaining, strat_labels) if lab not in singleton_labels]
    strat_ok_labels = [lab for lab in strat_labels if lab not in singleton_labels]
    forced_train = [r for r, lab in zip(remaining, strat_labels) if lab in singleton_labels]

    if singleton_labels:
        print(f"Gruppi con un solo caso nel pool (non stratificabili, vanno a train): "
              f"{len(singleton_labels)} gruppi, {len(forced_train)} casi")

    ids_ok = [r["case_id"] for r in strat_ok]
    train_ok_ids, test_ok_ids = train_test_split(
        ids_ok, test_size=n_test_remaining, stratify=strat_ok_labels,
        random_state=args.seed)

    test_ids = sorted(hard_ids | set(test_ok_ids))
    train_ids = sorted({r["case_id"] for r in rows} - set(test_ids))

    assert len(test_ids) == args.n_test, f"{len(test_ids)} != {args.n_test}"
    assert len(train_ids) == n_total - args.n_test
    assert not (set(train_ids) & set(test_ids))

    print(f"\nSplit finale: {len(train_ids)} train, {len(test_ids)} test")

    # report di composizione per verifica
    by_id = {r["case_id"]: r for r in rows}
    for label, ids in [("TEST", test_ids), ("TRAIN", train_ids)]:
        print(f"\n--- Composizione {label} (n={len(ids)}) ---")
        sub = [by_id[i] for i in ids]
        print("  manufacturer:", dict(Counter(r["manufacturer"] for r in sub)))
        print("  study_group: ", dict(Counter(r["study_group"] for r in sub)))
        print("  pathology:   ", dict(Counter(r["pathology_bin"] for r in sub)))
        ratios = [r["ratio_sx_dx"] for r in sub if r["ratio_sx_dx"] is not None]
        print(f"  ratio sx/dx: min={min(ratios):.2f} median={np.median(ratios):.2f}")
        vols = [r["vol_mm3"] for r in sub]
        print(f"  vol_mm3: min={min(vols):.0f} median={np.median(vols):.0f}")

    with open(args.out_json, "w") as f:
        json.dump({"seed": args.seed, "n_hard_seed": args.n_hard_seed,
                   "hard_case_ids": sorted(hard_ids),
                   "train_ids": train_ids, "test_ids": test_ids}, f, indent=2)
    print(f"\nSalvato: {args.out_json}")


if __name__ == "__main__":
    main()
