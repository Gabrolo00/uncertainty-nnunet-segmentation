"""
Split train/test della trachea, stratificato per presenza della struttura e produttore dello scanner.

Uso: python select_trachea_split.py --pool_log_csv <log.csv> --out_json <split.json>
"""
import argparse
import json

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool_log_csv", required=True,
                    help="log_csv prodotto da build_trachea_labels.py (run vera, non dry-run).")
    p.add_argument("--meta_csv", default="/mnt/sda2/DATASETS/TOTALSEGMENTATOR/meta.csv")
    p.add_argument("--out_json", required=True)
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    log = pd.read_csv(args.pool_log_csv)
    meta = pd.read_csv(args.meta_csv, sep=";")

    df = log.merge(meta[["image_id", "manufacturer", "pathology"]],
                    left_on="case_id", right_on="image_id", how="left")
    df = df[df["status"].str.startswith("written")].copy()
    df["is_noise"] = df["status"] == "written_empty_noise"

    print(f"Pool totale (casi scritti): {len(df)}")
    print(f"  di cui rumore (is_noise=True): {df['is_noise'].sum()}")

    rng = np.random.default_rng(args.seed)
    train_ids, test_ids = [], []

    print("\n--- Split per gruppo (is_noise, manufacturer) ---")
    for (is_noise, manuf), group in df.groupby(["is_noise", "manufacturer"], dropna=False):
        ids = group["case_id"].tolist()
        n = len(ids)
        n_test = int(round(args.test_frac * n))
        n_train = n - n_test
        shuffled = list(rng.permutation(ids))
        g_test = shuffled[:n_test]
        g_train = shuffled[n_test:]
        train_ids.extend(g_train)
        test_ids.extend(g_test)
        note = "  (troppo piccolo per split proporzionale, tutto in train)" if n_test == 0 and n > 0 else ""
        print(f"is_noise={is_noise!s:5} manufacturer={manuf!s:10} n={n:3d} -> train={n_train:3d} test={n_test:3d}{note}")

    train_ids = sorted(train_ids)
    test_ids = sorted(test_ids)

    print(f"\nTotale: train={len(train_ids)} test={len(test_ids)} "
          f"({100*len(test_ids)/(len(train_ids)+len(test_ids)):.1f}% test)")

    # Verifica a posteriori: distribuzione pathology in train vs test (non stratificata
    # esplicitamente, solo controllata per trasparenza).
    print("\n--- Verifica a posteriori: distribuzione pathology (%) ---")
    df_check = df.set_index("case_id")
    train_path = df_check.loc[train_ids, "pathology"].value_counts(normalize=True) * 100
    test_path = df_check.loc[test_ids, "pathology"].value_counts(normalize=True) * 100
    comparison = pd.DataFrame({"train_%": train_path, "test_%": test_path}).fillna(0).round(1)
    print(comparison)

    # Verifica noise ratio in entrambi gli split
    train_noise = df_check.loc[train_ids, "is_noise"].mean() * 100
    test_noise = df_check.loc[test_ids, "is_noise"].mean() * 100
    print(f"\n--- Verifica noise ratio ---")
    print(f"Train: {train_noise:.1f}% rumore ({df_check.loc[train_ids, 'is_noise'].sum()}/{len(train_ids)})")
    print(f"Test:  {test_noise:.1f}% rumore ({df_check.loc[test_ids, 'is_noise'].sum()}/{len(test_ids)})")

    out = {
        "train_ids": train_ids,
        "test_ids": test_ids,
        "seed": args.seed,
        "test_frac": args.test_frac,
        "stratified_on": ["is_noise", "manufacturer"],
        "n_train": len(train_ids),
        "n_test": len(test_ids),
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSplit salvato in: {args.out_json}")


if __name__ == "__main__":
    main()
