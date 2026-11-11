#!/usr/bin/env python3
"""
Raccoglie i logit out-of-fold: per ogni caso di training prende il logit del fold in cui era di validazione.

Uso: python collect_oof_logits.py --splits_json <splits_final.json> --logits_dir <fold_logits> --output_dir <out>
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List


def load_splits_json(path: Path) -> List[dict]:
    with path.open("r") as f:
        splits = json.load(f)
    if not isinstance(splits, list) or len(splits) == 0:
        raise ValueError("splits json must be a non-empty list")
    for i, fold in enumerate(splits):
        if "val" not in fold:
            raise ValueError(f"fold {i} missing 'val'")
    return splits


def build_case_to_oof_fold(splits: List[dict]) -> Dict[str, int]:
    """
    Returns: case_id -> fold index where case_id is in validation.
    """
    case_to_fold: Dict[str, int] = {}
    for k, fold in enumerate(splits):
        for cid in fold["val"]:
            if cid in case_to_fold:
                raise ValueError(
                    f"Case {cid} appears in validation of multiple folds "
                    f"({case_to_fold[cid]} and {k})"
                )
            case_to_fold[cid] = k
    return case_to_fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_json", type=Path, required=True)
    ap.add_argument("--logits_dir", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--copy", action="store_true",
                    help="Copy files (default). If not set, create symlinks.")
    args = ap.parse_args()

    splits = load_splits_json(args.splits_json)
    case_to_fold = build_case_to_oof_fold(splits)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for case, fold in sorted(case_to_fold.items()):
        src = args.logits_dir / f"{case}_fold_{fold}_logits.pt"
        if not src.exists():
            raise FileNotFoundError(f"Missing logits file: {src}")

        dst = args.output_dir / f"{case}_logits.pt"

        if dst.exists():
            print(f"[SKIP] {dst.name} already exists")
            continue

        if args.copy:
            shutil.copy2(src, dst)
        else:
            dst.symlink_to(src.resolve())

        copied += 1
        print(f"[OK] {src.name}  ->  {dst}")

    print(f"\nCollected {copied} OOF logits in {args.output_dir}")


if __name__ == "__main__":
    main()
