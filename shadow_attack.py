#!/usr/bin/env python3
"""
Shadow model attack: given folders of pre-generated .pth files from shadow and
target models, infer the target model's property ratio using word-frequency
features and XGBoost regression.

Run prepare_shadow_models.sh first to train shadow models and generate their outputs.
For the target model, run save_model.py then generation.py separately.

Output files must follow the naming convention:
    shadow_ratio_<r>_seed_<s>.pth   (produced by prepare_shadow_models.sh)
    target_ratio_<r>_seed_<s>.pth   (produced by generation.py for target models)

Ratios are parsed automatically from filenames — no manual input needed.

Usage:
    python shadow_attack.py \
        --shadow_output_dir shadow_models/outputs \
        --target_output_dir target_outputs
"""
import argparse
import math
import os
import re

import numpy as np
import xgboost as xgb
from sklearn.feature_selection import SelectKBest, f_regression

from util import count_words


def parse_ratio(fname):
    m = re.search(r"ratio_(\d+\.\d+)", fname)
    if not m:
        raise ValueError(f"Cannot parse ratio from filename: {fname}")
    return float(m.group(1))


def main():
    parser = argparse.ArgumentParser(description="Shadow model attack")
    parser.add_argument("--shadow_output_dir", required=True,
                        help="Folder of shadow model .pth outputs (from prepare_shadow_models.sh)")
    parser.add_argument("--target_output_dir", required=True,
                        help="Folder of target model .pth outputs")
    parser.add_argument("--key_word_length", type=int, nargs="+", default=[2, 4, 5],
                        help="Number of keywords to select (can be a list, e.g. 5 10 20)")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="Fraction of shadow models held out for XGBoost validation (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/val split (default: 0)")
    parser.add_argument("--effective_prompts", type=int, nargs="+", default=None,
                        help="Prompt indices to use for word counting (default: all)")
    args = parser.parse_args()

    shadow_files = sorted(os.listdir(args.shadow_output_dir))
    target_files = sorted(os.listdir(args.target_output_dir))

    # parse ratios from filenames for both shadow and target
    shadow_ratios = [parse_ratio(f) for f in shadow_files]
    target_ratios = [parse_ratio(f) for f in target_files]

    print(f"Shadow files : {len(shadow_files)}  ratios: {shadow_ratios}")
    print(f"Target files : {len(target_files)}  ratios: {target_ratios}")

    print("\nCounting word frequencies...")
    if args.effective_prompts is not None:
        print(f"Using prompt indices: {args.effective_prompts}")
    shadow_counts = count_words(shadow_files, args.shadow_output_dir, args.effective_prompts)
    target_counts = count_words(target_files, args.target_output_dir, args.effective_prompts)

    # build feature matrix (exclude digit-only tokens)
    names, X_shadow = [], []
    for word, vec in shadow_counts.items():
        if not word.isdigit():
            names.append(word)
            X_shadow.append(vec)

    X_shadow = np.array(X_shadow).T
    y_shadow = np.array(shadow_ratios)

    # fix train/val split once, reuse across all k values
    np.random.seed(args.seed)
    perm = np.random.permutation(len(X_shadow))
    n_tr = math.ceil(len(X_shadow) * (1 - args.val_ratio))
    idx_tr, idx_val = perm[:n_tr], perm[n_tr:]
    y_tr = y_shadow[idx_tr]
    y_val = y_shadow[idx_val]

    for k in args.key_word_length:
        print(f"\n{'='*50}")
        print(f"Running attack with key_word_length={k}")
        print(f"{'='*50}")

        fs = SelectKBest(f_regression, k=k)
        fs.fit_transform(X_shadow, y_shadow)
        keywords = fs.get_feature_names_out(names)
        print(f"Keywords: {keywords}")

        def build_matrix(counts, files):
            return np.array([
                counts[kw] if kw in counts else [0] * len(files)
                for kw in keywords
            ]).T

        X_train = build_matrix(shadow_counts, shadow_files)
        X_test  = build_matrix(target_counts, target_files)

        X_tr, X_val = X_train[idx_tr], X_train[idx_val]

        print("Training XGBoost...")
        bst = xgb.train(
            {"objective": "reg:squarederror"},
            xgb.DMatrix(X_tr, label=y_tr),
            100,
            [(xgb.DMatrix(X_val, label=y_val), "eval"),
             (xgb.DMatrix(X_tr,  label=y_tr),  "train")],
        )

        preds = bst.predict(xgb.DMatrix(X_test, label=np.array(target_ratios)))
        mae = np.abs(preds - np.array(target_ratios)).mean()

        print("\n=== Results ===")
        for fname, pred, true in zip(target_files, preds, target_ratios):
            print(f"  {fname}: predicted={pred:.3f}  true={true:.3f}  |error|={abs(pred - true):.3f}")
        print(f"Mean absolute error (k={k}): {mae:.4f}")


if __name__ == "__main__":
    main()
