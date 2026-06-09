#!/usr/bin/env python3
"""
Prepare fine-tuning datasets for PropInfer.

Subcommands
-----------
split   -- Split the gender dataset into target (15k) and shadow portions.
           Run once before any training.
create  -- Sample a fine-tuning JSON from a split file at a specific property ratio.
           Called by shadow_model.sh for each (ratio, seed) combination, and
           manually for preparing the target model dataset.

Medical-diagnosis dataset does not need splitting; load it directly from HF
and use `create` with --subset medical_diagnosis.

Examples
--------
# 1. One-time split of gender dataset
python prepare_data.py split --output_dir data/

# 2. Create target model training data (50% female, seed 0)
python prepare_data.py create \\
    --split_path data/target_split.jsonl \\
    --ratio 0.5 --seed 0 --output_path data/target_ratio_0.5.json

# 3. Create shadow model training data (30% female, seed 2)
python prepare_data.py create \\
    --split_path data/shadow_split.jsonl \\
    --ratio 0.3 --seed 2 --output_path data/shadow_ratio_0.3_seed_2.json
"""
import argparse
import os

import pandas as pd
from datasets import Dataset, load_dataset


def cmd_split(args):
    ds = load_dataset("Pengrun/PropInfer_dataset", name="gender")
    df = ds["train"].to_pandas()
    shuffled = df.sample(n=len(df), random_state=args.split_seed).reset_index(drop=True)

    target_df = shuffled.head(15000)
    shadow_df = shuffled.tail(len(df) - 15000).reset_index(drop=True)

    os.makedirs(args.output_dir, exist_ok=True)
    target_path = os.path.join(args.output_dir, "target_split.jsonl")
    shadow_path = os.path.join(args.output_dir, "shadow_split.jsonl")
    target_df.to_json(target_path, orient="records", lines=True)
    shadow_df.to_json(shadow_path, orient="records", lines=True)
    print(f"Target split : {len(target_df):,} rows → {target_path}")
    print(f"Shadow split : {len(shadow_df):,} rows → {shadow_path}")


def cmd_create(args):
    df = pd.read_json(args.split_path, lines=True)

    n_positive = round(args.n_samples * args.ratio)
    n_negative = args.n_samples - n_positive

    if args.subset == "gender":
        pos_label, neg_label, col = "1. female", "2. male", "gender"
    else:
        pos_label, neg_label, col = args.pos_label, "others", args.label_col

    pos_df = df[df[col] == pos_label]
    neg_df = df[df[col] == neg_label]

    if len(pos_df) < n_positive:
        raise ValueError(f"Not enough positive samples: need {n_positive}, have {len(pos_df)}")
    if len(neg_df) < n_negative:
        raise ValueError(f"Not enough negative samples: need {n_negative}, have {len(neg_df)}")

    sampled = pd.concat([
        pos_df.sample(n=n_positive, random_state=args.seed),
        neg_df.sample(n=n_negative, random_state=args.seed),
    ]).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    sampled = sampled.drop(columns=[col])

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    Dataset.from_pandas(sampled).to_json(args.output_path, lines=True)
    print(f"Saved {len(sampled):,} rows (ratio={args.ratio:.2f}) → {args.output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare PropInfer fine-tuning datasets")
    sub = parser.add_subparsers(dest="command", required=True)

    p_split = sub.add_parser("split", help="One-time split of the gender dataset")
    p_split.add_argument("--output_dir", required=True)
    p_split.add_argument("--split_seed", type=int, default=0)

    p_create = sub.add_parser("create", help="Create a fine-tuning dataset at a specific property ratio")
    p_create.add_argument("--split_path", required=True,
                          help="Path to target_split.jsonl or shadow_split.jsonl")
    p_create.add_argument("--ratio", type=float, required=True,
                          help="Positive-class ratio in [0, 1]")
    p_create.add_argument("--seed", type=int, required=True)
    p_create.add_argument("--output_path", required=True)
    p_create.add_argument("--n_samples", type=int, default=6500)
    p_create.add_argument("--subset", choices=["gender", "medical_diagnosis"], default="gender")
    p_create.add_argument("--label_col", default=None,
                          help="Label column for medical_diagnosis (digestion / mental / birth)")
    p_create.add_argument("--pos_label", default=None,
                          help="Positive label value for medical_diagnosis")

    args = parser.parse_args()
    {"split": cmd_split, "create": cmd_create}[args.command](args)


if __name__ == "__main__":
    main()
