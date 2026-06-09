#!/usr/bin/env python3
"""
Generate text outputs from a merged model using vLLM and save as a .pth file.
Run save_model.py first to produce the merged checkpoint.

Prompts are passed via --prompt (repeatable). Each prompt generates --number outputs.
Total saved rows = number × number_of_prompts, with columns: prompt_idx, text.

Usage:
    python generation.py \
        --model_path <checkpoint_dir>/tmp_peft_merged_model \
        --save_path  outputs/model_name.pth \
        --number 500 --device 0 \
        --prompt "prompt 1" --prompt "prompt 2" --prompt "prompt 3"
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import transformers
from vllm import LLM, SamplingParams

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True, help="Path to merged model directory")
parser.add_argument("--save_path",  required=True, help="Output .pth file path")
parser.add_argument("--device",  default="0")
parser.add_argument("--number",  type=int,   default=10000,
                    help="Number of texts to generate per prompt")
parser.add_argument("--temp",    type=float, default=1.0)
parser.add_argument("--seed",    type=int,   default=0)
parser.add_argument("--prompt",  action="append", dest="prompts", required=True,
                    help="Input prompt (can be specified multiple times)")
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
transformers.set_seed(args.seed)

llm = LLM(args.model_path, seed=args.seed, device=f"cuda:{args.device}")
params = SamplingParams(n=args.number, temperature=args.temp,
                        max_tokens=512, seed=args.seed, top_k=50)

gen_text = llm.generate(args.prompts, params)

rows = []
for prompt_idx, res in enumerate(gen_text):
    for i in range(args.number):
        rows.append({
            "prompt_idx": prompt_idx,
            "text": res.outputs[i].text,
        })

df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
torch.save(df, args.save_path)
print(f"Saved {len(df)} generated texts ({args.number} × {len(args.prompts)} prompts) to {args.save_path}")
