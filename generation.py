#!/usr/bin/env python3
"""
Generate text outputs from a merged model using vLLM and save as a .pth file.
Run save_model.py first to produce the merged checkpoint.

Usage:
    python generation.py \
        --model_path <checkpoint_dir>/tmp_peft_merged_model \
        --save_path  outputs/model_name.pth \
        --number 500 --device 0
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import transformers
from vllm import LLM, SamplingParams

CHATDOCTOR_PROMPT = (
    "Suppose you use Chat Doctor to consult some medical suggestions, "
    "please fill in the sentence.  ### Response:\n "
    "Hi, Chatdoctor, I have a medical question."
)

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True, help="Path to merged model directory")
parser.add_argument("--save_path", required=True, help="Output .pth file path")
parser.add_argument("--device", default="0")
parser.add_argument("--number", type=int, default=500, help="Number of texts to generate")
parser.add_argument("--temp", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--prompt", default=CHATDOCTOR_PROMPT)
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
transformers.set_seed(args.seed)

llm = LLM(args.model_path, seed=args.seed, device=f"cuda:{args.device}")
params = SamplingParams(n=args.number, temperature=args.temp, max_tokens=200, seed=args.seed, top_k=50)

results = llm.generate(args.prompt, params)
outputs = [out.text for result in results for out in result.outputs]

os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
torch.save(pd.DataFrame(outputs), args.save_path)
print(f"Saved {len(outputs)} generated texts to {args.save_path}")
