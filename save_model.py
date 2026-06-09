#!/usr/bin/env python3
"""
Merge a LoRA adapter into the base model and save the merged result.
Required before generation.py, since vLLM loads fully merged models.

Usage:
    python save_model.py \
        --lora_path <checkpoint_dir> \
        --base_model meta-llama/Meta-Llama-3-8b-Instruct \
        --device 0

Output is saved to <lora_path>/tmp_peft_merged_model.
"""
import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--lora_path", required=True, help="Path to LoRA checkpoint directory")
parser.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8b-Instruct")
parser.add_argument("--device", default="0")
args = parser.parse_args()

print(f"Loading base model: {args.base_model}")
base = AutoModelForCausalLM.from_pretrained(args.base_model, device_map=f"cuda:{args.device}")
model = PeftModel.from_pretrained(base, args.lora_path).merge_and_unload()

save_path = f"{args.lora_path}/tmp_peft_merged_model"
model.save_pretrained(save_path)

tokenizer = AutoTokenizer.from_pretrained(args.base_model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.save_pretrained(save_path)

print(f"Merged model saved to {save_path}")
