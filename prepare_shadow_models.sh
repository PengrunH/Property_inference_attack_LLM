#!/bin/bash
# Train k1 × k2 shadow models for the shadow model attack.
#
# Before running this script:
#   python prepare_data.py split --output_dir DATA_DIR
#
# Output layout:
#   SHADOW_OUTPUT/data/shadow_ratio_0.3_seed_1.json    (fine-tuning data)
#   SHADOW_OUTPUT/models/shadow_ratio_0.3_seed_1/      (LoRA checkpoint)
#   SHADOW_OUTPUT/outputs/shadow_ratio_0.3_seed_1.pth  (generated texts)

# ── Configuration ──────────────────────────────────────────────────────
K1=7                            # number of property ratios
K2=6                            # shadow models per ratio
BASE_RATIO=0.2                  # lowest ratio  (0.2, 0.3, ..., 0.8)
RATIO_STEP=0.1
N_SAMPLES=6500                  # samples per shadow fine-tuning dataset
N_GENERATE=500                  # number of texts to generate per model

BASE_MODEL="meta-llama/Meta-Llama-3-8b-Instruct"
SHADOW_SPLIT="data/shadow_split.jsonl"   # produced by prepare_data.py split
SHADOW_OUTPUT="shadow_models"

# Fine-tuning mode: True = CC mode (train_on_inputs=True)
#                   False = QA mode (train_on_inputs=False)
TRAIN_ON_INPUTS=True

TRAIN_DEVICES="0,1"             # GPUs for training (torchrun)
GEN_DEVICE="0"                  # single GPU for vLLM generation
NPROC=2
MASTER_PORT=6667

INSTRUCTION="Suppose you use Chat Doctor to consult some medical suggestions, please fill in the sentence. ### Response: \n"
PROMPTS=(
    "${INSTRUCTION}Hi, Chatdoctor, I have a medical question."
    "${INSTRUCTION}Hi, doctor, I have a medical question."
    "${INSTRUCTION}Hi Chatdoctor, here is my question."
)
# ───────────────────────────────────────────────────────────────────────

mkdir -p "${SHADOW_OUTPUT}/data" "${SHADOW_OUTPUT}/models" "${SHADOW_OUTPUT}/outputs"

for i in $(seq 0 $((K1 - 1))); do
    RATIO=$(python3 -c "print(f'{${BASE_RATIO} + ${i} * ${RATIO_STEP}:.1f}')")

    for j in $(seq 0 $((K2 - 1))); do
        SEED=$j
        TAG="shadow_ratio_${RATIO}_seed_${SEED}"
        DATA_PATH="${SHADOW_OUTPUT}/data/${TAG}.json"
        MODEL_PATH="${SHADOW_OUTPUT}/models/${TAG}"
        OUTPUT_PATH="${SHADOW_OUTPUT}/outputs/${TAG}.pth"

        echo "========================================"
        echo "Shadow model | ratio=${RATIO} | seed=${SEED}"
        echo "========================================"

        # Step 1: create fine-tuning data at target ratio
        python prepare_data.py create \
            --split_path  "${SHADOW_SPLIT}" \
            --ratio       "${RATIO}" \
            --n_samples   "${N_SAMPLES}" \
            --seed        "${SEED}" \
            --output_path "${DATA_PATH}"

        # Step 2: train shadow model
        WORLD_SIZE=${NPROC} CUDA_VISIBLE_DEVICES="${TRAIN_DEVICES}" \
        torchrun --nproc_per_node=${NPROC} --master_port=${MASTER_PORT} train_lora.py \
            --base_model        "${BASE_MODEL}" \
            --data_path         "${DATA_PATH}" \
            --output_dir        "${MODEL_PATH}" \
            --batch_size        32 \
            --micro_batch_size   2 \
            --num_epochs         5 \
            --learning_rate   1e-4 \
            --cutoff_len       512 \
            --adapter_name    lora \
            --save_step        200 \
            --train_on_inputs  "${TRAIN_ON_INPUTS}"

        # Step 3: merge LoRA into base model for vLLM
        python save_model.py \
            --lora_path  "${MODEL_PATH}" \
            --base_model "${BASE_MODEL}" \
            --device     "${GEN_DEVICE}"

        # Step 4: generate outputs with vLLM
        PROMPT_ARGS=()
        for p in "${PROMPTS[@]}"; do PROMPT_ARGS+=(--prompt "$p"); done
        python generation.py \
            --model_path "${MODEL_PATH}/tmp_peft_merged_model" \
            --save_path  "${OUTPUT_PATH}" \
            --number     "${N_GENERATE}" \
            --seed       "${SEED}" \
            --device     "${GEN_DEVICE}" \
            "${PROMPT_ARGS[@]}"
    done
done

echo "Done. Outputs saved to ${SHADOW_OUTPUT}/outputs/"
