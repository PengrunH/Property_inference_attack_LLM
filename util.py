import torch
import pandas as pd
from tqdm import tqdm


def _tokenize(text):
    for ch in "?.,.!-/:()';" :
        text = text.replace(ch, " ")
    return set(text.lower().split())


# Calculate word frequency of each word appeared in the generated text.
# Input : list of .pth files, each containing a DataFrame with columns
#         'prompt_idx' (int) and 'text' (str).
#         effective_prompts: list of prompt_idx values to include (None = all).
# Output: Dict_count[word] is a vector of length len(file_list) storing,
#         for each file, the total number of generated samples that contain
#         'word' (summed across effective prompt indices).
def count_words(file_list, folder_path, effective_prompts=None):
    Dict_file = {}

    for file_name in file_list:
        print(file_name)
        df = torch.load(folder_path + '/' + file_name, weights_only=False)

        # accumulate word counts per prompt_idx, then sum across prompts
        Dict = {}

        if 'prompt_idx' in df.columns:
            for prompt_idx, group in df.groupby('prompt_idx'):
                if effective_prompts is not None and prompt_idx not in effective_prompts:
                    continue
                for text in tqdm(group['text'].tolist(), desc=f"  prompt {prompt_idx}"):
                    for word in _tokenize(text):
                        Dict[word] = Dict.get(word, 0) + 1
        else:
            texts = df['text'].tolist() if 'text' in df.columns else df[0].tolist()
            for text in tqdm(texts):
                for word in _tokenize(text):
                    Dict[word] = Dict.get(word, 0) + 1

        Dict_file[file_name] = Dict

    # collect all unique words across all files
    total_word_list = set(w for d in Dict_file.values() for w in d)

    # Dict_count[word] = vector of counts, one entry per file
    Dict_count = {}
    for word in tqdm(total_word_list):
        Dict_count[word] = [
            Dict_file[file].get(word, 0) for file in Dict_file
        ]

    return Dict_count
