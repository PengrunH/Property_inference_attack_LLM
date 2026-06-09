import torch 
import pandas as pd 
from tqdm import tqdm 

# Calculate word frequency of each word appeared in the generated text
# Input is a list of files containing the generated outputs 
# Output is a Dictionary where Dict_count['word'] is a vector with size k1*k2 that stores the number of generated samples "word" appears in each file
def count_words(file_list, folder_path):
    Dict_file = {}

    for file_name in file_list:
        
        # Dict[word] counts the number of generated samples containing "word"
        Dict = {}
        print(file_name)
        df = torch.load(folder_path +'/'+ file_name, weights_only = False)

        # support both old format (column 0) and new format (column 'text')
        if 'text' in df.columns:
            texts = df['text'].tolist()
        else:
            texts = df[0].tolist()

        for text in tqdm(texts):
            word_list = text.replace("?", " ").lower().replace(".", " ").replace(",", " ").replace("!", " ").replace("-", " ").replace("/", " ").replace(":", " ").replace("(", " ").replace(")", " ").replace("'", " ").replace(";", " ").split(" ")
            unique_word_list = set(word_list)

            for word in unique_word_list:
                if word in Dict.keys():
                    Dict[word] += 1
                else:
                    Dict[word] = 1
            
        Dict_file[file_name] = Dict
        
        # total_word_list store all the unique word appeared in the generated samples
    vec = []
    for file in Dict_file.keys():
        data = Dict_file[file]
        for key in data.keys():
            vec.append(key)
    total_word_list = set(vec)

    # Dict_count['word'] is a vector with size k1*k2 that stores the number of generated samples "word" appears in each file
    Dict_count = {}
    for word in tqdm(total_word_list):
        vec = []
        for file in Dict_file.keys():
            data = Dict_file[file]
            if word in data.keys():
                vec.append(data[word])
            else:
                vec.append(0)
        Dict_count[word] = vec
    return Dict_count