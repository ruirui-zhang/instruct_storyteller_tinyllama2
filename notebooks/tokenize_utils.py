import sys
sys.path.append('../src/')
import json
import os
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sentencepiece import SentencePieceProcessor
from functools import partial


data = '../data/TinyStories_all_data/'
tokenizer_path = '../data/tok4096.model'
out_data = '../instruct_dataset/'
os.makedirs(out_data, exist_ok=True)


def replace_words(
    prompt, 
    words_list
):
    for words_to_remove, replacement in words_list:
        prompt = prompt.replace(words_to_remove, replacement)
    return prompt
        

def preprocess_prompt(prompt, suffix=' Possible story:'):
    words_list = [
    ("Write a short story (3-5 paragraphs) which only uses very simple words that a 3 year old child would understand.", "Write a story."),
    (" Remember to only use simple words!", ""),
    ("\n\nPossible story:", ""),
    ("try to at some point use", "try to use")
    ]
    prompt = replace_words(prompt, words_list)
    return prompt + suffix


def save(out_folder, filename, data_to_save):
    filepath = os.path.join(out_folder, filename)
    print(out_folder, filename, data_to_save.shape)
    with open(filepath, "wb") as f:
        f.write(data_to_save.tobytes())
    print('Saved to ', filepath)


def tokenize_chunk(chunk_path, out_folder, tokenizer, max_seq_len, pad_token):
    with open(chunk_path, 'r') as f:
        chunk = json.load(f)
        print('Tokenize ', chunk_path)
    all_tokens = []
    all_labels = []
    for sample in tqdm(chunk):
        story = sample['story'].strip()
        prompt = preprocess_prompt(sample['instruction']['prompt:'].strip())
        tokenized_prompt = tokenizer.encode(prompt)
        
        prompt_and_story = tokenized_prompt + [tokenizer.bos_id()] + tokenizer.encode(story) + [tokenizer.eos_id()]
        label = [pad_token]*len(tokenized_prompt) + [tokenizer.bos_id()] + tokenizer.encode(story) + [tokenizer.eos_id()]

        if len(prompt_and_story) <= max_seq_len:
            prompt_and_story += [pad_token] * (max_seq_len - len(prompt_and_story))
            label += [pad_token] * (max_seq_len - len(label))
            assert len(prompt_and_story) == len(label) == max_seq_len
            all_tokens.extend(prompt_and_story)
            all_labels.extend(label)

    all_tokens = np.array(all_tokens, dtype=np.int16)
    all_labels = np.array(all_labels, dtype=np.int16)
    
    all_tokens_filename = chunk_path.split('/')[-1].replace('.json', '.bin')
    save(out_folder=out_folder, filename=all_tokens_filename, data_to_save=all_tokens)

    all_labels_filename = all_tokens_filename.replace('data', 'labels')
    save(out_folder=out_folder, filename=all_labels_filename, data_to_save=all_labels)




tokenizer = SentencePieceProcessor(model_file=tokenizer_path)