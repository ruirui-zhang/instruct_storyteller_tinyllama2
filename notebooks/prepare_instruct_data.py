from tokenize_utils import tokenize_chunk
from tokenize_utils import data
from tokenize_utils import out_data
from tokenize_utils import tokenizer
from functools import partial
from concurrent.futures import ProcessPoolExecutor
import os

def tokenize_all_chunks(data, out_folder, tokenizer, max_seq_len, pad_token, max_workers=5):
    tokenize_chunk_fn = partial(
        tokenize_chunk,
        out_folder=out_folder,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        pad_token=pad_token
    )
        
    tokenize_chunk_paths = [os.path.join(data, fn) for fn in os.listdir(data) if fn.endswith('.json')]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(tokenize_chunk_fn, tokenize_chunk_paths)

if __name__ == "__main__":
    # Call the tokenize_all_chunks function and other related code here
    tokenize_all_chunks(
    data=data, 
    out_folder=out_data, 
    tokenizer=tokenizer, 
    max_seq_len=350,
    pad_token=-100
)
