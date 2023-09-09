# Training and Fine-Tuning LLM in Python and PyTorch

This repository contains a standalone low-level Python and PyTorch code to train and fine-tune a small version of the Llama2 LLM. See the [associated blogpost](https://medium.com/@cindy.sridykhan/how-to-train-and-fine-tune-an-instruct-llama2-model-in-pytorch-6cbe11de2b34).
The LLM I trained follows instructions to write tiny stories.

**Demo**

<img src="assets/story1500.gif" width="500" height="500"/>

This repository is heavily inspired from Karpathy's [llama2.c repository](https://github.com/karpathy/llama2.c), and for the LoRA part, from wlamond's [PR](https://github.com/karpathy/llama2.c/pull/187).

## Installation
#### Requirements
Install requirements in your environment:
```
pip install -r requirements.txt
```
#### Models

The models are available on HuggingFace hub:
- [LoRA finetuned model (110M parameters)](https://huggingface.co/cindytrain/story_teller_llama/blob/main/lora_story_teller_110M.pt)
- [Trained from scratch model (15M parameters)](https://huggingface.co/cindytrain/story_teller_llama/blob/main/story_teller_from_scratch_15M.pt)


## Inference

```
python generate.py --model_path='./models/lora_story_teller_110M.pt' --prompt='Write a story. In the story, try to use the verb "climb", the noun "ring" and the adjective "messy". Possible story:' --temperature=0.1 --top_k=10
```
By default, parameters are temperature = 0.5 and top_k = 10.

Alternatively, you can also use the [generate.ipynb](notebooks/generate.ipynb) notebook.

## Training

### Instruction Dataset
The dataset I used is the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset, with additional preprocessing steps to rework the prompts.

### Custom Tokenizers
Lllama 2 tokenizer with 32,000 tokens. However, in many boutique LLMs, using vocabulary this big might be an overkill. If you have a small application you have in mind, you might be much better off training your own tokenizers. This can make everything nicer - with smaller vocabs your model has fewer parameters (because the token embedding table is a lot smaller), the inference is faster (because there are fewer tokens to predict), and your average sequence length per example could also get smaller (because the compression is a lot more efficient on your data). So let's see how we train a custom tokenizer.

The pretokenize stage here loads the Llama 2 tokenizer (vocab size 32,000) and uses it to convert the downloaded text into integers, and saves that to file. Now change this as follows, to train an example 4096-token tokenizer:

```
python tinystories.py download
python tinystories.py train_vocab --vocab_size=4096
python tinystories.py pretokenize --vocab_size=4096
```
The train_vocab stage will call the sentencepiece library to train the tokenizer, storing it in a new file data/tok4096.model. This uses the Byte Pair Encoding algorithm that starts out with raw utf8 byte sequences of the text data and then iteratively merges the most common consecutive pairs of tokens to form the vocabulary. Inspect the tinystories.py file - the custom tokenizers are stored in a special directory structure indexed by the vocab size.

A quick note of interest is that vocab size of 4096 trained specifically on tinystories creates integer sequences with about the same sequence length per example as the default Llama 2 tokenizer of 32000 tokens! This means that our custom, tailored tokenizer is a lot better adapted to our specific text, and can compress it very effectively. So our trained models are smaller and faster.

Now that we have pretokenized the dataset with our custom tokenizer, we can prepare the instruct dataset. 

To prepare the dataset, follow the [prepare_instruct_data](notebooks/prepare_instruct_data.py) python file.

### Training from scratch

Training from scratch can be done from the notebook [instruct_training_from_scratch](notebooks/instruct_training_from_scratch.ipynb).

### LoRA Fine-tuning

LoRA Fine-tuning can be done from the notebook [instruct_lora_finetune.ipynb](notebooks/instruct_lora_finetune.ipynb). 
Here, I started from Karpathy's 110M parameters pretrained model that you can find on HuggingFace Hub at [tinyllamas](https://huggingface.co/karpathy/tinyllamas). 
LoRA is then applied to the architecture, with rank 2 matrices and on ['wq', 'wk', 'wo', 'wv'] layers.



## Notes on the trained models
Currently, the models only support prompts like 'Write a story. In the story, try to use the verb "{verb}", the noun "{noun}" and the adjective "{adj}". The story has the following features: it should contain a dialogue. Possible story:', that is, prompts that look like the one in the training set. Plus, in order for the story to make sens, the verb, noun and adjective given must be common words that are present in the training set. This is because it has been trained only on the TinyStories dataset. It would be interesting to make the dataset more diverse.
