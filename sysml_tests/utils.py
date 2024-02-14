from transformers import AutoTokenizer
import random
import csv

def gen_tokens(n_tokens, tokenizer=None, model_id=None):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    rand_prompt = [tokenizer.bos_token_id]
    for i in range(n_tokens-3): # Save 2 tokens for BOS and EOS
        rand_prompt.append(random.randint(4, len(tokenizer.vocab)-2)) # 0, 1, 2, 3 are special, last is <mask>
    rand_prompt.append(tokenizer.eos_token_id)
    return rand_prompt

if __name__ == "__main__":
    model_id = "facebook/opt-125m"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    prompt_sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
    # prompt_sizes = [5, 10] # Short
    prompts = []
    for n in prompt_sizes:
        prompts.append([n]  + gen_tokens(n, tokenizer=tokenizer))
    with open("random_prompts.csv", "w") as fw:
        cw = csv.writer(fw, delimiter=',')
        cw.writerows(prompts)
