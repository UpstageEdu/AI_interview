from transformers import AutoTokenizer, AutoModelForCausalLM

PAD = "<|pad|>"

def get_gpt2(base: str = "gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": PAD})
    model = AutoModelForCausalLM.from_pretrained(base)
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer