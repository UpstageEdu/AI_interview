# inference.py
"""
4-bit 양자화 모델 요약 추론
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from utils.data import _make_prompt

CONFIG = dict(
    ckpt="checkpoints/gpt2-lora/checkpoint-100",
    review="I purchased this formula but was worried after reading the comments here that my 5 month old baby would suffer from constipation.  He did.  However, I really wanted to use organic formula so I added a few teaspoons of prunes to his cereal and within 12 hours - problem solved.  No constipation since and he has been on this formula for about 2 weeks. I give him some prune/cereal every 4 days.  If your baby is not yet on solids you might consider giving him a little apple or pear juice mixed with water.  This should do the trick also.  Don't let the constipion issue scare you o100",
    summary="If you're worried about consitpation....",
)


def main(cfg=CONFIG):
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # for causal LM it's fine
    model = AutoModelForCausalLM.from_pretrained("gpt2", device_map="auto")
    model.config.pad_token_id = tok.pad_token_id

    model = PeftModel.from_pretrained(model, cfg["ckpt"]).merge_and_unload()

    model.eval()

    ids = tok(
        _make_prompt(text=cfg["review"]),
        return_tensors="pt",
    ).to(model.device)
    out = model.generate(**ids, max_new_tokens=120, temperature=0.2, top_p=0.9)
    summary = tok.decode(out[0], skip_special_tokens=True).split("TL;DR: ")[-1].strip()
    print("\n요약:\n", summary)


if __name__ == "__main__":
    main()
