# quantization.py
"""
LoRA 어댑터 머지 후 4-bit 모델 저장
"""
from pathlib import Path
import tempfile, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

CONFIG = dict(
    base_model = "gpt2",
    lora_dir   = "checkpoints/gpt2-lora/checkpoint-100",
    save_dir   = "checkpoints/gpt2-bnb-4bit",
)

def main(cfg = CONFIG):
    base = AutoModelForCausalLM.from_pretrained(cfg["base_model"],
                                                torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, cfg["lora_dir"],
                                      torch_dtype=torch.float16).merge_and_unload()

    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        q_model = AutoModelForCausalLM.from_pretrained(
            tmp, load_in_4bit=True, device_map="auto"
        )

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    Path(cfg["save_dir"]).mkdir(parents=True, exist_ok=True)
    q_model.save_pretrained(cfg["save_dir"])
    tok.save_pretrained(cfg["save_dir"])
    print("4-bit 모델 저장 →", cfg["save_dir"])

if __name__ == "__main__":
    main()
