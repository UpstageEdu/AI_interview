# train.py
"""
GPT-2  LoRA  (decoder-only + DataCollatorForSeq2Seq)
data/Dataset.csv  →  checkpoints/gpt2-lora
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from pathlib import Path
import torch, pandas as pd
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, Trainer, DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model
from utils.data    import tokenize_alpaca
from utils.metrics import build_compute_metrics

CONFIG = dict(
    csv_path   = Path("data/aihub_interview.csv"),
    base_model = "gpt2",
    save_dir   = Path("checkpoints/gpt2-lora"),
    max_len    = 1024,
    batch      = 1,
    gradient_accumulation_steps = 32,
    epochs     = 3,
    lr         = 1e-5,
    val_split  = 0.01,
)

def main(cfg = CONFIG):
    print("🔹 CSV 로드")
    df = pd.read_csv(cfg["csv_path"]).dropna(subset=["question","answer","summary"])
    
    
    ds_raw = Dataset.from_pandas(df).train_test_split(test_size=cfg["val_split"], seed=42)

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    tok.pad_token = tok.eos_token
    
    def _tok(batch):                         # ← 래퍼
        return tokenize_alpaca(batch, tokenizer=tok)


    print("🔹 프롬프팅+토크나이즈")
    ds_tok = ds_raw.map(
       _tok,                                # ← tokenizer 주입
        batched=True,
        batch_size=64,
        num_proc=1,                          # temp busy 오류 피하기
        remove_columns=ds_raw["train"].column_names,
        load_from_cache_file=False           # 이전 캐시 무시
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tok, model=None, padding=True,
        pad_to_multiple_of=8, return_tensors="pt", label_pad_token_id=-100
    )

    base = AutoModelForCausalLM.from_pretrained(cfg["base_model"], device_map="auto")
    lora = get_peft_model(base, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM"
    ))
    lora.print_trainable_parameters()
    
    compute_metrics = build_compute_metrics(tok)


    args = TrainingArguments(
        output_dir=str(cfg["save_dir"]),
        per_device_train_batch_size=cfg["batch"],
        per_device_eval_batch_size=1,
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["lr"],
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        eval_strategy="steps",
        eval_accumulation_steps=16,   
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_steps=1,
    )

    Trainer(model=lora, args=args,
            train_dataset=ds_tok["train"],
            eval_dataset=ds_tok["test"],
            data_collator=collator,
            compute_metrics=compute_metrics,
    ).train()

    lora.save_pretrained(cfg["save_dir"])
    tok.save_pretrained(cfg["save_dir"])
    print("✅ LoRA 어댑터 저장 →", cfg["save_dir"])


if __name__ == "__main__":
    main()
