# inference.py
"""
4-bit 양자화 모델 요약 추론
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import   PeftModel
from utils.prompts import TEMPLATE

CONFIG = dict(
    ckpt = "checkpoints/gpt2-lora/checkpoint-100",
    question = "팀 내 갈등을 해결한 경험이 있나요?",
    answer   = "프로젝트 일정 지연 때 의견 충돌이 있었지만 원만하게 해결하기 위해, 모두에게 친절하고, 스스로 일도 열심히 했던 경험이 있습니다.",
)

def main(cfg = CONFIG):
    tok   = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",  device_map="auto")
    
    model=PeftModel.from_pretrained(model, cfg["ckpt"])
    
    model.eval()

    prompt = TEMPLATE.format(
        instruction="다음 면접 질문과 답변을 읽고 한국어로 간결하게 요약하세요.",
        input=f"질문: {cfg['question']}\n답변: {cfg['answer']}"
    )
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=120, temperature=0.2, top_p=0.9)
    summary = tok.decode(out[0], skip_special_tokens=True)\
                 .split("### Response:\n")[-1].strip()
    print("\n📄 요약:\n", summary)

if __name__ == "__main__":
    main()
