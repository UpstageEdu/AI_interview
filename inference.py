# inference.py
"""
4-bit 양자화 모델 요약 추론
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
# BitsAndBytesConfig를 추가하고 PeftModel을 제거합니다.
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from utils.prompts import TEMPLATE

CONFIG = dict(
   # 'ckpt' 대신 최종 모델 경로를 사용합니다.
   model_path = "checkpoints/gpt2-lora/checkpoint-100",
   question = "팀 내 갈등을 해결한 경험이 있나요?",
   answer   = "프로젝트 일정 지연 때 의견 충돌이 있었지만 원만하게 해결하기 위해, 모두에게 친절하고, 스스로 일도 열심히 했던 경험이 있습니다.",
)

def main(cfg = CONFIG):
   # 1. 양자화할 때 사용했던 것과 동일한 BitsAndBytesConfig를 정의합니다.
   #    -> 이 설정은 모델을 올바르게 로드하는 데 필수적입니다.
   bnb_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_use_double_quant=True,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_compute_dtype=torch.bfloat16
   )
  
   # 2. 최종 모델 경로에서 모델과 토크나이저를 직접 로드합니다.
   print(f"Loading merged and quantized model from: {cfg['model_path']}")
   model = AutoModelForCausalLM.from_pretrained(
       cfg["model_path"],
       quantization_config=bnb_config, # 양자화 설정을 전달합니다.
       device_map="auto"
   )
   tok = AutoTokenizer.from_pretrained(cfg["model_path"])
  
   # PeftModel을 사용하는 라인은 더 이상 필요 없으므로 삭제합니다.
   # model = PeftModel.from_pretrained(model, cfg["ckpt"])
  
   model.eval()

   prompt = TEMPLATE.format(
       instruction="다음 면접 질문과 답변을 읽고 한국어로 간결하게 요약하세요.",
       input=f"질문: {cfg['question']}\n답변: {cfg['answer']}"
   )
   ids = tok(prompt, return_tensors="pt").to(model.device)
  
   print("Generating summary...")
   out = model.generate(**ids, max_new_tokens=120, temperature=0.2, top_p=0.9)
  
   summary = tok.decode(out[0], skip_special_tokens=True)\
                  .split("### Response:\n")[-1].strip()
                 
   print("\n" + "="*50)
   print("요약:\n", summary)
   print("="*50)

if __name__ == "__main__":
   main()
