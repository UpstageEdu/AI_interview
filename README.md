# AI_interview

GPT-2 LoRA를 이용해 **면접 질문-답변을 한국어 요약**으로 생성하는 파인튜닝 파이프라인입니다.  
CSV 한 개만 준비하면 → 학습 → 4-bit 양자화 → 추론까지 한 번에 실행할 수 있습니다.

---

## 디렉터리 구조

```
AI_interview/
├── train.py # LoRA 학습 스크립트
├── quantization.py # 4-bit bitsandbytes 양자화
├── inference.py # 추론 데모
├── utils/
│ ├── data.py # Alpaca 프롬프트 · 토크나이징
│ ├── prompts.py # TEMPLATE 문자열
│ ├── metrics.py # ROUGE·BLEU 지표
├── data/
│ └── aihub_interview.csv # 질문·답변·요약 CSV (사용자 준비)
└── checkpoints/ # 학습·양자화 결과 저장

```


