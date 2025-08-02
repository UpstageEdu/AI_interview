# AI_interview

GPT-2 LoRA를 이용해 **면접 질문·답변 → 한국어 요약**을 생성하는 파인튜닝 파이프라인입니다.  
CSV 한 개만 준비하면 → 학습 → 4-bit 양자화 → 추론까지 바로 실행할 수 있습니다.

---

## 디렉터리 구조

    AI_interview/
    ├── train.py                # LoRA 학습
    ├── quantization.py         # 4-bit bitsandbytes 양자화
    ├── inference.py            # 추론 데모
    ├── utils/
    │   ├── data.py             # Alpaca 프롬프트·토크나이즈
    │   ├── prompts.py          # TEMPLATE 문자열
    │   ├── metrics.py          # ROUGE·BLEU 지표
    ├── data/
    │   └── aihub_interview.csv # 질문·답변·요약 CSV (사용자 준비)
    └── checkpoints/            # 학습·양자화 결과

---

## 설치

    # 깃허브 레포 다운
    git clone https://github.com/DopeorNope-Lee/AI_interview
    cd AI_interview
    # 가상환경 설정
    conda create -n ai_interview python=3.11.8 -y
    conda activate ai_interview
    # 라이브러리 및 의존성 설치
    python setup.py

    

> **GPU 12 GB 이상**을 권장합니다 (fp16 + grad-accum 32).

---

## 데이터 준비

`data/aihub_interview.csv` — 헤더는 **question, answer, summary** 3열이어야 합니다.

| question | answer | summary |
|----------|--------|---------|
| 질문 텍스트 | 답변 텍스트 | 요약(정답) |

---

## 학습

    python train.py

- train/val = 99 : 1  
- 배치 1 × gradient_accumulation_steps 32  
- 1 에폭마다 평가 & 체크포인트(`checkpoints/gpt2-lora/...`) 저장

---

## LoRA → 4-bit 양자화

    python quantization.py
    # 결과: checkpoints/gpt2-bnb-4bit/

---

## 추론

    python inference.py \

스크립트는 Alpaca 템플릿에 질답을 삽입해 최대 120 토큰을 생성하고  
`### Response:` 뒤의 요약만 출력합니다.

---

## 메트릭

- **ROUGE-1/2/L**, **BLEU** (BERTScore 제거로 속도 향상)

---

## 커스터마이징

| 항목               | 파일               | 변경 위치 |
|--------------------|--------------------|-----------|
| 프롬프트 문구      | `utils/prompts.py` | `TEMPLATE` |
| 최대 입력 길이     | `utils/data.py`    | `max_len` |
| LoRA r·α           | `train.py`         | `LoraConfig` |
| 평가/저장 주기     | `train.py`         | `eval_strategy`, `save_strategy` |
| 8-bit 양자화       | `quantization.py`  | `load_in_4bit=False`, `load_in_8bit=True` |

---

## 라이선스

- **코드**: MIT  
- **데이터**: AI Hub 면접 QA 데이터셋 (해당 라이선스 준수)

---

Happy fine-tuning!
