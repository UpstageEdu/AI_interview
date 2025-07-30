# AI_interview

GPT-2 LoRA를 이용해 **면접 질문·답변 → 한국어 요약**을 생성하는 파인튜닝 파이프라인입니다.  
CSV 한 개만 준비하면 → 학습 → 4-bit 양자화 → 추론까지 바로 실행할 수 있습니다.

---

## 사전 요구사항

- **Python**: 3.11.8 이상
- **GPU**: CUDA 지원 GPU (권장, 최소 12GB VRAM, (fp16 + grad-accum 32))

### 운영체제
- Windows 10/11
- macOS 10.15 이상
- Ubuntu 18.04 이상

## 프로젝트 구조

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

## 빠른 시작

```bash
# 1. 저장소 클론 & 의존성 설치
$ git clone <repository-url>
$ cd AI_interview
$ python setup.py

# 2. 기본 설정으로 학습 (GPU 권장)
$ python train.py

# 3. 4‑bit 양자화 (선택)
$ python quantization.py 

# 4. 추론
$ python inference.py
```

---

## 데이터 준비

`data/aihub_interview.csv` — 헤더는 **question, answer, summary** 3열이어야 합니다.

| question | answer | summary |
|----------|--------|---------|
| 질문 텍스트 | 답변 텍스트 | 요약(정답) |

---

## 모델 학습

```bash
python train.py \
```

- train/val = 99 : 1  
- 배치 1 × gradient_accumulation_steps 32  
- 1 에폭마다 평가 & 체크포인트(`checkpoints/gpt2-lora/...`) 저장

---

## 4-bit 양자화

```bash
python quantization.py \
```

경로 오류 발생 시, 아래 경로를 확인 후 quantization.py 파일에서 경로를 수정해주세요!

```
%ls model-checkpoints/gpt2-lora/
```

```
CONFIG = dict(
    base_model = "gpt2",
    lora_dir   = "checkpoints/gpt2-lora/checkpoint-100", # 이 부분에서 경로 설정!
    save_dir   = "checkpoints/gpt2-bnb-4bit",
)
```

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
