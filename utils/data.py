# utils/data.py
from utils.prompts import TEMPLATE

def _make_prompt(q: str, a: str) -> str:
    return TEMPLATE.format(
        instruction="다음 면접 질문과 답변을 읽고 한국어로 간결하게 요약하세요.",
        input=f"질문: {q}\n답변: {a}"
    )

def tokenize_alpaca(batch, tokenizer, max_len=1024,
                    prompt_max=880, summary_max=120):
    """
    ★ 1024 토큰 절대 초과 방지
    ★ 배치(list) 입력 전용
    """
    ins, labs = [], []
    for q, a, s in zip(batch["question"], batch["answer"], batch["summary"]):
        p = tokenizer(_make_prompt(q, a), add_special_tokens=False).input_ids[:prompt_max]
        y = tokenizer(s.strip(), add_special_tokens=False).input_ids[:summary_max]
        y += [tokenizer.eos_token_id]

        overflow = len(p) + len(y) - max_len
        if overflow > 0:
            p = p[overflow:]                  # prompt 앞부분 잘라서 길이 맞춤

        ids = p + y
        ins.append(ids)
        labs.append([-100] * len(p) + y)

    return {"input_ids": ins, "labels": labs}
