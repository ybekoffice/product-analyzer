"""
원본 대본 전문을 구조 템플릿으로 → 새 소재로 재생성. 순수 알고리즘.
호출자가 original_script(db에서 조회)와 client를 주입한다. DB 접근 없음.

핵심: 원본 대본을 통째로 넣어 "느낌(화법·호흡·톤)"을 보존하고, 내용(제품/소구점)만 교체한다.

사용 예:
    from shared.regenerate import regenerate
    new_script, cost = regenerate(original_script, "주방 세정제",
                                  "기름때 한 번에 닦임", client=client)
"""

_MODEL_PRICING = {
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-sonnet-4-6":         {"input": 3.0,  "output": 15.0},
    "claude-opus-4-7":           {"input": 15.0, "output": 75.0},
}


def _cost(model: str, usage) -> float:
    p = _MODEL_PRICING.get(model, {"input": 3.0, "output": 15.0})
    return (usage.input_tokens * p["input"] + usage.output_tokens * p["output"]) / 1_000_000


_PROMPT = """당신은 숏폼 영상 대본 전문 카피라이터입니다.
아래 [구조 참고 대본]의 구조와 느낌을 그대로 살려서, [새 소재]로 완전히 새로운 대본을 써주세요.

---

## [구조 참고 대본]

여기서는 **오직 호흡·순서·구조·분위기·화법만** 가져옵니다.
이 대본의 내용(제품, 소재)은 절대 가져오지 마세요.

```
__ORIGINAL_SCRIPT__
```

---

## [새 소재]

이 대본의 실제 내용은 아래 소재로만 채웁니다.

- 새 주제(제품/대상): __TOPIC__
- 강조할 소구점: __SELLING_POINT__

---

__EMPHASIS_RULE__## 작성 규칙

1. **구조 복제**: [구조 참고 대본]의 전개 순서(도입 → 전개 → 마무리의 흐름)를 그대로 따라가세요.
2. **느낌 보존 (가장 중요)**: 원본의 다음 요소를 최대한 똑같이 유지하세요.
   - 문장 길이와 호흡 (짧게 끊는지, 길게 가는지)
   - 구어체 말투와 어미 ("~더라구요", "~거든요" 같은 톤)
   - 화법 특징 (의성어/의태어, 숫자 활용, 질문형, 감탄 등)
   - 전체 분위기 (친근함, 다급함, 유머 등)
3. **내용 교체**: 제품·상황·구체 표현은 [새 소재]에 맞게 완전히 새로 쓰세요. 원본 제품명은 절대 등장 금지.
4. 숏폼 길이(원본과 비슷한 분량)를 유지하세요.
__HOOK_RULE__
---

## 출력 형식

설명, 인사말, 제목 없이 **새 대본 본문만** 출력하세요.
"""


def regenerate(original_script: str, topic: str, selling_point: str,
               client=None, model: str = "claude-sonnet-4-6",
               hook_style: str = None, emphasis: str = None) -> tuple:
    """
    원본 대본 + 새 소재 → 재생성 대본.

    반환: (new_script, cost)
    hook_style이 주어지면 도입부를 그 스타일로 강하게 시작하라는 규칙을 추가한다.
    emphasis가 주어지면 '사용자 최우선 요구'로 프롬프트 맨 위에 박아 가장 강하게 반영한다
    (후킹·소구점·등장인물/톤 등 자유 텍스트). 비어 있으면 기존과 동일한 프롬프트.
    """
    if client is None:
        import __main__
        client = __main__.client

    hook_rule = ""
    if hook_style:
        hook_rule = f"5. **도입부 후킹**: 첫 문장은 '{hook_style}' 스타일로 강하게 시작하세요.\n"

    emphasis_rule = ""
    if emphasis and emphasis.strip():
        emphasis_rule = (
            "## ⚠️ [최우선 사용자 요구] — 아래 모든 규칙보다 우선합니다\n"
            "사용자가 이 대본에서 반드시 반영하길 원하는 요구입니다. "
            "무엇보다 먼저, 가장 강하게 반영하세요:\n"
            f"\"{emphasis.strip()}\"\n"
            "(후킹 방식·소구점·등장인물/톤 등 어떤 요구든 위 내용을 최우선으로 따르되, "
            "원본의 화법·호흡 보존과 자연스럽게 어우러지게 녹이세요.)\n\n"
        )

    prompt = (
        _PROMPT
        .replace("__ORIGINAL_SCRIPT__", original_script)
        .replace("__TOPIC__", topic)
        .replace("__SELLING_POINT__", selling_point)
        .replace("__HOOK_RULE__", hook_rule)
        .replace("__EMPHASIS_RULE__", emphasis_rule)
    )
    kwargs = {"model": model, "max_tokens": 1024,
              "messages": [{"role": "user", "content": prompt}]}
    if "opus" not in model:  # opus는 temperature+thinking 병용 제약 → rewrite.py와 동일 가드
        kwargs["temperature"] = 0.7
    resp = client.messages.create(**kwargs)
    return resp.content[0].text.strip(), _cost(model, resp.usage)


# (2단계 예정) 한 원본으로 후킹 스타일 N종 변형 동시 생성
# def regenerate_variations(original_script, topic, selling_point,
#                           hook_styles: list, client=None, model="claude-sonnet-4-6") -> tuple:
#     results, total = [], 0.0
#     for hs in hook_styles:
#         text, c = regenerate(original_script, topic, selling_point,
#                              client=client, model=model, hook_style=hs)
#         results.append({"hook_style": hs, "script": text}); total += c
#     return results, total
