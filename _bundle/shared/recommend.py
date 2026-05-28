"""
hub.db 대본 추천 — 순수 알고리즘.
호출자가 df(load_patterns_df 결과)와 client를 주입한다. DB 접근 없음.

사용 예:
    from shared.recommend import recommend, classify_category, CATEGORIES
    results, cost = recommend(df, "찌든 기름때", n=5, exclude_category="청소·세탁", client=client)
    # results: list[dict] — rank, code, selling_point, hook_sentence, product_name, product_category, hook_type
    # 대본 전문(text)은 미포함 — 호출자가 db.resolve_transcript_by_code(code)로 붙인다
"""
from difflib import SequenceMatcher

CATEGORIES = [
    "가구·인테리어", "식품·음료", "뷰티·화장품", "청소·세탁",
    "수납·정리", "기타", "가전·디지털", "주방·조리", "패션·의류",
    "미술·문구", "장난감·교육", "자동차", "운동·캠핑", "취미·공예",
    "건강·의약", "식물·원예", "반려동물",
]

_HAIKU_MODEL = "claude-haiku-4-5-20251001"
_HAIKU_PRICING = {"input": 0.80, "output": 4.00}


def _haiku_cost(usage) -> float:
    return (usage.input_tokens * _HAIKU_PRICING["input"] + usage.output_tokens * _HAIKU_PRICING["output"]) / 1_000_000


def _str(val) -> str:
    """pandas NaN 포함 모든 값을 안전하게 str 변환. NaN → ''."""
    if val is None:
        return ""
    s = str(val)
    return "" if s == "nan" else s


def prefilter(df, query: str, size: int = 120) -> set:
    """어휘 유사도(difflib) 기준 상위 size개 code 집합 반환. 랜덤 없음."""
    if len(df) <= size:
        return set(df["code"].tolist())
    scored = [
        (row["code"], SequenceMatcher(None, query, _str(row["selling_point"])).ratio())
        for _, row in df.iterrows()
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return {code for code, _ in scored[:size]}


def classify_category(product: str, client=None, categories=None) -> tuple:
    """제품명 → hub.db 카테고리 17종 중 하나. (matched_category, cost) 반환."""
    if categories is None:
        categories = CATEGORIES
    if client is None:
        import __main__
        client = __main__.client
    cats = "\n".join(f"- {c}" for c in categories)
    resp = client.messages.create(
        model=_HAIKU_MODEL,
        max_tokens=32,
        temperature=0,
        messages=[{
            "role": "user",
            "content": (
                f"다음 제품이 아래 카테고리 중 어디에 속하는지 카테고리 이름만 답하세요.\n\n"
                f"제품: {product}\n\n카테고리:\n{cats}\n\n"
                f"카테고리 이름만, 다른 말 없이:"
            ),
        }],
    )
    cost = _haiku_cost(resp.usage)
    raw = resp.content[0].text.strip()
    for cat in categories:
        if cat in raw:
            return cat, cost
    return raw, cost


def recommend(df, selling_point: str, n: int = 5,
              exclude_category: str = "", client=None) -> tuple:
    """
    df + 소구점 → 추천 결과.

    반환: (list[dict], 총비용)
    dict 키: rank, code, selling_point, hook_sentence, product_name, product_category, hook_type
    대본 전문(text)은 미포함 — 호출자가 db.resolve_transcript_by_code(code)로 붙인다.
    """
    from shared.pools import rank_top_n

    if client is None:
        import __main__
        client = __main__.client

    df_filtered = df
    if exclude_category:
        df_filtered = df[df["product_category"] != exclude_category].reset_index(drop=True)

    pool = prefilter(df_filtered, selling_point)
    top_codes, cost = rank_top_n(df_filtered, pool, [selling_point], n, client=client)

    results = []
    for rank_num, code in enumerate(top_codes, 1):
        row = df_filtered[df_filtered["code"] == code]
        results.append({
            "rank": rank_num,
            "code": code,
            "selling_point": _str(row["selling_point"].iloc[0]) if not row.empty else "",
            "hook_sentence": _str(row["hook_sentence"].iloc[0]) if not row.empty else "",
            "product_name": _str(row["product_name"].iloc[0]) if not row.empty else "",
            "product_category": _str(row["product_category"].iloc[0]) if not row.empty else "",
            "hook_type": _str(row["hook_type"].iloc[0]) if not row.empty else "",
        })

    return results, cost
