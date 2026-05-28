"""
소구점 의미 유사도 랭킹 (shared).

후보 대본들의 selling_point를 Haiku에게 보내, 사용자 소구점과 의미가 가장
비슷한 top-N을 고른다. 추천 파이프라인의 마지막 의미 매칭 단계.

shared/pools.py 에서 분리(2026-05-28). pools.py는 호환을 위해 여기서 재수출한다.
"""
import json
import re


_HAIKU_PRICING = {"input": 0.80, "output": 4.00}


def _haiku_cost(usage) -> float:
    return (usage.input_tokens * _HAIKU_PRICING["input"] + usage.output_tokens * _HAIKU_PRICING["output"]) / 1_000_000


def rank_top_n(df, pool_codes: set, selling_points: list[str], n: int, client=None) -> tuple[list[str], float]:
    if client is None:
        import __main__
        client = __main__.client

    pool_df = df[df["code"].isin(pool_codes)][["code", "selling_point"]].dropna(subset=["selling_point"])
    if pool_df.empty:
        return [], 0.0
    candidates = pool_df.to_dict("records")
    if len(candidates) <= n:
        return [c["code"] for c in candidates], 0.0

    candidates_text = "\n".join(
        f"[{i}] code={c['code']} sp={c['selling_point'][:80]}"
        for i, c in enumerate(candidates)
    )
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        temperature=0,
        messages=[{
            "role": "user",
            "content": (
                f"사용자 소구점과 의미가 가장 비슷한 대본 top {n}개를 선택하세요.\n\n"
                f"사용자 소구점:\n{json.dumps(selling_points, ensure_ascii=False)}\n\n"
                f"후보 대본 목록:\n{candidates_text}\n\n"
                f"반드시 JSON 배열만 반환하세요 (code {n}개):\n[\"code1\", \"code2\", ...]"
            ),
        }],
    )
    cost = _haiku_cost(resp.usage)
    raw = resp.content[0].text.strip()
    match = re.search(r"\[[^\[\]]*\]", raw)
    if not match:
        return [c["code"] for c in candidates[:n]], cost
    return json.loads(match.group())[:n], cost


def rank_top_n_explained(df, pool_codes: set, selling_points: list[str], n: int, client=None) -> tuple[list[dict], float]:
    """rank_top_n의 설명 버전: 각 pick에 일치율(0~100)·한 줄 이유를 함께 반환.

    rank_top_n과 별개 함수다 — code만 필요한 다른 프로젝트(ringbob·data-hub·
    script-analyzer 등) 호출부를 깨지 않으려고 시그니처를 건드리지 않고 분리했다.
    product-analyzer(recommend_engine)만 사용한다.

    반환: (picks, cost)
      picks: [{"code", "score"(0~100 또는 None), "reason"}, ...] — 유사도 높은 순
    """
    if client is None:
        import __main__
        client = __main__.client

    pool_df = df[df["code"].isin(pool_codes)][["code", "selling_point"]].dropna(subset=["selling_point"])
    if pool_df.empty:
        return [], 0.0
    candidates = pool_df.to_dict("records")
    valid_codes = {str(c["code"]) for c in candidates}

    candidates_text = "\n".join(
        f"[{i}] code={c['code']} sp={c['selling_point'][:80]}"
        for i, c in enumerate(candidates)
    )
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        temperature=0,
        messages=[{
            "role": "user",
            "content": (
                f"사용자 소구점과 의미가 가장 비슷한 대본 top {n}개를 고르고, "
                f"각각 일치율과 한국어 이유를 쓰세요.\n"
                f"- 일치율: 두 소구점이 겨냥하는 핵심 효용·타겟·맥락이 얼마나 겹치는지를 0~100 정수로.\n"
                f"- 이유: 무엇이 겹쳐서 일치하는지 15~40자로 구체적으로(예: '둘 다 자취생 가성비·공간절약 강조').\n\n"
                f"사용자 소구점:\n{json.dumps(selling_points, ensure_ascii=False)}\n\n"
                f"후보 대본 목록:\n{candidates_text}\n\n"
                f"반드시 JSON 배열만 반환하세요 (유사도 높은 순 {n}개):\n"
                f'[{{"code": "...", "score": 0, "reason": "..."}}]'
            ),
        }],
    )
    cost = _haiku_cost(resp.usage)
    raw = resp.content[0].text.strip()

    def _fallback():
        return [{"code": str(c["code"]), "score": None, "reason": ""} for c in candidates[:n]]

    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return _fallback(), cost
    try:
        arr = json.loads(match.group())
    except (ValueError, json.JSONDecodeError):
        return _fallback(), cost

    picks, seen = [], set()
    for item in arr:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code", ""))
        if code not in valid_codes or code in seen:
            continue
        score = item.get("score")
        try:
            score = max(0, min(100, int(round(float(score)))))
        except (TypeError, ValueError):
            score = None
        seen.add(code)
        picks.append({"code": code, "score": score, "reason": str(item.get("reason", "")).strip()})
        if len(picks) >= n:
            break
    return (picks or _fallback()), cost
