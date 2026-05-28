"""공유 분류 함수. LLM 프롬프트: harness/prompts/classify_caption.md, classify_coupang.md"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import anthropic

MODEL = "claude-haiku-4-5-20251001"
BATCH_SIZE = 25


def _load_prompt(name: str, prompts_dir: Path) -> str:
    text = (prompts_dir / name).read_text(encoding="utf-8")
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            return parts[2].lstrip("\n")
    return text


def classify_batch(client: anthropic.Anthropic, batch: list[dict], prompts_dir: Path, model: str = MODEL,
                   prompt_name: str = "classify_caption.md") -> list[dict]:
    """캡션+대본 배치 → topic, product_type, is_recipe 추출.

    batch 항목: {"id": str, "caption": str, "transcript": str}  (transcript 없으면 빈 문자열)
    반환: [{"id": str, "topic": str, "product_type": str, "is_recipe": bool}]
    prompt_name으로 추출 프롬프트 파일 교체 가능(기본 classify_caption.md).
    """
    parts = []
    for i, item in enumerate(batch):
        caption = (item.get("caption") or "")[:400]
        transcript = (item.get("transcript") or "")[:600]
        if transcript:
            parts.append(f"[{i+1}] id={item['id']}\n캡션: {caption}\n대본: {transcript}")
        else:
            parts.append(f"[{i+1}] id={item['id']}\n캡션: {caption}")
    items_text = "\n\n".join(parts)

    prompt = _load_prompt(prompt_name, prompts_dir).replace("__ITEMS__", items_text)
    resp = client.messages.create(
        model=model,
        max_tokens=4000,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    start = raw.find("[")
    if start == -1:
        raise ValueError("JSON 배열 없음")
    depth = 0
    for i, ch in enumerate(raw[start:], start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return json.loads(raw[start:i + 1])
    raise ValueError("JSON 배열 종료 괄호 없음")


def _warn_missing(out: dict, product_types: list[str]) -> None:
    """LLM이 일부 입력을 누락해 키가 비면 stderr 경고(침묵 실패 조기 발견)."""
    missing = [t for t in product_types if t not in out]
    if missing:
        print(f"[classify] 매핑 누락 {len(missing)}/{len(product_types)}종: {missing[:5]}", file=sys.stderr)


def _resolve_key(r: dict, product_types: list[str]) -> str | None:
    """매핑 결과 항목 → 원래 product_type 키 복원.

    index(입력 번호, 1-based)가 있으면 우선 사용(긴 문자열 에코 누락 예방).
    없거나 범위 밖이면 type 에코로 폴백(v1 등 index 미출력 프롬프트 호환).
    """
    idx = r.get("index")
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        idx = None
    if idx is not None and 1 <= idx <= len(product_types):
        return product_types[idx - 1]
    if "type" in r:
        return r["type"]
    return None


def map_coupang_batch(client: anthropic.Anthropic, product_types: list[str], prompts_dir: Path, model: str = MODEL,
                      prompt_name: str = "classify_coupang.md") -> dict[str, dict]:
    """product_type 목록 → 쿠팡 대/중분류 매핑.

    반환: {product_type: {"major": str, "mid": str}}
    prompt_name으로 매핑 프롬프트 파일 교체 가능(기본 classify_coupang.md).
    키 복원은 index(입력 번호) 우선, 없으면 type 에코 폴백.
    """
    items_text = "\n".join(f"{i+1}. {t}" for i, t in enumerate(product_types))
    prompt = _load_prompt(prompt_name, prompts_dir).replace("__ITEMS__", items_text)
    resp = client.messages.create(
        model=model,
        max_tokens=8000,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    start = raw.find("[")
    if start == -1:
        raise ValueError("JSON 배열 없음")
    depth = 0
    for i, ch in enumerate(raw[start:], start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                results = json.loads(raw[start:i + 1])
                out: dict[str, dict] = {}
                for r in results:
                    key = _resolve_key(r, product_types)
                    if key is None:
                        continue
                    out[key] = {"major": r.get("major", "기타"), "mid": r.get("mid", "기타")}
                _warn_missing(out, product_types)
                return out
    raise ValueError("JSON 배열 종료 괄호 없음")


def map_coupang_batch_multi(client: anthropic.Anthropic, product_types: list[str], prompts_dir: Path,
                            model: str = MODEL, prompt_name: str = "classify_coupang_v4.md",
                            max_cats: int = 2) -> dict[str, list]:
    """product_type 목록 → 쿠팡 카테고리 매핑(경계성 제품 다중 표현).

    반환: {product_type: [ {"major","mid"}, ... ]}  (주 먼저, 1~max_cats개)
    프롬프트는 [{"index","categories":[{"major","mid"},...]}] 형식을 반환해야 함.
    """
    items_text = "\n".join(f"{i+1}. {t}" for i, t in enumerate(product_types))
    prompt = _load_prompt(prompt_name, prompts_dir).replace("__ITEMS__", items_text)
    resp = client.messages.create(
        model=model,
        max_tokens=8000,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    start = raw.find("[")
    if start == -1:
        raise ValueError("JSON 배열 없음")
    depth = 0
    for i, ch in enumerate(raw[start:], start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                results = json.loads(raw[start:i + 1])
                out: dict[str, list] = {}
                for r in results:
                    key = _resolve_key(r, product_types)
                    if key is None:
                        continue
                    cats = r.get("categories") or []
                    norm = [
                        {"major": c.get("major", "기타"), "mid": c.get("mid", "기타")}
                        for c in cats if isinstance(c, dict)
                    ][:max_cats]
                    if not norm:
                        norm = [{"major": "기타", "mid": "기타"}]
                    out[key] = norm
                _warn_missing(out, product_types)
                return out
    raise ValueError("JSON 배열 종료 괄호 없음")


def classify_to_category(
    client: anthropic.Anthropic,
    items: list[dict],
    prompts_dir: Path,
    cache: dict | None = None,
    caption_prompt: str = "classify_caption.md",
    coupang_prompt: str = "classify_coupang.md",
    model: str = MODEL,
    mapping_model: str | None = None,
    batch_size: int = BATCH_SIZE,
    max_categories: int = 1,
) -> tuple[list[dict], dict]:
    """캡션+대본 → topic, product_type, 카테고리(최대 max_categories개)까지. DB 접근 없음(순수).

    어떤 DB에도 묶이지 않는 이식 가능한 분류 로직.
    - items 항목: {"id", "caption", "transcript"}
    - cache: 주입식 메모리 dict. 없으면 새로 생성. 같은 product_type은 LLM 매핑 1회만.
    - model: 추출 단계 모델(기본 Haiku).
    - mapping_model: 매핑 단계 모델. None이면 model과 동일(하위호환). 추출=Haiku·매핑=Sonnet 등 분리 지원.
    - max_categories: 유지할 최대 카테고리 수. 파서 선택은 coupang_prompt 출력 형식으로 자동 결정.
    반환: (results, cache)
      results 항목: {"id","topic","product_type","is_recipe","categories":[{"major","mid"},...],"major","mid"}
      (major/mid = categories[0], 하위호환 편의)
    """
    if cache is None:
        cache = {}
    mapping_model = mapping_model or model
    multi = max_categories > 1
    coupang_text = _load_prompt(coupang_prompt, prompts_dir)
    multi_parser = '"categories"' in coupang_text

    items_by_id = {item.get("id"): item for item in items}
    out: list[dict] = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        extracted = classify_batch(client, batch, prompts_dir, model=model, prompt_name=caption_prompt)

        unknown = sorted({
            r["product_type"] for r in extracted
            if r.get("product_type") and r["product_type"] != "미상" and r["product_type"] not in cache
        })
        if unknown:
            if multi_parser:
                cache.update(map_coupang_batch_multi(client, unknown, prompts_dir, model=mapping_model, prompt_name=coupang_prompt, max_cats=max(max_categories, 1)))
            else:
                cache.update(map_coupang_batch(client, unknown, prompts_dir, model=mapping_model, prompt_name=coupang_prompt))

        ext_by_id = {r.get("id"): r for r in extracted}
        for item in batch:
            iid = item.get("id")
            r = ext_by_id.get(iid)
            if r is None:
                out.append({
                    "id": iid,
                    "topic": "",
                    "product_type": "미상",
                    "is_recipe": False,
                    "categories": [{"major": "", "mid": ""}],
                    "major": "",
                    "mid": "",
                    "extract_status": "dropped",
                    "map_status": "skip",
                })
                continue
            pt = r.get("product_type", "미상")
            entry = cache.get(pt)
            if multi:
                cats = entry if isinstance(entry, list) else ([entry] if entry else [{"major": "", "mid": ""}])
            else:
                cats = [entry] if isinstance(entry, dict) else (entry if isinstance(entry, list) and entry else [{"major": "", "mid": ""}])
            primary = cats[0]
            if pt == "미상":
                extract_status = "unknown"
                map_status = "skip"
            else:
                extract_status = "ok"
                map_status = "ok" if primary.get("major") else "miss"
            out.append({
                "id": r.get("id"),
                "topic": r.get("topic", ""),
                "product_type": pt,
                "is_recipe": r.get("is_recipe", False),
                "categories": cats,
                "major": primary.get("major", ""),
                "mid": primary.get("mid", ""),
                "extract_status": extract_status,
                "map_status": map_status,
            })

    # dropped 항목 1회 자동 재시도
    dropped_ids = {r["id"] for r in out if r.get("extract_status") == "dropped"}
    if dropped_ids:
        retry_items = [items_by_id[iid] for iid in dropped_ids if iid in items_by_id]
        print(f"[classify] 추출 누락 {len(retry_items)}건 재시도...", file=sys.stderr)
        retry_extracted = classify_batch(client, retry_items, prompts_dir, model=model, prompt_name=caption_prompt)

        retry_unknown = sorted({
            r["product_type"] for r in retry_extracted
            if r.get("product_type") and r["product_type"] != "미상" and r["product_type"] not in cache
        })
        if retry_unknown:
            if multi_parser:
                cache.update(map_coupang_batch_multi(client, retry_unknown, prompts_dir, model=mapping_model, prompt_name=coupang_prompt, max_cats=max(max_categories, 1)))
            else:
                cache.update(map_coupang_batch(client, retry_unknown, prompts_dir, model=mapping_model, prompt_name=coupang_prompt))

        retry_by_id = {r.get("id"): r for r in retry_extracted}
        out_idx = {r["id"]: i for i, r in enumerate(out)}
        for item in retry_items:
            iid = item.get("id")
            r = retry_by_id.get(iid)
            idx = out_idx.get(iid)
            if idx is None or r is None:
                continue
            pt = r.get("product_type", "미상")
            entry = cache.get(pt)
            if multi:
                cats = entry if isinstance(entry, list) else ([entry] if entry else [{"major": "", "mid": ""}])
            else:
                cats = [entry] if isinstance(entry, dict) else (entry if isinstance(entry, list) and entry else [{"major": "", "mid": ""}])
            primary = cats[0]
            if pt == "미상":
                extract_status, map_status = "unknown", "skip"
            else:
                extract_status = "ok"
                map_status = "ok" if primary.get("major") else "miss"
            out[idx] = {
                "id": r.get("id"),
                "topic": r.get("topic", ""),
                "product_type": pt,
                "is_recipe": r.get("is_recipe", False),
                "categories": cats,
                "major": primary.get("major", ""),
                "mid": primary.get("mid", ""),
                "extract_status": extract_status,
                "map_status": map_status,
            }

    return out, cache
