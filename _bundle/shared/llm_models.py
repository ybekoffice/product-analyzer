"""여러 LLM 모델로 같은 작업을 동시에 돌려 결과·비용을 비교하거나, 모델 개수/종류를 선택하는 범용 유틸.

작업 종류(재생성·분석·분류·요약 등)와 무관하다. 호출자는 `call(client, model_id) -> (결과, 비용)`
함수 하나만 주면, 선택된 모델들에 병렬로 적용하고 결과·비용을 모아준다.
여기서는 **모델 카탈로그·선택·병렬·비용집계**만 담당하고, 무엇을 시키는지(프롬프트·파싱)는 call이 정한다.

cf. shared.llm_batch / usability_run 은 'DB 수천~수만 행을 한 모델로' 배치하는 다른 축이다.
    여기는 '한 작업을 여러 모델로' 비교/선택하는 축. (knowledge/multi-model-compare.md)

사용 예:
    from shared.llm_models import run_on_models
    def call(client, model_id):
        resp = client.messages.create(model=model_id, max_tokens=512,
                                       messages=[{"role": "user", "content": prompt}])
        return resp.content[0].text, cost_of(resp)      # (결과, 비용)
    out = run_on_models(call, n=3, client=client)         # 1/2/3개 프리셋
    # out = {"results":[{key,label,tier,model, result, cost} | {..., error}], "total_cost", "models"}
"""
from concurrent.futures import ThreadPoolExecutor

# 비교/선택 가능한 모델 카탈로그 (저렴 → 고품질). key = 호출자/UI가 쓰는 짧은 이름.
MODELS = {
    "haiku":  {"id": "claude-haiku-4-5-20251001", "label": "하이쿠 4.5", "tier": "저렴·빠름"},
    "sonnet": {"id": "claude-sonnet-4-6",         "label": "소넷 4.6",   "tier": "균형"},
    "opus":   {"id": "claude-opus-4-7",           "label": "오푸스 4.7", "tier": "고품질·비쌈"},
}
# 개수 선택 시 채우는 순서(저렴→고품질). 1/2/3개는 이 순서 앞에서부터 잘라 쓴다.
MODEL_ORDER = ["haiku", "sonnet", "opus"]
# 개수별 기본 프리셋. n=1/2/3로 호출하면 이 조합을 쓴다.
#  1개=소넷(균형 단일), 2개=하이쿠+소넷(저렴·균형 비교), 3개=전체 비교.
PRESETS = {
    1: ["sonnet"],
    2: ["haiku", "sonnet"],
    3: ["haiku", "sonnet", "opus"],
}


def resolve_models(models=None, n=3) -> list:
    """models(키 리스트) 또는 n(1/2/3 프리셋) → 정규화된 모델 키 리스트.

    - models가 주어지면 그것을 쓰되 알 수 없는 키는 ValueError.
    - models가 없으면 PRESETS[n]. n이 1~3이 아니면 ValueError.
    - 반환은 항상 MODEL_ORDER(저렴→고품질) 순서로 정렬·중복 제거.
    """
    if models is None:
        if n not in PRESETS:
            raise ValueError(f"n은 1·2·3만 가능합니다 (받은 값: {n}). 또는 models=로 직접 지정하세요.")
        models = PRESETS[n]
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        raise ValueError(f"알 수 없는 모델 키: {unknown}. 가능: {list(MODELS)}")
    seen = set(models)
    return [k for k in MODEL_ORDER if k in seen]


def run_on_models(call, models=None, n=3, client=None, max_workers=None) -> dict:
    """call(client, model_id) -> (결과, 비용)을 선택된 모델들에 병렬 적용.

    call: 모델 1개로 작업을 수행하고 (결과, 비용(float))을 반환하는 함수.
          비용을 계산하기 어려우면 0.0을 반환해도 된다.
    models/n: resolve_models 규칙. client: call에 그대로 전달(없으면 __main__.client).
    max_workers: 기본은 모델 수(보통 ≤3이라 그대로 동시 실행).

    반환: {
      "results": [  # MODEL_ORDER(저렴→고품질) 순
        {"key","label","tier","model","result","cost"}  # 성공
        | {"key","label","tier","model","error"}          # 실패 (한 모델 실패가 나머지를 막지 않음)
      ],
      "total_cost": float,   # 성공분 비용 합
      "models": [key, ...],  # 실제 사용한 모델 키
    }
    """
    if client is None:
        import __main__
        client = __main__.client

    keys = resolve_models(models=models, n=n)
    workers = max_workers or len(keys)

    def _one(key):
        return call(client, MODELS[key]["id"])

    by_key = {}
    with ThreadPoolExecutor(max_workers=workers) as exr:
        futs = {exr.submit(_one, k): k for k in keys}
        for fut in futs:
            k = futs[fut]
            meta = MODELS[k]
            base = {"key": k, "label": meta["label"], "tier": meta["tier"], "model": meta["id"]}
            try:
                result, cost = fut.result()
                by_key[k] = {**base, "result": result, "cost": cost}
            except Exception as ex:
                by_key[k] = {**base, "error": str(ex)}

    results = [by_key[k] for k in keys]  # MODEL_ORDER 순 유지
    total = sum(r.get("cost", 0.0) for r in results)
    return {"results": results, "total_cost": total, "models": keys}
