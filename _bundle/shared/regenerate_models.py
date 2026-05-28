"""원본 대본을 여러 모델로 동시에 재작성해 결과·비용을 비교 (재생성 전용 어댑터).

모델 카탈로그·선택·병렬·집계는 범용 shared.llm_models가 담당하고, 여기서는 '재생성 호출'만
끼워 넣는다(중복 없음). 재작성 알고리즘(프롬프트·느낌 보존)은 shared.regenerate.regenerate.

사용 예:
    from shared.regenerate_models import regenerate_models
    out = regenerate_models(original, "주방 세정제", "기름때 한 번에", n=3, client=client)
    # out = {"results": [{key,label,tier,model, script, cost} | {..., error}], "total_cost", "models"}
    # (모델 개수/종류는 n=1/2/3 프리셋 또는 models=["haiku","sonnet"] 키 리스트)
"""
# MODELS·PRESETS·resolve_models는 범용 모듈 것을 재수출 (기존 import 호환).
from shared.llm_models import run_on_models, resolve_models, MODELS, PRESETS  # noqa: F401
from shared.regenerate import regenerate


def regenerate_models(original_script: str, topic: str, selling_point: str,
                      models=None, n: int = 3, client=None, hook_style: str = None,
                      emphasis: str = None) -> dict:
    """선택된 모델들로 각각 재작성(병렬). 모델별 결과(script)·비용 + 합계 반환.

    반환: {
      "results": [  # 저렴→고품질 순
        {"key","label","tier","model","script","cost"}  # 성공
        | {"key","label","tier","model","error"}          # 실패
      ],
      "total_cost": float,
      "models": [key, ...],
    }
    """
    def _call(c, model_id):
        return regenerate(original_script, topic, selling_point,
                          client=c, model=model_id, hook_style=hook_style, emphasis=emphasis)

    out = run_on_models(_call, models=models, n=n, client=client)
    for r in out["results"]:
        if "result" in r:          # 범용 'result' → 재생성 맥락의 'script'로 노출
            r["script"] = r.pop("result")
    return out
