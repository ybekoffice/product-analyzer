"""
main_test.db 세계 대본 추천 엔진 (소구점 의미 유사 + 같은 쿠팡 중분류 제외).

데이터 출처 (모두 읽기 전용, hub.db 미사용):
- usability.db    : usable=1 대본의 selling_point (소구점)
- classification.db: product_type + coupang_mid (카테고리)  ← hub.db 절대 안 씀
- main_test.db    : analysis.transcript (원본 대본 전문)

세 DB를 media_id로 매 호출마다 즉석 조립하므로, 데이터가 늘면 추천 풀도 자동으로 커진다.

핵심 원칙 (FEATURE_PLAN 2-2):
  소구점은 의미적으로 가깝게(Haiku 랭킹) + 같은 카테고리(coupang_mid)는 제외.
  카테고리 제외가 신선함의 안전장치이므로, 입력 제품의 중분류 판정을 하이브리드로 신뢰성 있게 한다.

입력 제품 중분류 판정 = 하이브리드:
  1) 기존 product_type과 글자 유사도가 임계값 이상이면 그 product_type의 coupang_mid를 무료로 상속
  2) 애매하면(임계 미만) shared.classify.classify_to_category(Haiku 추출 + Sonnet 매핑)로 정식 분류
"""
import os
import sqlite3
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

ENGINE_DIR = Path(__file__).resolve().parent          # .../harness/projects/main process
DATA_DIR = ENGINE_DIR / "data"
USABILITY_DB = DATA_DIR / "usability.db"
CLASSIFICATION_DB = DATA_DIR / "classification.db"
MAIN_TEST_DB = DATA_DIR / "main_test.db"
HOOK_LABELS_DB = DATA_DIR / "hook_labels.db"           # 훅 2축 라벨 (devices = 기법 태그)

HARNESS = ENGINE_DIR.parent.parent                    # .../harness (로컬) 또는 product-analyzer (번들)
# PROMPTS_DIR 폴백: 같은 엔진 파일이 (a) harness 원본 위치 (b) Streamlit Cloud 배포 번들
# (product-analyzer/_bundle/) 두 곳에서 그대로 동작해야 한다. 배포 시 _bundle/ 옆에 prompts/를
# 함께 두므로, 엔진 자기 옆에 prompts/가 있으면 그것을 우선 사용한다. 없으면 harness 원본 경로.
PROMPTS_DIR = ENGINE_DIR / "prompts"
if not PROMPTS_DIR.exists():
    PROMPTS_DIR = HARNESS / "prompts"
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))

# harness 공통 .env 로드. override=True로 ~/.zshrc의 낡은 키(특히 ANTHROPIC_API_KEY)가
# .env를 shadow하는 안티패턴(2026-05-28 승격) 차단. SUPABASE_DB_URL도 여기서 노출.
# app.py가 자기 부트스트랩에서 이미 같은 동작을 하지만, 엔진을 단독 import해 검증/스크립트로
# 쓸 때도 동일 환경이 보장돼야 백엔드 분기가 일관된다.
try:
    from dotenv import load_dotenv  # noqa: E402
    load_dotenv(HARNESS / ".env", override=True)
except Exception:
    pass  # dotenv 없는 환경(클라우드 Secrets 직접 주입)도 정상 동작

from shared.recommend import prefilter               # noqa: E402  소구점 어휘 1차 거르개
from shared.ranking import rank_top_n_explained       # noqa: E402  소구점 의미 유사도 랭킹 + 일치율·이유 (Haiku)
from shared.classify import classify_to_category      # noqa: E402  LLM 정식 분류 (하이브리드 폴백)

# 입력 제품을 기존 product_type에 글자 유사도로 매칭할 때의 임계값.
# 정확도 우선: 거의 동일한 경우만 무료 상속하고, 나머지는 LLM 폴백으로 보낸다.
# 0.90인 이유: 짧은 한글 제품명은 글자 겹침으로 의미가 다른데도 고득점한다
# (예: 수건→때수건 0.80, 바구니→장바구니 0.857 = 잘못된 카테고리 제외 위험).
# 관측된 오매칭(최대 0.857) 위로 안전 마진을 둬, 이런 함정을 LLM 분류로 넘긴다. (튜닝 가능)
SIMILARITY_THRESHOLD = 0.90
# 소구점 의미 랭킹 전 1차 어휘 거르개 크기. 의미 매칭이 미리 잘리지 않게 넉넉히.
PREFILTER_SIZE = 200
# 하이브리드 LLM 폴백 모델 (classification.db를 만든 v3+v5 설정과 동일)
EXTRACT_MODEL = "claude-haiku-4-5-20251001"
MAPPING_MODEL = "claude-sonnet-4-6"
CAPTION_PROMPT = "classify_caption_v3.md"
COUPANG_PROMPT = "classify_coupang_v5.md"
# 유형별 추천에서 기법(devices) 표시·중복제거 우선순위. 한 대본이 두 기법에
# 동시에 뽑혀 같은 순위면, 이 순서에서 앞선 기법 쪽에 남긴다. (CONTEXT.md 훅 2축 + direct_tip)
DEVICE_ORDER = [
    "shock", "demonstration", "social_proof", "story", "scarcity",
    "authority", "curiosity", "question", "empathy", "direct_tip",
]


# ---- 백엔드 분기 ------------------------------------------------------------
# 환경에 SUPABASE_DB_URL이 있으면 Postgres의 `script_candidates`(sync 스크립트가 만든
# 단일 denormalized 테이블)에서 읽고, 없으면 기존 SQLite 세 DB(usability + classification
# + main_test) 조합을 그대로 쓴다. 호출자 시그니처/반환 컬럼은 양쪽이 동일.
SUPABASE_TABLE = "script_candidates"


def _get_backend() -> str:
    """현재 활성 백엔드를 반환한다. 'supabase' 또는 'sqlite'.
    판별은 단순: SUPABASE_DB_URL 환경변수 유무. 비어있는 문자열도 미설정으로 본다."""
    return "supabase" if (os.getenv("SUPABASE_DB_URL") or "").strip() else "sqlite"


def _pg_connect():
    """Supabase Postgres 연결. psycopg2는 함수 안에서만 import 해서, 로컬 SQLite만
    쓰는 환경(테스트·CLI)에 의존성을 강제하지 않는다."""
    import psycopg2  # noqa: E402  의도된 지연 import
    dsn = os.getenv("SUPABASE_DB_URL", "").strip()
    if not dsn:
        raise RuntimeError("SUPABASE_DB_URL이 비어 있다 — 백엔드 분기 호출 순서 확인 필요")
    return psycopg2.connect(dsn)


def _connect_ro(path: Path) -> sqlite3.Connection:
    """읽기 전용 연결 (mode=ro). 활성 WAL 데이터까지 정확히 읽는다.

    immutable=1 폴백은 쓰지 않는다 — 외부 분류 프로세스가 classification.db의 WAL에
    실시간으로 쓰는 중이라, immutable로 떨어지면 WAL을 무시해 후보 풀이 조용히
    붕괴(예: 1만3천행→100행)할 수 있다. 실패하면 명시적으로 알린다(조용한 오답 방지)."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        conn.execute("SELECT name FROM sqlite_master LIMIT 1")  # 첫 쿼리에서 강제 오픈
    except sqlite3.OperationalError as ex:
        conn.close()
        raise RuntimeError(f"읽기 전용 DB 열기 실패: {path} ({ex})") from ex
    return conn


def load_candidates() -> pd.DataFrame:
    """추천 후보 df를 즉석 구성 (매 호출 = 최신 데이터 반영).

    백엔드 분기:
      - sqlite(로컬): usability + classification + main_test 세 DB를 media_id로 묶음
      - supabase: sync로 적재된 단일 테이블 `script_candidates`를 한 번에 읽음

    반환 컬럼: code(=media_id, str), selling_point, product_type, coupang_mid, coupang_major
    소구점·카테고리(coupang_mid)·transcript를 모두 갖춘 행만 남는다.
    """
    if _get_backend() == "supabase":
        # Supabase: sync 스크립트가 이미 transcript 있는 행만 적재 + selling_point/mid 채워진 행만
        # 적재했지만, 방어적으로 필터를 다시 건다(스키마 변경/부분 적재 안전망).
        conn = _pg_connect()
        try:
            df = pd.read_sql_query(
                f"SELECT code, selling_point, product_type, coupang_mid, coupang_major "
                f"FROM {SUPABASE_TABLE} "
                f"WHERE selling_point IS NOT NULL AND selling_point != '' "
                f"  AND coupang_mid IS NOT NULL AND coupang_mid != '' "
                f"  AND transcript IS NOT NULL AND transcript != ''",
                conn,
            )
        finally:
            conn.close()
        # sync 스크립트가 code를 str로 통일했지만, 로컬 경로와 dtype 일관성 확보.
        df["code"] = df["code"].astype(str)
        return df.reset_index(drop=True)

    # --- 로컬 SQLite 경로 (기존 로직 그대로) ---
    conn = _connect_ro(USABILITY_DB)
    try:
        sp = pd.read_sql_query(
            "SELECT media_id, selling_point FROM usability "
            "WHERE usable = 1 AND selling_point IS NOT NULL AND selling_point != ''",
            conn,
        )
    finally:
        conn.close()

    conn = _connect_ro(CLASSIFICATION_DB)
    try:
        cat = pd.read_sql_query(
            "SELECT media_id, product_type, coupang_mid, coupang_major FROM classification "
            "WHERE coupang_mid IS NOT NULL AND coupang_mid != ''",
            conn,
        )
    finally:
        conn.close()

    df = sp.merge(cat, on="media_id", how="inner")

    # 원본 대본(transcript)이 있는 행만 — 추천 결과가 항상 원본을 갖도록 보장
    conn = _connect_ro(MAIN_TEST_DB)
    try:
        tr = pd.read_sql_query(
            "SELECT media_id FROM analysis "
            "WHERE transcript IS NOT NULL AND transcript != ''",
            conn,
        )
    finally:
        conn.close()
    df = df[df["media_id"].isin(set(tr["media_id"]))]

    df = df.rename(columns={"media_id": "code"}).reset_index(drop=True)
    # Supabase 경로와 dtype 일치(호출 측 set/dict 키 일관성). 로컬은 정수일 수 있음.
    df["code"] = df["code"].astype(str)
    return df


def resolve_transcripts(media_ids: list) -> dict:
    """media_id 목록 → transcript dict (key: str). 한 번에 조회.

    백엔드 분기:
      - sqlite: main_test.db.analysis (기존 그대로)
      - supabase: script_candidates에서 code = ANY(...) — sync 시 transcript 동일 적재됨
    """
    if not media_ids:
        return {}
    # 호출 측이 int/str 섞어 줄 수 있으므로 str로 정규화(양쪽 백엔드 공통).
    ids = [str(m) for m in media_ids]

    if _get_backend() == "supabase":
        conn = _pg_connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT code, transcript FROM {SUPABASE_TABLE} "
                    f"WHERE code = ANY(%s) AND transcript IS NOT NULL AND transcript != ''",
                    (ids,),
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        return {str(mid): txt for mid, txt in rows}

    # --- 로컬 SQLite (기존 로직, ids를 str로 통일한 것만 차이) ---
    conn = _connect_ro(MAIN_TEST_DB)
    try:
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT media_id, transcript FROM analysis "
            f"WHERE media_id IN ({placeholders}) AND transcript IS NOT NULL AND transcript != ''",
            ids,
        ).fetchall()
    finally:
        conn.close()
    return {str(mid): txt for mid, txt in rows}


def classify_input_category(product_name: str, selling_points: list, df: pd.DataFrame, client,
                            threshold: float = SIMILARITY_THRESHOLD) -> dict:
    """입력 제품 → 같은 체계의 coupang_mid (하이브리드).

    1) 제품명이 기존 product_type과 글자 유사도 best가 임계 이상 → 그 product_type의 coupang_mid 상속 (method='similar')
    2) 미만 → classify_to_category(Haiku+Sonnet)로 정식 분류 (method='llm')
    반환: {coupang_mid, method, matched_product_type, ratio}
    """
    sim_text = (product_name or "").strip()  # 유사도는 깔끔한 제품명으로만 비교

    # product_type → coupang_mid 매핑 (최빈값)
    pt_map = {}
    if not df.empty:
        grouped = df.dropna(subset=["product_type", "coupang_mid"]).groupby("product_type")["coupang_mid"]
        for pt, series in grouped:
            if not pt:
                continue
            mode = series.mode()
            pt_map[pt] = mode.iloc[0] if not mode.empty else series.iloc[0]

    best_pt, best_ratio = "", 0.0
    if sim_text and pt_map:
        for pt in pt_map:
            r = SequenceMatcher(None, sim_text, pt).ratio()
            if r > best_ratio:
                best_ratio, best_pt = r, pt

    if best_ratio >= threshold:
        return {
            "coupang_mid": pt_map[best_pt],
            "method": "similar",
            "matched_product_type": best_pt,
            "ratio": round(best_ratio, 3),
            "cost": 0.0,  # 글자 유사도 상속 = LLM 미호출 = 무료
        }

    # LLM 폴백 — classification.db와 동일 방법. 제품명+소구점을 합친 캡션으로 추출 정확도↑
    sps = [s for s in (selling_points or []) if s and s.strip()]
    caption = sim_text
    if sps:
        caption = (sim_text + ". 소구점: " + " ".join(sps)).strip(". ")
    # classify_to_category의 2번째 반환은 cache(dict)지 비용이 아니다(docstring: (results, cache)).
    # 과거엔 이걸 cls_cost로 받아 "cost"에 dict를 넣어, recommend_by_device의
    # total_cost 합산에서 float+dict 크래시가 났다. 분류 비용은 이 함수가 노출하지 않으므로
    # 현재 미집계(0.0)로 둔다. (실제 분류 비용 추적은 shared.classify 변경이 필요한 별도 작업)
    results, _cache = classify_to_category(
        client,
        [{"id": "input", "caption": caption, "transcript": ""}],
        PROMPTS_DIR,
        caption_prompt=CAPTION_PROMPT,
        coupang_prompt=COUPANG_PROMPT,
        model=EXTRACT_MODEL,
        mapping_model=MAPPING_MODEL,
        max_categories=1,
    )
    r = results[0] if results else {}
    return {
        "coupang_mid": r.get("mid", ""),
        "method": "llm",
        "matched_product_type": r.get("product_type", ""),
        "ratio": round(best_ratio, 3),
        "cost": 0.0,  # classify_to_category는 비용을 반환하지 않음 → 분류 비용 미집계
    }


def recommend(product_name: str, selling_points: list, n: int = 5, client=None,
              threshold: float = SIMILARITY_THRESHOLD, prefilter_size: int = PREFILTER_SIZE,
              emphasis: str = "") -> tuple:
    """소구점 분석 결과 → 어울리는 원본 대본 추천.

    반환: (results, meta)
      results 항목: {rank, media_id, selling_point, product_type, coupang_mid, transcript,
                     match_score(0~100 또는 None), match_reason(한 줄 일치 이유)}
      meta: {pool_total, pool_after_exclude, excluded_mid, classify_method,
             matched_product_type, ratio, rank_cost, n_returned}
    """
    if client is None:
        import __main__
        client = __main__.client

    selling_points = [s for s in (selling_points or []) if s and s.strip()]
    emphasis = (emphasis or "").strip()
    query = " ".join(selling_points)

    df = load_candidates()
    pool_total = len(df)

    # 입력 제품의 중분류 판정 (하이브리드) — 같은 제품군 제외용
    cat = classify_input_category(product_name, selling_points, df, client, threshold=threshold)
    exclude_mid = cat["coupang_mid"]

    dff = df
    if exclude_mid:
        dff = df[df["coupang_mid"] != exclude_mid].reset_index(drop=True)
    pool_after_exclude = len(dff)

    meta = {
        "pool_total": pool_total,
        "pool_after_exclude": pool_after_exclude,
        "excluded_mid": exclude_mid,
        "classify_method": cat["method"],
        "matched_product_type": cat["matched_product_type"],
        "ratio": cat["ratio"],
        "rank_cost": 0.0,
        "n_returned": 0,
    }

    # 강조(emphasis)는 '원본 선택' 랭킹에만 최우선으로 얹는다.
    # 카테고리 제외 판정(위 classify_input_category)에는 넣지 않아 기존 분류 로직을 보존한다.
    # emphasis가 비어 있으면 rank_sps/rank_query == 기존 selling_points/query 라 동작이 완전히 같다.
    rank_sps = ([emphasis] + selling_points) if emphasis else selling_points
    rank_query = (emphasis + " " + query).strip() if emphasis else query

    if dff.empty or not rank_query:
        return [], meta

    # 소구점: 어휘 1차 거르개 → Haiku 의미 유사도 랭킹 (+ 일치율·이유)
    pool = prefilter(dff, rank_query, size=prefilter_size)
    picks, cost = rank_top_n_explained(dff, pool, rank_sps, n, client=client)
    meta["rank_cost"] = cost
    top_codes = [p["code"] for p in picks]
    explain = {p["code"]: p for p in picks}

    transcripts = resolve_transcripts(top_codes)
    results = []
    for rank_num, code in enumerate(top_codes, 1):
        row = dff[dff["code"] == code]
        if row.empty:
            continue
        ex = explain.get(str(code), {})
        results.append({
            "rank": rank_num,
            "media_id": code,
            "selling_point": str(row["selling_point"].iloc[0]),
            "product_type": str(row["product_type"].iloc[0]),
            "coupang_mid": str(row["coupang_mid"].iloc[0]),
            "transcript": transcripts.get(str(code), ""),
            "match_score": ex.get("score"),
            "match_reason": ex.get("reason", ""),
        })
    meta["n_returned"] = len(results)
    return results, meta


def load_candidates_with_devices() -> pd.DataFrame:
    """devices(기법 태그)가 채워진 후보 풀. 유형별 추천 전용.

    백엔드 분기:
      - sqlite: load_candidates() + hook_labels.devices를 media_id로 inner join (기존)
      - supabase: script_candidates에서 devices IS NOT NULL인 행만 한 번에 읽음
        (sync 시 hook_labels를 left join해 두었으므로 단일 쿼리)

    devices는 'shock,story'처럼 콤마로 여러 태그가 들어있는 CSV 문자열.
    반환 컬럼: code, selling_point, product_type, coupang_mid, coupang_major, devices
    """
    if _get_backend() == "supabase":
        conn = _pg_connect()
        try:
            df = pd.read_sql_query(
                f"SELECT code, selling_point, product_type, coupang_mid, coupang_major, devices "
                f"FROM {SUPABASE_TABLE} "
                f"WHERE selling_point IS NOT NULL AND selling_point != '' "
                f"  AND coupang_mid IS NOT NULL AND coupang_mid != '' "
                f"  AND transcript IS NOT NULL AND transcript != '' "
                f"  AND devices IS NOT NULL AND devices != ''",
                conn,
            )
        finally:
            conn.close()
        df["code"] = df["code"].astype(str)
        return df.reset_index(drop=True)

    # --- 로컬 SQLite (기존 inner join 그대로) ---
    df = load_candidates()
    conn = _connect_ro(HOOK_LABELS_DB)
    try:
        hk = pd.read_sql_query(
            "SELECT media_id, devices FROM hook_labels "
            "WHERE devices IS NOT NULL AND devices != ''",
            conn,
        )
    finally:
        conn.close()
    hk = hk.rename(columns={"media_id": "code"})
    hk["code"] = hk["code"].astype(str)  # load_candidates()와 dtype 일치(merge 안전성)
    return df.merge(hk, on="code", how="inner").reset_index(drop=True)


def _device_tags(devices_str) -> list:
    """'shock,story' → ['shock', 'story'] (공백·빈값 제거)."""
    return [t.strip() for t in str(devices_str or "").split(",") if t.strip()]


def _device_sort_key(device: str):
    """기법 표시·정렬 순서. DEVICE_ORDER에 없으면 뒤로."""
    return DEVICE_ORDER.index(device) if device in DEVICE_ORDER else len(DEVICE_ORDER)


def _lexical_top_codes(sub: pd.DataFrame, query: str, k: int) -> list:
    """어휘 유사도(difflib) 상위 k개 code (순서 보존). Haiku 랭커가 유효한 code를
    충분히 못 줄 때의 보충용. prefilter와 같은 점수, 단 정렬된 리스트로 반환."""
    scored = [
        (str(row["code"]), SequenceMatcher(None, query, str(row["selling_point"] or "")).ratio())
        for _, row in sub.iterrows()
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [code for code, _ in scored[:k]]


def recommend_by_device(product_name: str, selling_points: list, per_type: int = 2, client=None,
                        threshold: float = SIMILARITY_THRESHOLD, prefilter_size: int = PREFILTER_SIZE) -> tuple:
    """소구점 분석 결과 → 훅 기법(devices) 유형별 어울리는 원본 대본 추천.

    각 기법 태그별로 소구점 의미 유사도 top per_type개를 랭킹한다(한 대본이 여러
    기법에 후보로 들어가는 건 허용). 최종 출력에서는 같은 대본이 두 기법에 겹치면
    순위가 더 높은(rank가 작은) 쪽에만 남기고 다른 기법에서는 제거한다.
    같은 순위로 겹치면 DEVICE_ORDER에서 앞선 기법에 남긴다. 보충(backfill)은 안 한다.

    반환: (results_by_device, meta)
      results_by_device: list[{device, items}] (DEVICE_ORDER 순, 후보 있는 기법만)
        items: list[{rank, media_id, selling_point, product_type, coupang_mid, devices, transcript,
                     match_score(0~100 또는 None), match_reason(한 줄 일치 이유)}]
      meta: {pool_total, pool_after_exclude, excluded_mid, classify_method,
             matched_product_type, ratio, rank_cost, per_type,
             devices_covered, n_unique, n_dropped_dup}
    """
    if client is None:
        import __main__
        client = __main__.client

    selling_points = [s for s in (selling_points or []) if s and s.strip()]
    query = " ".join(selling_points)

    df = load_candidates_with_devices()
    pool_total = len(df)

    # 입력 제품의 중분류 판정(하이브리드) — 같은 제품군 제외용. 기존 recommend()와 동일.
    cat = classify_input_category(product_name, selling_points, df, client, threshold=threshold)
    exclude_mid = cat["coupang_mid"]

    dff = df
    if exclude_mid:
        dff = df[df["coupang_mid"] != exclude_mid].reset_index(drop=True)
    pool_after_exclude = len(dff)

    meta = {
        "pool_total": pool_total,
        "pool_after_exclude": pool_after_exclude,
        "excluded_mid": exclude_mid,
        "classify_method": cat["method"],
        "matched_product_type": cat["matched_product_type"],
        "ratio": cat["ratio"],
        "rank_cost": 0.0,
        "classify_cost": cat.get("cost", 0.0),
        "total_cost": cat.get("cost", 0.0),  # 랭킹 전이라 분류 비용만. 아래에서 랭킹 비용 합산.
        "per_type": per_type,
        "devices_covered": [],
        "n_unique": 0,
        "n_dropped_dup": 0,
    }

    if dff.empty or not query:
        return [], meta

    # 기법 태그별 후보 code 집합 (devices CSV 펼치기 = 중복 허용)
    device_to_codes = {}
    for _, row in dff.iterrows():
        for tag in _device_tags(row["devices"]):
            device_to_codes.setdefault(tag, set()).add(row["code"])

    # 기법별 소구점 의미 랭킹 (어휘 prefilter → Haiku top per_type)
    # 공용 rank_top_n이 가끔 후보에 없는 code(인덱스/환각)를 주거나 파싱 예외를 내므로,
    # 여기서 유효 code만 취하고(순서·중복 정리), 부족하면 어휘 유사도로 보충해 per_type를 채운다.
    explain_by_code = {}  # code -> {"score","reason"}  (기법 간 공유 — 같은 입력sp·후보sp면 점수 동일)
    rank_cost_total = 0.0
    per_device_ranked = {}  # device -> [code, ...] (rank 순)
    for device, codes in device_to_codes.items():
        sub = dff[dff["code"].isin(codes)].reset_index(drop=True)
        if sub.empty:
            continue
        valid = set(sub["code"].astype(str))
        try:
            pool = prefilter(sub, query, size=prefilter_size)
            picks, cost = rank_top_n_explained(sub, pool, selling_points, per_type, client=client)
            rank_cost_total += cost
        except Exception:
            picks = []  # 한 기법의 랭킹 실패가 전체를 죽이지 않게 격리 → 아래 어휘 보충으로 대체
        clean, seen = [], set()
        for p in picks:
            cs = str(p["code"])
            if cs in valid and cs not in seen:
                seen.add(cs)
                clean.append(cs)
                explain_by_code.setdefault(cs, {"score": p["score"], "reason": p["reason"]})
        if len(clean) < per_type:  # 유효 picks 부족 → 어휘 유사도로 보충
            for cs in _lexical_top_codes(sub, query, per_type * 3):
                if cs in valid and cs not in seen:
                    seen.add(cs)
                    clean.append(cs)
                    if cs not in explain_by_code:  # 보충 건은 difflib 점수 + 보충 표기
                        sp_val = str(sub[sub["code"].astype(str) == cs]["selling_point"].iloc[0] or "")
                        ratio = SequenceMatcher(None, query, sp_val).ratio()
                        explain_by_code[cs] = {"score": int(round(ratio * 100)), "reason": "어휘 유사도 기반 보충"}
                if len(clean) >= per_type:
                    break
        clean = clean[:per_type]
        if clean:
            per_device_ranked[device] = clean
    meta["rank_cost"] = rank_cost_total
    meta["total_cost"] = rank_cost_total + meta["classify_cost"]

    # 중복 제거: 같은 code가 여러 기법에 있으면 (rank 낮은=상위, 동률이면 DEVICE_ORDER 앞선) 한 곳만
    best_device = {}  # code -> chosen device
    appearances = {}  # code -> list[(device, rank0)]
    for device, codes in per_device_ranked.items():
        for rank0, code in enumerate(codes):
            appearances.setdefault(code, []).append((device, rank0))
    n_dropped = 0
    for code, apps in appearances.items():
        chosen = sorted(apps, key=lambda x: (x[1], _device_sort_key(x[0])))[0][0]
        best_device[code] = chosen
        n_dropped += len(apps) - 1
    meta["n_dropped_dup"] = n_dropped

    # 살아남은 code만 모아 transcript 일괄 조회
    kept_codes = list(best_device.keys())
    transcripts = resolve_transcripts(kept_codes)

    # 기법별 최종 리스트 구성 (DEVICE_ORDER 순, 중복 제거 후 rank 재부여)
    results_by_device = []
    for device in sorted(per_device_ranked.keys(), key=_device_sort_key):
        kept = [c for c in per_device_ranked[device] if best_device.get(c) == device]
        if not kept:
            continue
        items = []
        for rank_num, code in enumerate(kept, 1):
            row = dff[dff["code"] == code]
            if row.empty:
                continue
            ex = explain_by_code.get(str(code), {})
            items.append({
                "rank": rank_num,
                "media_id": code,
                "selling_point": str(row["selling_point"].iloc[0]),
                "product_type": str(row["product_type"].iloc[0]),
                "coupang_mid": str(row["coupang_mid"].iloc[0]),
                "devices": str(row["devices"].iloc[0]),
                "transcript": transcripts.get(str(code), ""),
                "match_score": ex.get("score"),
                "match_reason": ex.get("reason", ""),
            })
        if items:
            results_by_device.append({"device": device, "items": items})

    meta["devices_covered"] = [d["device"] for d in results_by_device]
    meta["n_unique"] = sum(len(d["items"]) for d in results_by_device)
    return results_by_device, meta
