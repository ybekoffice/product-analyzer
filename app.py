import os
import sys
import re
import json
import time
import random
import tempfile
import threading
from datetime import datetime
from urllib.parse import urlparse
import streamlit as st
from google import genai as google_genai
from google.genai import types as genai_types
from dotenv import load_dotenv
from PIL import Image
import io

BASE = os.path.dirname(os.path.abspath(__file__))
MAIN_PROCESS_DIR = os.path.abspath(os.path.join(BASE, '..'))  # 대본 추천 엔진 위치
# harness 공통 .env: 현재 위치는 main process/product-analyzer 이므로 3단계 위가 harness 루트.
# (예전 projects/product-analyzer 시절엔 2단계였음 — 폴더 이동 시 이 깊이 함께 조정)
# override=True: 셸(~/.zshrc)에 박힌 낡은 ANTHROPIC_API_KEY가 .env를 가리지 않도록 강제 덮어쓰기.
load_dotenv(os.path.join(BASE, '..', '..', '..', '.env'), override=True)
load_dotenv(os.path.join(BASE, '.env'), override=True)

GEMINI_MODEL = "gemini-2.5-flash"
# 2.5-flash가 과부하(503/overload)일 때 자동으로 갈아탈 폴백 모델 (2.0-flash는 폐기됨)
GEMINI_FALLBACK_MODEL = "gemini-2.5-flash-lite"
# 2.5 계열이 통째로 503 버스트일 때를 위한 다른 세대(별도 용량) 최후 폴백
GEMINI_FALLBACK_MODEL_2 = "gemini-3-flash-preview"
# 검색(grounding)은 느리고 과부하에 취약 → 빠르게 시도하되, 2.5-flash가 503/타임아웃이면
# 별도 용량 풀인 3-flash-preview로 한 번 더 grounding 시도(검색 살리기) 후 그래도 막히면 검색 없이 강등.
# (2.5-flash-lite는 2.5-flash와 같은 풀이라 버스트 때 같이 죽음 → 검색 체인에 넣지 않음)
SEARCH_CHAIN = [(GEMINI_MODEL, 1), (GEMINI_FALLBACK_MODEL_2, 1)]
# 검색 없는 분석(직접/강등)은 가볍고 안정적 → 서로 다른 모델을 넓게 시도(503 버스트 대응).
NOSEARCH_CHAIN = [(GEMINI_MODEL, 2), (GEMINI_FALLBACK_MODEL, 1), (GEMINI_FALLBACK_MODEL_2, 1)]
# 요청 최대 대기(ms, HttpOptions.timeout은 밀리초). 검색은 grounding 최대 ~70초 관측 → 90초.
# 검색 없는 요청은 보통 <10초라 60초면 충분. 초과 시 타임아웃 → 일시적 처리로 폴백/강등.
SEARCH_TIMEOUT_MS = 90000
NOSEARCH_TIMEOUT_MS = 60000
# 일시적 신호만 재시도/폴백. 인증·잘못된 요청 등은 즉시 실패시킨다.
# 'timed out'/'timeout'은 ReadTimeout(예: "The read operation timed out")까지 포괄.
TRANSIENT_MARKERS = ('503', '429', 'overload', 'unavailable', 'resource_exhausted',
                     'deadline', 'timeout', 'timed out')
# grounding(Gemini 웹검색)이 전부 막혔을 때의 폴백: Claude 웹검색으로 후기·개발의도·브랜드스토리 보강.
# Gemini grounding 풀과 무관한 별도 인프라라 버스트에도 안정적. 막힐 때만 호출돼 비용 최소(웹검색 1회당 약 $0.01).
CLAUDE_SEARCH_MODEL = "claude-haiku-4-5-20251001"
CLAUDE_SEARCH_MAX_USES = 5
WELCOME_MSG = "어떤 제품을 분석해드릴까요?\n영상, 사진 또는 제품 설명을 넣어주세요"

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}
VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv'}
MIME_MAP = {
    '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp',
    '.mp4': 'video/mp4', '.mov': 'video/quicktime', '.avi': 'video/x-msvideo', '.mkv': 'video/x-matroska',
}

_JSON_SHAPE = """{{
  "product_name": "제품 이름",
  "features": ["특징1", "특징2"],
  "selling_points": ["소구점1", "소구점2"],
  "target": "타겟 설명",
  "dev_intent": [{{"text": "개발자 의도", "source": "https://..."}}],
  "reviews": [{{"text": "실제 후기", "source": "https://..."}}],
  "brand_story": [{{"text": "브랜드 스토리", "source": "https://..."}}],
  "content_proposals": [{{"angle": "방향명", "idea": "구체적 영상 아이디어", "why": "활용할 소구점·스토리"}}]
}}"""

_KO_RULES = (
    "한국어 작성 규칙(필수):\n"
    "- 모든 출력은 자연스러운 한국어. 한자·중국어·영어 등 외국 문자를 그대로 노출 금지.\n"
    "- 제품명·브랜드명·회사명도 한글 음역으로 표기(예: 扬庆颜→양칭옌). 원어·한자 병기 금지.\n"
    "- 직역투·기계번역투 금지. 한국 쇼핑 콘텐츠에 어울리는 매끄러운 카피 톤으로.\n"
    "- 제조사·인증 같은 부수 정보에 외국 회사명/한자를 끼워 넣지 말 것.\n\n"
)

PROMPT_FILE_OFF = (
    "첨부된 파일(영상/사진)을 분석해서 아래 JSON 형식으로만 출력해줘. 반드시 JSON만, 마크다운 펜스 없이.{product_line}\n"
    "배경/부가 제품 무시, 주요 제품 집중.\n"
    "dev_intent·reviews·brand_story는 빈 배열 []. content_proposals는 소구점 활용 영상 방향 3가지.\n\n"
    + _KO_RULES
    + _JSON_SHAPE
)

PROMPT_FILE_ON = (
    "첨부된 파일(영상/사진)을 분석하고 웹 검색으로 추가 정보를 찾아서 아래 JSON 형식으로만 출력해줘. 반드시 JSON만, 마크다운 펜스 없이.{product_line}\n"
    "배경/부가 제품 무시, 주요 제품 집중.\n\n"
    "웹 섹션(dev_intent·reviews·brand_story) 규칙:\n"
    "- 웹 검색으로 실제 확인된 내용만 포함. 추측·일반론 금지.\n"
    "- 각 항목 source 필드에 검색 결과 출처 URL을 그대로 넣어라. URL 형식 때문에 항목을 빼지 말 것.\n"
    "- 내용 자체를 찾지 못한 섹션만 빈 배열 [].\n"
    "content_proposals는 소구점·스토리 활용 영상 방향 3가지 (출처 불필요).\n\n"
    + _KO_RULES
    + _JSON_SHAPE
)

PROMPT_TEXT_OFF = (
    "너는 쇼핑 제품 분석 도우미다.\n\n"
    "사용자 입력: {user_input}\n\n"
    "특정 제품(제품명·외형·기능·용도 중 하나라도)을 설명하면 → 아래 JSON 출력.\n"
    "아니면 → 텍스트로 2~3문장 친근하게 응대, 제품 정보 요청 (영상/사진 첨부 가능 안내).\n\n"
    "JSON 출력 시: 마크다운 펜스 없이. dev_intent·reviews·brand_story=[], content_proposals 3가지.\n\n"
    + _KO_RULES
    + _JSON_SHAPE
)

PROMPT_TEXT_ON = (
    "너는 쇼핑 제품 분석 도우미다.\n\n"
    "사용자 입력: {user_input}\n\n"
    "특정 제품을 설명하면 → 분석 + 웹 검색 후 아래 JSON 출력.\n"
    "아니면 → 텍스트로 2~3문장 친근하게 응대.\n\n"
    "JSON 출력 시: 마크다운 펜스 없이.\n"
    "웹 섹션: 실제 확인된 내용만(추측·일반론 금지). source 필드에 검색 출처 URL 그대로 기입. URL 형식 때문에 항목 제외 금지. 내용 못 찾은 섹션만 []. content_proposals 3가지.\n\n"
    + _KO_RULES
    + _JSON_SHAPE
)


@st.cache_resource
def get_gemini_client():
    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            pass
    return google_genai.Client(api_key=key)


def is_video(name):
    return os.path.splitext(name)[1].lower() in VIDEO_EXTS


def extract_frames(file_bytes, n=6):
    import cv2
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    cap = cv2.VideoCapture(tmp_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total > 0:
        for i in range(n):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * i / n))
            ret, frame = cap.read()
            if ret:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    os.unlink(tmp_path)
    return frames


def upload_media(uploaded_files):
    client = get_gemini_client()
    gemini_files = []
    for uf in uploaded_files:
        ext = os.path.splitext(uf.name)[1].lower() or '.mp4'
        mime = MIME_MAP.get(ext, 'video/mp4')
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uf.getvalue())
            tmp_path = tmp.name
        gf = client.files.upload(
            file=tmp_path,
            config=genai_types.UploadFileConfig(mime_type=mime, display_name=uf.name)
        )
        os.unlink(tmp_path)
        gemini_files.append(gf)
    for i, gf in enumerate(gemini_files):
        while True:
            gf = client.files.get(name=gf.name)
            gemini_files[i] = gf
            state = gf.state.name if hasattr(gf.state, 'name') else str(gf.state)
            if state == 'ACTIVE':
                break
            if state == 'FAILED':
                raise ValueError(f'파일 처리 실패: {gf.display_name}')
            time.sleep(2)
    return gemini_files


def parse_json_response(text):
    raw = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.MULTILINE)
    raw = re.sub(r'```\s*$', '', raw, flags=re.MULTILINE).strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and 'product_name' in data:
            return data
    except Exception:
        pass
    return None


def save_result(payload):
    results_dir = os.path.join(BASE, 'results')
    os.makedirs(results_dir, exist_ok=True)
    name = (payload.get('parsed') or {}).get('product_name') or 'unparsed'
    safe = re.sub(r'\W+', '_', name).strip('_')[:30] or 'result'
    ts = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    path = os.path.join(results_dir, f'{ts}_{safe}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return os.path.relpath(path, BASE)


_BLOCKED_DOMAINS = ('vertexaisearch.cloud.google.com',)

def validate_web_items(items):
    result = []
    for item in (items or []):
        if not isinstance(item, dict):
            continue
        if item.get('text'):
            result.append(item)
    return result


def _is_linkable(url):
    if not (isinstance(url, str) and url.startswith('http')):
        return False
    return not any(d in url for d in _BLOCKED_DOMAINS)


def domain_label(url):
    try:
        return urlparse(url).netloc.replace('www.', '') or url[:30]
    except Exception:
        return url[:30]


def _extract_grounded(resp):
    grounded = []
    try:
        meta = resp.candidates[0].grounding_metadata if resp.candidates else None
        if meta and meta.grounding_chunks:
            for chunk in meta.grounding_chunks:
                if chunk.web:
                    grounded.append({
                        'uri': chunk.web.uri or '',
                        'title': chunk.web.title or '',
                    })
    except Exception:
        pass
    return grounded


def _log_analysis_event(line):
    """진단용: stderr + 영구 파일(results/_error_log.txt)에 한 줄 기록.
    Streamlit 로그가 job 폴더 정리로 유실돼도 원인을 추적할 수 있게 안정 경로에 남긴다."""
    print(line, file=sys.stderr)
    try:
        log_dir = os.path.join(BASE, 'results')
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, '_error_log.txt'), 'a', encoding='utf-8') as f:
            f.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  {line}\n')
    except Exception:
        pass


def _generate_once(client, model, prompt, gemini_files, use_search, timeout_ms):
    """단일 Gemini 호출. 성공 시 (text, grounded), 실패 시 예외. 빈 응답도 예외로 처리한다."""
    if use_search:
        config = genai_types.GenerateContentConfig(
            tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
            http_options=genai_types.HttpOptions(timeout=timeout_ms),
        )
    else:
        config = genai_types.GenerateContentConfig(
            thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
            http_options=genai_types.HttpOptions(timeout=timeout_ms),
        )
    chat = client.chats.create(model=model)
    if gemini_files:
        parts = [
            genai_types.Part.from_uri(file_uri=f.uri, mime_type=f.mime_type)
            for f in gemini_files
        ]
        parts.append(prompt)
        resp = chat.send_message(parts, config=config)
    else:
        resp = chat.send_message(prompt, config=config)
    text = resp.text
    if not (text and text.strip()):
        raise ValueError('empty response (no text part)')
    return text, _extract_grounded(resp)


def _run_model_chain(client, prompt, gemini_files, use_search, chain, timeout_ms, result_container):
    """chain의 모델들을 순서대로 재시도.
       - 성공: (text, grounded, model) 튜플 반환
       - 비일시적 오류(인증·잘못된 요청 등): result_container['error'] 세팅 후 None 반환
       - 모두 일시적 실패(503·타임아웃 등)로 소진: 문자열 'EXHAUSTED' 반환"""
    for model, attempts in chain:
        for attempt in range(attempts):
            try:
                text, grounded = _generate_once(client, model, prompt, gemini_files, use_search, timeout_ms)
                return text, grounded, model
            except Exception as e:
                err = str(e)
                # 예외 타입명까지 포함해 판정 (예: ReadTimeout → 'timeout' 매칭)
                blob = f'{type(e).__name__}: {err}'.lower()
                is_empty = 'empty response' in err
                is_transient = is_empty or any(m in blob for m in TRANSIENT_MARKERS)
                _log_analysis_event(
                    f'[run_analysis] model={model} attempt={attempt + 1}/{attempts} '
                    f'검색={use_search} transient={is_transient} '
                    f'{type(e).__name__}: {err}'
                )
                if '404' in blob or 'not_found' in blob:
                    # 폴백 모델이 폐기/미가용 — 이 모델은 건너뛰고 다음 모델로.
                    break
                if not is_transient:
                    # 인증·잘못된 요청 등 — 재시도·폴백 무의미. 즉시 실패.
                    result_container['error'] = err
                    return None
                if attempt < attempts - 1:
                    base = min(3 * (2 ** attempt), 12)  # 3 → 6 → 12초 + 지터
                    time.sleep(base + random.uniform(0, 1.5))
    return 'EXHAUSTED'


def _claude_web_search(product_name, feature_hint=''):
    """Claude 웹검색 도구로 제품의 후기·개발의도·브랜드스토리를 조사한다.
       반환: ({dev_intent, reviews, brand_story} 항목들, grounded 출처 리스트) / 실패 시 None.
       Gemini grounding과 무관한 별도 인프라라 버스트에도 안정적."""
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not (product_name and key):
        return None
    try:
        import anthropic
    except Exception:
        return None
    instr = (
        "너는 제품 리서치 어시스턴트다. 아래 제품을 웹에서 검색해, 실제로 확인된 사실만으로 JSON을 만들어라.\n\n"
        f"제품: {product_name}\n"
        + (f"참고 특징: {feature_hint}\n" if feature_hint else "")
        + "\n채울 필드:\n"
        "- dev_intent: 브랜드/제조사가 밝힌 개발·기획 의도\n"
        "- reviews: 실제 사용자 후기\n"
        "- brand_story: 브랜드 스토리·역사·철학\n\n"
        "규칙:\n"
        '- 각 항목은 {"text": "자연스러운 한국어 1~2문장", "source": "그 내용을 확인한 실제 웹페이지 URL"} 형식.\n'
        "- source에는 검색으로 실제 확인한 출처 URL을 그대로 넣어라.\n"
        "- 추측·일반론·미확인 내용 금지. 확인된 게 없는 필드는 빈 배열 [].\n"
        "- 각 필드 최대 3개.\n"
        "- 다른 말 없이 JSON만 출력(마크다운 펜스 금지):\n"
        '{"dev_intent":[...],"reviews":[...],"brand_story":[...]}'
    )
    try:
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model=CLAUDE_SEARCH_MODEL,
            max_tokens=2000,
            tools=[{"type": "web_search_20250305", "name": "web_search",
                    "max_uses": CLAUDE_SEARCH_MAX_USES}],
            messages=[{"role": "user", "content": instr}],
        )
    except Exception as e:
        _log_analysis_event(f'[claude_search] 호출 실패: {type(e).__name__}: {str(e)[:150]}')
        return None

    texts, grounded = [], []
    verified_urls, verified_domains = set(), set()
    for block in resp.content:
        bt = getattr(block, 'type', None)
        if bt == 'text':
            texts.append(block.text or '')
            for c in (getattr(block, 'citations', None) or []):
                u = getattr(c, 'url', '') or ''
                if u:
                    verified_urls.add(u)
                    verified_domains.add(domain_label(u))
                    grounded.append({'uri': u, 'title': getattr(c, 'title', '') or ''})
        elif bt == 'web_search_tool_result':
            for r in (getattr(block, 'content', None) or []):
                u = getattr(r, 'url', '') or ''
                if u:
                    verified_urls.add(u)
                    verified_domains.add(domain_label(u))

    raw = re.sub(r'```(?:json)?', '', ''.join(texts)).strip()
    m = re.search(r'\{.*\}', raw, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except Exception:
        return None

    def _verify_items(items):
        out = []
        for it in (items or [])[:3]:
            if not isinstance(it, dict) or not it.get('text'):
                continue
            src = it.get('source', '') or ''
            # 실제 검색해 확인된 URL(또는 그 도메인)만 출처 링크로 인정. 아니면 링크 제거(텍스트는 유지).
            if not (src in verified_urls or (_is_linkable(src) and domain_label(src) in verified_domains)):
                src = ''
            out.append({'text': it['text'], 'source': src})
        return out

    fields = {k: _verify_items(data.get(k)) for k in ('dev_intent', 'reviews', 'brand_story')}
    if not any(fields.values()):
        return None
    return fields, grounded


def _enrich_with_claude_search(base_json_text):
    """검색 없이 만든 기본 분석 JSON(text)에 Claude 웹검색 결과를 병합.
       성공 시 (병합된 JSON 문자열, grounded) 반환, 불가 시 None."""
    parsed = parse_json_response(base_json_text)
    if not parsed or not parsed.get('product_name'):
        return None
    res = _claude_web_search(parsed.get('product_name', ''),
                             ', '.join(parsed.get('features') or [])[:200])
    if not res:
        return None
    fields, grounded = res
    parsed['dev_intent'] = fields['dev_intent']
    parsed['reviews'] = fields['reviews']
    parsed['brand_story'] = fields['brand_story']
    return json.dumps(parsed, ensure_ascii=False), grounded


def run_analysis(gemini_files, prompt, use_search, result_container, fallback_prompt=None):
    client = get_gemini_client()

    # 1) 검색 ON이면: grounding은 느리고 과부하에 취약하므로 한 번만 빠르게 시도.
    if use_search:
        out = _run_model_chain(client, prompt, gemini_files, True,
                               SEARCH_CHAIN, SEARCH_TIMEOUT_MS, result_container)
        if out is None:
            return  # 비일시적 오류 — error 이미 세팅됨
        if out != 'EXHAUSTED':
            text, grounded, _model = out
            result_container['response'] = text
            result_container['grounded'] = grounded
            return
        # grounding 전부 실패 → ① Gemini로 분석만(검색OFF, 안정 경로) ② Claude 웹검색으로 웹 보강.
        # 검색 없는 요청은 훨씬 가벼워 같은 순간에도 잘 통과한다.
        _log_analysis_event('[run_analysis] grounding 실패/지연 → 검색없이 분석 + Claude 웹검색 보강')
        deg_prompt = fallback_prompt or prompt
        out = _run_model_chain(client, deg_prompt, gemini_files, False,
                               NOSEARCH_CHAIN, NOSEARCH_TIMEOUT_MS, result_container)
        if out is None:
            return
        if out != 'EXHAUSTED':
            text, grounded, model = out
            # Claude 웹검색으로 dev_intent/reviews/brand_story 보강(별도 인프라 → 버스트 무관).
            enriched = _enrich_with_claude_search(text)
            if enriched is not None:
                merged_text, claude_grounded = enriched
                result_container['response'] = merged_text
                result_container['grounded'] = claude_grounded
                # search_skipped 미설정 → 웹검색이 됐으므로 리포트에 웹 섹션 정상 표시.
                _log_analysis_event(f'[run_analysis] 강등→Claude 웹검색 보강 성공 (분석:{model})')
                return
            # Claude 웹검색까지 실패 → 그때만 검색 생략 결과.
            result_container['response'] = text
            result_container['grounded'] = grounded
            result_container['search_skipped'] = True  # UI에서 "검색 생략" 안내용
            _log_analysis_event(f'[run_analysis] Claude 웹검색 보강 실패 → 검색 생략: {model}')
            return
    # 2) 검색 OFF(또는 강등 실패): 검색 없는 체인으로 폭넓게 시도.
    else:
        out = _run_model_chain(client, prompt, gemini_files, False,
                               NOSEARCH_CHAIN, NOSEARCH_TIMEOUT_MS, result_container)
        if out is None:
            return
        if out != 'EXHAUSTED':
            text, grounded, model = out
            result_container['response'] = text
            result_container['grounded'] = grounded
            if model != GEMINI_MODEL:
                _log_analysis_event(f'[run_analysis] 폴백 모델 사용: {model}')
            return

    # 3) 그래도 실패 → 혼잡 메시지
    result_container['error'] = (
        '⚠️ Gemini 서버가 혼잡합니다. 여러 번 재시도했지만 응답을 받지 못했어요. 잠시 후 다시 시도해주세요.'
    )


def generate_emphasis_proposals(data, emphasis):
    client = get_gemini_client()
    product_name = data.get('product_name', '')
    features = ', '.join(data.get('features') or [])
    selling_points = ', '.join(data.get('selling_points') or [])
    target = data.get('target', '')
    existing = '; '.join(
        p.get('angle', '') for p in (data.get('content_proposals') or []) if isinstance(p, dict)
    )
    prompt = (
        "너는 쇼핑 콘텐츠 대본 기획자다.\n\n"
        f"제품 분석 결과:\n"
        f"- 제품명: {product_name}\n"
        f"- 특징: {features}\n"
        f"- 소구점: {selling_points}\n"
        f"- 타겟: {target}\n"
        f"- 기존 제안 방향: {existing}\n\n"
        f"사용자 강조 포인트(최우선·최고 중요도로 반영 필수): \"{emphasis}\"\n\n"
        "위 강조 포인트를 가장 강하게 반영하고 기존 분석과 결합해서 "
        "새로운 영상 대본 방향 2~3개를 제안해라.\n"
        "반드시 JSON 배열만 출력. 마크다운 펜스 없이.\n"
        "[{\"angle\":\"방향명\",\"idea\":\"구체적 영상 아이디어\",\"why\":\"강조 포인트를 어떻게 활용했는지\"}]\n\n"
        + _KO_RULES
    )
    try:
        chat = client.chats.create(model=GEMINI_MODEL)
        config = genai_types.GenerateContentConfig(
            thinking_config=genai_types.ThinkingConfig(thinking_budget=0)
        )
        resp = chat.send_message(prompt, config=config)
        raw = re.sub(r'^```(?:json)?\s*', '', resp.text.strip(), flags=re.MULTILINE)
        raw = re.sub(r'```\s*$', '', raw, flags=re.MULTILINE).strip()
        result = json.loads(raw)
        if isinstance(result, list):
            return [p for p in result if isinstance(p, dict)]
    except Exception:
        pass
    return []


def _site_name(title):
    for sep in (' - ', ' | ', ' › ', ': ', ' : '):
        if sep in title:
            return title.split(sep)[0].strip()[:20]
    return title[:20]


def _web_items_html(items, use_search, grounded_lookup=None, fallback_site=''):
    lines = []
    for item in items:
        text = item.get('text', '')
        src = item.get('source', '')
        chip = ''
        if use_search and isinstance(src, str) and src.startswith('http'):
            if _is_linkable(src):
                lbl = domain_label(src)
            else:
                title = (grounded_lookup or {}).get(src, '')
                if title:
                    lbl = _site_name(title)
                elif fallback_site:
                    lbl = fallback_site
                else:
                    lbl = '출처'
            chip = f' <a href="{src}" target="_blank" class="source-chip">{lbl} ↗</a>'
        lines.append(f'<li>{text}{chip}</li>')
    return '\n'.join(lines)


def build_report_html(data, grounded, use_search, source_type, emphasis_proposals=None):
    product_name = data.get('product_name', '분석 결과')
    features = data.get('features') or []
    selling_points = data.get('selling_points') or []
    target = data.get('target', '')
    proposals = data.get('content_proposals') or []

    dev_intent = validate_web_items(data.get('dev_intent', [])) if use_search else []
    reviews = validate_web_items(data.get('reviews', [])) if use_search else []
    brand_story = validate_web_items(data.get('brand_story', [])) if use_search else []

    grounded_lookup = {g['uri']: g['title'] for g in (grounded or []) if g.get('uri')}
    site_names = {_site_name(g['title']) for g in (grounded or []) if g.get('title')}
    fallback_site = next(iter(site_names)) if len(site_names) == 1 else ''

    date_str = datetime.now().strftime('%Y.%m.%d')
    search_badge = '<span class="badge-search">🌐 웹 검색 포함</span>' if use_search else ''

    def card_section(items_html, title, icon, css_cls):
        if not items_html:
            return ''
        return f'''<div class="section {css_cls}">
  <h2 class="sec-title">{icon} {title}</h2>
  <ul class="item-list">{items_html}</ul>
</div>'''

    features_html = ''.join(f'<li>{f}</li>' for f in features)
    sp_html = ''.join(f'<li>{sp}</li>' for sp in selling_points)
    dev_html = _web_items_html(dev_intent, use_search, grounded_lookup, fallback_site)
    rev_html = _web_items_html(reviews, use_search, grounded_lookup, fallback_site)
    bs_html = _web_items_html(brand_story, use_search, grounded_lookup, fallback_site)

    target_section = ''
    if target:
        target_section = f'<div class="section s-target"><h2 class="sec-title">🎯 타겟</h2><p class="target-text">{target}</p></div>'

    proposals_html = ''
    for p in proposals:
        if not isinstance(p, dict):
            continue
        angle = p.get('angle', '')
        idea = p.get('idea', '')
        why = p.get('why', '')
        proposals_html += f'''<div class="proposal-card">
  <span class="angle-badge">{angle}</span>
  <p class="idea">{idea}</p>
  <p class="why">{why}</p>
</div>'''

    proposals_section = ''
    if proposals_html:
        proposals_section = f'''<div class="section s-proposals">
  <h2 class="sec-title">🎬 콘텐츠 제작 방향 제안</h2>
  <div class="proposals-grid">{proposals_html}</div>
</div>'''

    emphasis_html = ''
    for p in (emphasis_proposals or []):
        if not isinstance(p, dict):
            continue
        angle = p.get('angle', '')
        idea = p.get('idea', '')
        why = p.get('why', '')
        emphasis_html += f'''<div class="proposal-card">
  <span class="angle-badge">{angle}</span>
  <p class="idea">{idea}</p>
  <p class="why">{why}</p>
</div>'''

    emphasis_section = ''
    if emphasis_html:
        emphasis_section = f'''<div class="section s-emphasis">
  <h2 class="sec-title">✨ 강조 반영 대본 방향</h2>
  <div class="proposals-grid">{emphasis_html}</div>
</div>'''

    return f'''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#f1f5f9;font-family:"Apple SD Gothic Neo","Malgun Gothic","Noto Sans KR",sans-serif;color:#1e293b;padding:20px 0 60px}}
.report{{max-width:900px;margin:0 auto;padding:0 16px}}
.report-header{{background:linear-gradient(135deg,#1e3a8a 0%,#2563eb 100%);color:white;border-radius:16px;padding:28px 32px;margin-bottom:18px}}
.report-header h1{{font-size:24px;font-weight:700;margin-bottom:8px;line-height:1.3}}
.report-meta{{font-size:13px;opacity:.8;display:flex;gap:14px;align-items:center;flex-wrap:wrap}}
.badge-search{{background:rgba(255,255,255,.2);border-radius:20px;padding:2px 10px;font-size:12px}}
.section{{background:white;border-radius:14px;padding:22px 24px;margin-bottom:14px;box-shadow:0 1px 4px rgba(0,0,0,.06)}}
.sec-title{{font-size:15px;font-weight:700;margin-bottom:14px}}
.s-features .sec-title{{color:#1d4ed8}}
.s-sp .sec-title{{color:#047857}}
.s-target .sec-title{{color:#475569}}
.s-web .sec-title{{color:#b45309}}
.s-proposals .sec-title{{color:#6d28d9}}
.s-emphasis .sec-title{{color:#0f766e}}
.s-emphasis .proposal-card{{background:#f0fdfa;border:2px solid #0d9488}}
.s-emphasis .angle-badge{{background:#0f766e}}
.s-emphasis .why{{color:#0f766e}}
.item-list{{list-style:none;display:flex;flex-direction:column;gap:8px}}
.item-list li{{padding:10px 14px;border-radius:8px;font-size:14px;line-height:1.65}}
.s-features .item-list li{{background:#eff6ff}}
.s-sp .item-list li{{background:#f0fdf4;font-weight:500}}
.s-web .item-list li{{background:#fffbeb}}
.target-text{{font-size:14px;line-height:1.7;color:#334155}}
.source-chip{{display:inline-block;margin-left:8px;padding:1px 8px;background:#f8fafc;border:1px solid #cbd5e1;border-radius:20px;font-size:11px;color:#64748b;text-decoration:none;white-space:nowrap;vertical-align:middle}}
.source-chip:hover{{background:#e2e8f0;color:#1e293b}}
.proposals-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}}
.proposal-card{{background:#f5f3ff;border-radius:12px;padding:18px;border:1px solid #ede9fe}}
.angle-badge{{display:inline-block;background:#6d28d9;color:white;font-size:11px;font-weight:700;padding:3px 10px;border-radius:20px;margin-bottom:10px}}
.idea{{font-size:14px;font-weight:600;color:#1e293b;margin-bottom:7px;line-height:1.55}}
.why{{font-size:12px;color:#7c3aed;line-height:1.55}}
</style>
</head>
<body>
<div class="report">
  <div class="report-header">
    <h1>{product_name}</h1>
    <div class="report-meta">
      <span>📅 {date_str}</span>
      <span>📂 {source_type}</span>
      {search_badge}
    </div>
  </div>
  {card_section(features_html, '제품 특징', '🔍', 's-features')}
  {card_section(sp_html, '소구점', '💡', 's-sp')}
  {target_section}
  {card_section(dev_html, '개발·디자인 의도', '🛠️', 's-web')}
  {card_section(rev_html, '실제 사용자 후기', '💬', 's-web')}
  {card_section(bs_html, '브랜드 스토리', '📖', 's-web')}
  {proposals_section}
  {emphasis_section}
</div>
</body>
</html>'''


@st.cache_resource
def get_anthropic_client():
    import anthropic
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))


def run_recommendation(data, n=5):
    """소구점 분석 결과 → 어울리는 원본 대본 추천 (main_test.db 엔진 호출)."""
    if MAIN_PROCESS_DIR not in sys.path:
        sys.path.insert(0, MAIN_PROCESS_DIR)
    import recommend_engine
    client = get_anthropic_client()
    return recommend_engine.recommend(
        product_name=data.get('product_name', ''),
        selling_points=data.get('selling_points') or [],
        n=n, client=client,
    )


def render_recommendations(recos, meta):
    st.divider()
    if meta.get('error'):
        st.error(f"대본 추천 중 오류가 발생했습니다: {meta['error']}")
        return
    method_label = {'similar': '유사도 매칭', 'llm': 'LLM 분류'}.get(
        meta.get('classify_method'), meta.get('classify_method', ''))
    st.markdown(f"#### 🎬 어울리는 원본 대본 {meta.get('n_returned', len(recos))}개")
    st.caption(
        f"제외한 같은 카테고리: **{meta.get('excluded_mid') or '(판정 실패)'}**"
        f" · 카테고리 판정: {method_label}"
        f" · 후보 풀 {meta.get('pool_total', 0):,}개 중 추천"
    )
    if not recos:
        st.info("어울리는 대본을 찾지 못했습니다. 소구점을 더 구체적으로 입력하면 결과가 좋아집니다.")
        return
    for r in recos:
        with st.container(border=True):
            st.markdown(f"**{r['rank']}위**  ·  참고 분류: {r['coupang_mid']} / {r['product_type']}")
            st.markdown(f"💡 **이 대본의 소구점**: {r['selling_point']}")
            with st.expander("원본 대본 전문 보기"):
                st.write(r['transcript'] or '(대본 없음)')
            st.caption(f"media_id: {r['media_id']}")


# 훅 기법(devices) 태그 → 한글 라벨 (recommend_engine.DEVICE_ORDER와 같은 키)
DEVICE_LABELS = {
    'shock': '충격·반전',
    'demonstration': '시연·결과',
    'social_proof': '사회적 증거',
    'story': '스토리',
    'scarcity': '희소·긴박',
    'authority': '권위·전문성',
    'curiosity': '호기심',
    'question': '질문',
    'empathy': '공감',
    'direct_tip': '정보·팁',
}


def run_recommendation_by_device(data, per_type=2):
    """소구점 분석 결과 → 훅 기법(devices) 유형별 어울리는 원본 대본 추천."""
    if MAIN_PROCESS_DIR not in sys.path:
        sys.path.insert(0, MAIN_PROCESS_DIR)
    import recommend_engine
    client = get_anthropic_client()
    return recommend_engine.recommend_by_device(
        product_name=data.get('product_name', ''),
        selling_points=data.get('selling_points') or [],
        per_type=per_type, client=client,
    )


def render_recommendations_by_device(groups, meta):
    st.divider()
    if meta.get('error'):
        st.error(f"유형별 추천 중 오류가 발생했습니다: {meta['error']}")
        return
    method_label = {'similar': '유사도 매칭', 'llm': 'LLM 분류'}.get(
        meta.get('classify_method'), meta.get('classify_method', ''))
    st.markdown(f"#### 🎭 훅 유형별 어울리는 원본 대본 (유형당 최대 {meta.get('per_type', 2)}개)")
    st.caption(
        f"제외한 같은 카테고리: **{meta.get('excluded_mid') or '(판정 실패)'}**"
        f" · 카테고리 판정: {method_label}"
        f" · 후보 풀 {meta.get('pool_total', 0):,}개 중 추천"
        f" · 중복 제거 {meta.get('n_dropped_dup', 0)}건"
    )
    if not groups:
        st.info("어울리는 대본을 찾지 못했습니다. 소구점을 더 구체적으로 입력하면 결과가 좋아집니다.")
        return
    for g in groups:
        device = g['device']
        label = DEVICE_LABELS.get(device, device)
        st.markdown(f"##### 🏷️ {label}  `{device}`")
        for r in g['items']:
            with st.container(border=True):
                st.markdown(f"**{r['rank']}위**  ·  참고 분류: {r['coupang_mid']} / {r['product_type']}")
                st.markdown(f"💡 **이 대본의 소구점**: {r['selling_point']}")
                st.caption(f"훅 기법 태그: {r['devices']}")
                with st.expander("원본 대본 전문 보기"):
                    st.write(r['transcript'] or '(대본 없음)')
                st.caption(f"media_id: {r['media_id']}")


def render_report():
    data = st.session_state.report_data
    grounded = st.session_state.get('report_grounded', [])
    use_search = st.session_state.get('report_use_search', False)
    source_type = st.session_state.get('report_source_type', '분석')
    emphasis_proposals = st.session_state.get('report_emphasis', [])

    _, btn_col, _ = st.columns([1, 4, 1])
    with btn_col:
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button('← 새 분석', use_container_width=True):
                st.session_state.step = 'chat'
                st.session_state.messages = [{'role': 'assistant', 'content': WELCOME_MSG}]
                st.session_state.report_data = None
                st.session_state.report_grounded = []
                st.session_state.report_emphasis = []
                st.session_state.recommendations = None
                st.session_state.reco_meta = {}
                st.session_state.device_recommendations = None
                st.session_state.device_reco_meta = {}
                st.rerun()
        with c2:
            if st.button('📝 어울리는 대본 추천', use_container_width=True, type='primary'):
                with st.spinner('소구점에 어울리는 원본 대본을 찾는 중...'):
                    try:
                        recos, meta = run_recommendation(data)
                        st.session_state.recommendations = recos
                        st.session_state.reco_meta = meta
                    except Exception as ex:
                        st.session_state.recommendations = []
                        st.session_state.reco_meta = {'error': str(ex)}
                st.rerun()
        with c3:
            if st.button('🎭 유형별 추천', use_container_width=True):
                with st.spinner('훅 기법 유형별로 어울리는 원본 대본을 찾는 중...'):
                    try:
                        groups, meta = run_recommendation_by_device(data)
                        st.session_state.device_recommendations = groups
                        st.session_state.device_reco_meta = meta
                    except Exception as ex:
                        st.session_state.device_recommendations = []
                        st.session_state.device_reco_meta = {'error': str(ex)}
                st.rerun()

    if st.session_state.get('last_saved'):
        st.caption(f'💾 분석 기록 저장됨: {st.session_state.last_saved}')
    st.html(build_report_html(data, grounded, use_search, source_type, emphasis_proposals))

    if st.session_state.get('recommendations') is not None:
        render_recommendations(st.session_state.recommendations, st.session_state.get('reco_meta', {}))

    if st.session_state.get('device_recommendations') is not None:
        render_recommendations_by_device(
            st.session_state.device_recommendations, st.session_state.get('device_reco_meta', {}))

    _, form_col, _ = st.columns([1, 2, 1])
    with form_col:
        with st.form('emphasis_form', clear_on_submit=True):
            emphasis = st.text_area(
                '강조하고 싶은 소구점·타겟·메시지를 입력하세요',
                placeholder='예: 1인 가구 자취생 타겟 강조, 가성비 소구점 부각',
                height=100,
            )
            submitted = st.form_submit_button('✨ 제안대본 추가하기', use_container_width=True, type='primary')
        if submitted and emphasis.strip():
            with st.spinner('강조 내용 반영해서 새 대본 방향 생성 중...'):
                new_props = generate_emphasis_proposals(data, emphasis.strip())
            if new_props:
                st.session_state.report_emphasis.extend(new_props)
                st.rerun()
            else:
                st.error('생성에 실패했습니다. 다시 시도해주세요.')


def render_chat():
    _, main_col, _ = st.columns([1, 2, 1])

    with main_col:
        for msg in st.session_state.messages:
            with st.chat_message(msg['role'], avatar='🤖' if msg['role'] == 'assistant' else '👤'):
                if msg.get('frames'):
                    for item in msg['frames']:
                        icon = '📹' if item['type'] == 'video' else '🖼️'
                        st.caption(f"{icon} {item['name']}")
                        cols = st.columns(min(len(item['images']), 6))
                        for col, img in zip(cols, item['images']):
                            col.image(img, use_container_width=True)
                st.markdown(msg['content'])

        use_search = st.toggle('🌐 웹에서 후기·스토리까지 검색 (건당 약 $0.04 추가)', key='web_search', value=False)

    chat_input = st.chat_input(
        '영상·사진 첨부 또는 제품 설명을 입력하세요',
        accept_file='multiple',
        file_type=['mp4', 'mov', 'avi', 'mkv', 'jpg', 'jpeg', 'png', 'webp'],
    )

    if not chat_input:
        return

    user_input = chat_input.text or ''
    uploaded = chat_input.files or []
    use_search = st.session_state.get('web_search', False)

    if not user_input.strip() and not uploaded:
        return

    st.session_state.messages.append({'role': 'user', 'content': user_input})
    with main_col:
        with st.chat_message('user', avatar='👤'):
            st.markdown(user_input)

    gemini_files = []
    if uploaded:
        with st.spinner(f'파일 {len(uploaded)}개 업로드 중...'):
            try:
                gemini_files = upload_media(uploaded)
            except Exception as e:
                st.error(f'업로드 오류: {e}')
                return

    frame_data = []
    for uf in uploaded:
        if is_video(uf.name):
            try:
                frames = extract_frames(uf.getvalue(), n=6)
                if frames:
                    frame_data.append({'name': uf.name, 'images': frames, 'type': 'video'})
            except Exception:
                pass
        else:
            try:
                img = Image.open(io.BytesIO(uf.getvalue()))
                frame_data.append({'name': uf.name, 'images': [img], 'type': 'image'})
            except Exception:
                pass

    if gemini_files:
        product_line = f" 사용자 추가 설명: '{user_input}'" if user_input.strip() else ''
        prompt = (PROMPT_FILE_ON if use_search else PROMPT_FILE_OFF).format(product_line=product_line)
        # 검색이 과부하로 다 막혔을 때 쓰는 검색 없는(OFF) 강등 프롬프트
        fallback_prompt = PROMPT_FILE_OFF.format(product_line=product_line) if use_search else None
        has_vid = any(is_video(uf.name) for uf in uploaded)
        source_type = '영상' if has_vid and len(uploaded) == 1 else '사진' if not has_vid and len(uploaded) == 1 else '파일'
    else:
        prompt = (PROMPT_TEXT_ON if use_search else PROMPT_TEXT_OFF).format(user_input=user_input)
        fallback_prompt = PROMPT_TEXT_OFF.format(user_input=user_input) if use_search else None
        source_type = '텍스트 입력'

    result_container = {}
    thread = threading.Thread(
        target=run_analysis,
        args=(gemini_files, prompt, use_search, result_container, fallback_prompt)
    )
    thread.start()

    if gemini_files:
        with main_col:
            with st.chat_message('assistant', avatar='🤖'):
                with st.status('분석 중...', expanded=True) as status:
                    has_video = any(it.get('type') == 'video' for it in frame_data)
                    if frame_data:
                        st.write('🖼️ 첨부 파일 확인 중...')
                        for item in frame_data:
                            icon = '📹' if item.get('type') == 'video' else '🖼️'
                            st.caption(f"{icon} {item['name']}")
                            cols = st.columns(min(len(item['images']), 6))
                            for col, img in zip(cols, item['images']):
                                col.image(img, use_container_width=True)
                        time.sleep(0.6)
                    if has_video:
                        st.write('🎧 오디오 분석 중...')
                        time.sleep(0.4)
                    if use_search:
                        st.write('🌐 웹 검색 중...')
                    st.write('✍️ 소구점 및 제안 정리 중...')
                    thread.join()
                    status.update(label='분석 완료!', state='complete')
    else:
        with main_col:
            with st.chat_message('assistant', avatar='🤖'):
                with st.spinner(''):
                    thread.join()

    if 'error' in result_container:
        st.error(result_container['error'])
        return

    response_text = result_container.get('response', '')
    grounded = result_container.get('grounded', [])
    search_skipped = result_container.get('search_skipped', False)  # 검색 과부하로 검색 없이 분석됨
    parsed = parse_json_response(response_text)

    if gemini_files or parsed:
        saved_rel = save_result({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source_type': source_type,
            'use_search': use_search,
            'user_input': user_input,
            'uploaded_files': [uf.name for uf in uploaded],
            'prompt': prompt,
            'raw_response': response_text,
            'parsed': parsed,
            'parse_ok': parsed is not None,
            'grounded_sources': grounded,
        })
        st.session_state.last_saved = saved_rel

    if parsed:
        name = parsed.get('product_name', '제품')
        done_msg = f'✅ **{name}** 분석 완료! 제안서를 생성합니다...'
        if search_skipped:
            done_msg = (
                f'⚠️ 웹 검색 서버가 혼잡해서 **웹 후기·스토리 없이** 분석했어요. '
                f'(나중에 다시 시도하면 웹 정보까지 받을 수 있어요)\n\n' + done_msg
            )
        st.session_state.messages.append({
            'role': 'assistant',
            'content': done_msg,
            'frames': frame_data,
        })
        st.session_state.report_data = parsed
        st.session_state.report_grounded = grounded
        # 검색이 생략됐으면 리포트도 검색 없는 모드로 렌더(없는 웹 출처를 표시하지 않도록)
        st.session_state.report_use_search = use_search and not search_skipped
        st.session_state.report_source_type = source_type
        st.session_state.report_emphasis = []
        st.session_state.step = 'report'
        st.rerun()
    else:
        if not response_text.strip():
            response_text = '죄송합니다, 다시 시도해주세요.'
        st.session_state.messages.append({'role': 'assistant', 'content': response_text})
        st.rerun()


def main():
    st.set_page_config(page_title='소구점 분석기', page_icon='🔍', layout='wide')

    for key, default in [
        ('step', 'chat'),
        ('messages', [{'role': 'assistant', 'content': WELCOME_MSG}]),
        ('report_data', None),
        ('report_grounded', []),
        ('report_use_search', False),
        ('report_source_type', ''),
        ('report_emphasis', []),
        ('recommendations', None),
        ('reco_meta', {}),
        ('device_recommendations', None),
        ('device_reco_meta', {}),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    if st.session_state.step == 'report' and st.session_state.report_data:
        render_report()
    else:
        render_chat()


if __name__ == '__main__':
    main()
