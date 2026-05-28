"""Streamlit Cloud 배포용 번들 빌더.

목적: product-analyzer 저장소(=클라우드 배포 단위) 안에 harness 의 의존 코드/프롬프트를
복사해 두어, harness 경로 없이도 앱이 동작하게 한다. **단일 소스 원칙 유지**: 손편집 금지,
이 스크립트가 항상 harness 원본에서 다시 복사한다.

출력 구조:
  product-analyzer/
    _bundle/
      shared/
        __init__.py
        recommend.py
        ranking.py
        classify.py
        regenerate.py
        regenerate_models.py
        llm_models.py
      prompts/
        classify_caption_v3.md
        classify_coupang_v5.md
      recommend_engine.py

배포 후 클라우드에서는 app.py 부트스트랩이 harness 경로가 없음을 감지하고
`_bundle/` 을 sys.path 에 추가 → `from shared.X` / `import recommend_engine` 그대로 동작.

사용:
  python3 scripts/build_deploy_bundle.py           # 빌드
  python3 scripts/build_deploy_bundle.py --check   # 누락만 확인 (CI/푸시 전 검증용)
"""
import argparse
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent              # product-analyzer/scripts
PROJECT_DIR = SCRIPT_DIR.parent                            # product-analyzer
MAIN_PROCESS_DIR = PROJECT_DIR.parent                      # main process
HARNESS = MAIN_PROCESS_DIR.parent.parent                   # harness (= projects/../..)

BUNDLE_DIR = PROJECT_DIR / "_bundle"

# 어떤 파일이 어디로 가는지: (소스 절대 경로, 번들 상대 경로)
COPIES = [
    # shared 모듈 (recommend_engine + regenerate가 실제로 쓰는 것만)
    (HARNESS / "shared" / "__init__.py",          "shared/__init__.py"),
    (HARNESS / "shared" / "recommend.py",         "shared/recommend.py"),
    (HARNESS / "shared" / "ranking.py",           "shared/ranking.py"),
    (HARNESS / "shared" / "classify.py",          "shared/classify.py"),
    (HARNESS / "shared" / "regenerate.py",        "shared/regenerate.py"),
    (HARNESS / "shared" / "regenerate_models.py", "shared/regenerate_models.py"),
    (HARNESS / "shared" / "llm_models.py",        "shared/llm_models.py"),
    # 분류 프롬프트 (classify_to_category 가 로드)
    (HARNESS / "prompts" / "classify_caption_v3.md", "prompts/classify_caption_v3.md"),
    (HARNESS / "prompts" / "classify_coupang_v5.md", "prompts/classify_coupang_v5.md"),
    # 추천 엔진 본체 (Supabase/SQLite 백엔드 분기 내장)
    (MAIN_PROCESS_DIR / "recommend_engine.py", "recommend_engine.py"),
]


def verify_sources():
    missing = [src for src, _ in COPIES if not src.exists()]
    if missing:
        for m in missing:
            print(f"  ❌ 소스 누락: {m}", file=sys.stderr)
        sys.exit(2)


def do_build():
    verify_sources()
    # 기존 _bundle 초기화 (스테일 파일 방지)
    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    BUNDLE_DIR.mkdir()

    for src, rel in COPIES:
        dst = BUNDLE_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  {src.relative_to(HARNESS)} → _bundle/{rel}")

    # 번들이 손편집 대상이 아님을 알리는 표지
    (BUNDLE_DIR / "README.txt").write_text(
        "이 폴더는 scripts/build_deploy_bundle.py 가 자동 생성합니다.\n"
        "직접 편집하지 마세요. harness 원본을 수정한 뒤 빌드 스크립트를 다시 실행하세요.\n",
        encoding="utf-8",
    )
    print(f"\n완료: {len(COPIES)}개 파일 → {BUNDLE_DIR.relative_to(PROJECT_DIR.parent.parent.parent)}")


def do_check():
    verify_sources()
    missing = []
    for _, rel in COPIES:
        if not (BUNDLE_DIR / rel).exists():
            missing.append(rel)
    if missing:
        for m in missing:
            print(f"  ❌ 번들 누락: _bundle/{m}", file=sys.stderr)
        print("  → scripts/build_deploy_bundle.py 를 실행해 다시 빌드하세요.", file=sys.stderr)
        sys.exit(1)
    print(f"  ✅ 번들 {len(COPIES)}개 파일 모두 존재")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="빌드 없이 누락만 확인")
    args = ap.parse_args()
    if args.check:
        do_check()
    else:
        do_build()


if __name__ == "__main__":
    main()
