"""실행 환경 설정 (공개용 저장소에서 추가한 모듈).

환경변수
- OLLAMA_HOST    : Ollama 서버 주소. 예) 127.0.0.1:11434, localhost:11435, http://192.168.0.10:11434
                   지정하지 않으면 Ollama 기본값 http://127.0.0.1:11434 를 사용한다.
                   해석 규칙은 Ollama CLI와 같다: scheme이 없으면 기본 포트 11434,
                   http:// 로 시작하고 포트가 없으면 80, https:// 는 443.
- APP_OUTPUT_DIR : 실행 중 생성되는 파일(평가 기록, A/B 기록, 문장 검토 결과, 사용자 페르소나, 데이터 캐시)을
                   저장할 폴더. 지정하지 않으면 <프로젝트>/outputs. 상대경로는 프로젝트 폴더 기준.

소스코드·기본 데이터(data/)와 실행 결과(outputs/)를 분리해, 새 환경에서도 빈 폴더로 시작할 수 있게 한다.
"""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
OLLAMA_DEFAULT_PORT = 11434


def resolve_ollama_host(raw: str | None) -> str:
    """OLLAMA_HOST 값을 'http://host:port' 형태의 기본 URL로 바꾼다."""
    s = (raw or "").strip()
    if not s:
        return f"http://127.0.0.1:{OLLAMA_DEFAULT_PORT}"
    scheme, sep, rest = s.partition("://")
    if not sep:
        scheme, rest, default_port = "http", s, OLLAMA_DEFAULT_PORT
    else:
        scheme = scheme.lower()
        default_port = {"http": 80, "https": 443}.get(scheme, OLLAMA_DEFAULT_PORT)
    try:
        u = urlsplit(f"{scheme}://{rest}")
        port = u.port or default_port
    except ValueError as e:
        raise ValueError(f"OLLAMA_HOST 값을 해석할 수 없습니다: {raw!r}") from e
    host = u.hostname or "127.0.0.1"
    if ":" in host:  # IPv6
        host = f"[{host}]"
    return f"{scheme}://{host}:{port}{u.path.rstrip('/')}"


def _output_dir() -> Path:
    raw = os.environ.get("APP_OUTPUT_DIR", "").strip()
    p = Path(raw).expanduser() if raw else ROOT / "outputs"
    return p if p.is_absolute() else ROOT / p


OLLAMA_HOST = resolve_ollama_host(os.environ.get("OLLAMA_HOST"))

OUTPUT_DIR = _output_dir()
RUNS_DIR = OUTPUT_DIR / "runs"                     # 새 콘셉트 평가 기록
AB_RUNS_DIR = OUTPUT_DIR / "ab_runs"               # 목표 기반 A/B 비교 기록
REVIEW_DIR = OUTPUT_DIR / "reviews"                # 문장별 승인·제외 결과
USER_PERSONA_DIR = OUTPUT_DIR / "personas"         # 사용자가 만들거나 불러온 페르소나
NEMOTRON_CACHE_DIR = OUTPUT_DIR / "cache" / "nemotron"  # NVIDIA 합성 페르소나 인구통계 열 캐시
HF_HOME_DIR = OUTPUT_DIR / "cache" / "hf_home"     # huggingface_hub 설정·캐시 (홈 디렉터리에 쓰지 않음)


def ensure_output_dirs() -> None:
    for d in (RUNS_DIR, AB_RUNS_DIR, REVIEW_DIR, USER_PERSONA_DIR, NEMOTRON_CACHE_DIR, HF_HOME_DIR):
        d.mkdir(parents=True, exist_ok=True)


def display_path(p: Path | str) -> str:
    """화면 표시용 경로: 프로젝트 안이면 상대경로, 밖이면 그대로."""
    p = Path(p)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)
