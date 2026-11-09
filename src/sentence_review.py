"""문장(항목) 단위 검토 (STEP 6.2).

- 검토 단위: 첫인상 1개 + 좋은 점 각 항목 + 우려 사항 각 항목 (구매 기준은 보고서 근거로 쓰지 않아 제외)
- 자동 표시: 같은 검사(run_checks)를 그 항목 하나에만 적용해 구분
    왜곡 의심 / 의미·형식 경고 / 건강 효과 단정 / 설명 부족(입력과 일치) / 설명 부족 / 경고 없음
  짧지만 입력값과 일치하는 표현('무설탕', '강한 청량감')은 사실 오류로 분류하지 않음
- 사람 검토: 승인 / 제외 / 검토 전 (기본값 '검토 전', 자동 승인 없음)
  '승인'은 AI가 만든 문장이 입력 정보와 정합한지 사람이 확인했다는 뜻이며, 실제 소비자 의견이 아님
- 저장: outputs/reviews/<run_id>.json (원본 실행 JSON은 수정하지 않음). 문장 원문을 함께 저장해,
  원문이 달라지면 해당 검토는 적용하지 않음
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import runtime_config as rc  # [공개본 변경]
from qualitative_feedback import ProductConcept, QualitativeFeedback, effective_view, run_checks

ROOT = Path(__file__).resolve().parents[1]
REVIEW_DIR = rc.REVIEW_DIR  # [공개본 변경] 원본: results/reviews. 테스트에서 바꿀 수 있도록 함수 안에서 매번 읽음
APPROVED, EXCLUDED, PENDING = "승인", "제외", "검토 전"
DECISIONS = [PENDING, APPROVED, EXCLUDED]
APPROVAL_MEANING = ("'승인'은 AI가 생성한 문장이 입력한 제품 정보와 맞는지 사람이 확인했다는 뜻입니다. "
                    "실제 소비자의 의견이나 검증된 소비자 인사이트가 아닙니다.")
FIELDS = [("first_impression", "첫인상"), ("positive_points", "좋은 점"), ("concerns", "우려 사항")]

AUTO_DISTORT, AUTO_WARN, AUTO_HEALTH = "왜곡 의심", "의미·형식 경고", "건강 효과 단정"
AUTO_THIN_OK, AUTO_THIN, AUTO_OK = "설명 부족 (입력과 일치)", "설명 부족", "경고 없음"


def _input_vocab(c: ProductConcept) -> set[str]:
    words = " ".join([c.name, c.category, *c.features, c.description, c.sugar_text(), "당류",
                      f"{c.price_krw:,}원", f"{c.price_krw}원", f"{c.volume_ml}mL", f"{c.volume_ml}ml"])
    vocab = set(re.findall(r"[가-힣A-Za-z0-9,]+", words))
    if c.sugar_g == 0:
        vocab |= {"무설탕", "0g", "당", "없음", "제로"}
    return vocab


def consistent_with_input(text: str, c: ProductConcept) -> bool:
    """짧은 표현의 모든 단어가 입력값에 있는지 (예: '무설탕', '강한 청량감', '0g 당류')."""
    toks = re.findall(r"[가-힣A-Za-z0-9,]+", text)
    return bool(toks) and all(t in _input_vocab(c) for t in toks)


def _unit_checks(field: str, text: str, c: ProductConcept) -> dict:
    kw = {"first_impression": "", "positive_points": [], "concerns": [], "key_purchase_factors": []}
    kw[field] = text if field == "first_impression" else [text]
    return run_checks(QualitativeFeedback.model_construct(**kw), c)


def auto_label(flags: list[str], text: str, c: ProductConcept) -> str:
    if any(f.startswith("[입력 불일치]") for f in flags):
        return AUTO_DISTORT
    if any(f.startswith("[건강 효과 단정]") for f in flags):
        return AUTO_HEALTH
    if any(not f.startswith("[근거 부족]") for f in flags):
        return AUTO_WARN
    if flags:  # 근거 부족만
        return AUTO_THIN_OK if consistent_with_input(text, c) else AUTO_THIN
    return AUTO_OK


def unit_id(i: int, field: str, j: int) -> str:
    return f"{i}|{field}|{j}"


def units(run: dict) -> list[dict]:
    """검토 단위 목록. 응답 단위 상태(effective_view)는 그대로 함께 기록."""
    c = ProductConcept(**run["concept"])
    out = []
    for i, r in enumerate(run["results"]):
        v = effective_view(run, r)
        fb = v.get("feedback")
        if not fb:
            continue
        for field, fname in FIELDS:
            items = [fb[field]] if field == "first_impression" else fb[field]
            for j, text in enumerate(items):
                chk = _unit_checks(field, text, c)
                out.append({"unit_id": unit_id(i, field, j), "persona_id": r["persona_id"], "persona_label": r["persona_label"],
                            "field": field, "field_label": fname, "text": text, "auto_flags": chk["flags"],
                            "auto_notes": chk.get("notes", []), "auto_label": auto_label(chk["flags"], text, c),
                            "response_status": v.get("status")})
    return out


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def review_path(run_id: str) -> Path:
    return REVIEW_DIR / f"{run_id}.json"


def load_review(run: dict) -> dict:
    """{unit_id: {decision, text, ...}}. 저장된 원문과 현재 문장이 다르면 그 항목은 적용하지 않음."""
    p = review_path(run["run_id"])
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    current = {u["unit_id"]: u["text"] for u in units(run)}
    return {uid: d for uid, d in data.get("items", {}).items() if current.get(uid) == d.get("text")}


def decision_of(review: dict, uid: str) -> str:
    return review.get(uid, {}).get("decision", PENDING)


def save_review(run: dict, decisions: dict, runs_dir: Path | None = None) -> Path:
    """decisions: {unit_id: '승인'|'제외'|'검토 전'}. 원본 실행 JSON은 읽기만 함."""
    us = {u["unit_id"]: u for u in units(run)}
    items = {}
    now = datetime.now().isoformat(timespec="seconds")
    for uid, dec in decisions.items():
        if uid not in us or dec not in DECISIONS:
            continue
        u = us[uid]
        items[uid] = {"decision": dec, "text": u["text"], "persona_label": u["persona_label"], "field": u["field"],
                      "auto_label": u["auto_label"], "reviewed_at": now}
    from qualitative_feedback import RUNS_DIR
    src = (runs_dir or RUNS_DIR) / f"{run['run_id']}.json"
    data = {"run_id": run["run_id"], "run_file_sha256": _sha(src), "updated_at": now, "meaning": APPROVAL_MEANING,
            "counts": {d: sum(x["decision"] == d for x in items.values()) for d in DECISIONS}, "items": items}
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    p = review_path(run["run_id"])
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def approved_units(run: dict, review: dict | None = None) -> list[dict]:
    review = load_review(run) if review is None else review
    return [u for u in units(run) if decision_of(review, u["unit_id"]) == APPROVED]
