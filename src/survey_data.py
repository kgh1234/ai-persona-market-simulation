"""실제 소비자 조사 CSV 연동 — 구조 설계·템플릿·형식 검증만 (STEP 6).

- 이번 단계에서 구현: 필드 정의, 빈 입력 템플릿, 업로드 CSV의 형식 검증(행 수·필수 필드·값 범위).
- 구현하지 않음(계획): AI 결과와의 비교, 보정, 저장, 통계 분석.
- 임의의 설문 결과를 만들지 않으며, 템플릿에는 헤더만 있음.
"""
from __future__ import annotations

import csv
import io
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

# (컬럼, 필수 여부, 설명)
FIELDS = [
    ("respondent_id", True, "응답자 ID (익명화, 개인정보 미포함)"),
    ("survey_id", False, "조사 차수 또는 조사명"),
    ("collected_at", False, "수집 일자 (예: 2026-12-01)"),
    ("age_band", False, "연령대 (자기 보고)"),
    ("gender", False, "성별 (자기 보고)"),
    ("purchase_situation", False, "주 구매 상황"),
    ("consumer_traits", False, "기타 소비자 특성 (예: 당류 섭취를 줄이려 함)"),
    ("product_id", True, "평가한 제품 ID (콘셉트와 연결)"),
    ("product_name", False, "평가한 제품명"),
    ("presentation_order", False, "제시 순서: single / A_first / B_first"),
    ("purchase_intent", False, "구매의향 1~5점"),
    ("price_acceptance", False, "가격 적절성 1~5점"),
    ("preferred_product", False, "두 제품 비교 시 선호 제품: A / B / none"),
    ("choice_reason", False, "선택 이유 또는 구매하고 싶은/망설여지는 이유 (자유 응답)"),
]
COLUMNS = [f[0] for f in FIELDS]
REQUIRED = [f[0] for f in FIELDS if f[1]]


class SurveyRow(BaseModel):
    respondent_id: str = Field(min_length=1)
    survey_id: Optional[str] = None
    collected_at: Optional[str] = None
    age_band: Optional[str] = None
    gender: Optional[str] = None
    purchase_situation: Optional[str] = None
    consumer_traits: Optional[str] = None
    product_id: str = Field(min_length=1)
    product_name: Optional[str] = None
    presentation_order: Optional[Literal["single", "A_first", "B_first"]] = None
    purchase_intent: Optional[int] = Field(default=None, ge=1, le=5)
    price_acceptance: Optional[int] = Field(default=None, ge=1, le=5)
    preferred_product: Optional[Literal["A", "B", "none"]] = None
    choice_reason: Optional[str] = None

    @field_validator("*", mode="before")
    @classmethod
    def _blank_to_none(cls, v):
        return None if isinstance(v, str) and not v.strip() else v

    def has_outcome(self) -> bool:
        return any(x is not None for x in (self.purchase_intent, self.price_acceptance, self.preferred_product,
                                           self.choice_reason))


def template_csv_bytes() -> bytes:
    """헤더만 있는 빈 템플릿 (엑셀 한글 호환 UTF-8 BOM)."""
    buf = io.StringIO()
    csv.writer(buf).writerow(COLUMNS)
    return buf.getvalue().encode("utf-8-sig")


def validate_csv(data: bytes) -> dict:
    """형식 검증만 수행. 데이터를 저장하거나 분석하지 않음."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return {"ok": False, "errors": ["UTF-8 인코딩이 아닙니다. CSV를 UTF-8로 저장해 주세요."], "n_rows": 0}
    reader = csv.DictReader(io.StringIO(text))
    cols = reader.fieldnames or []
    missing = [c for c in REQUIRED if c not in cols]
    extra = [c for c in cols if c not in COLUMNS]
    if missing:
        return {"ok": False, "errors": [f"필수 컬럼 없음: {missing}"], "n_rows": 0, "extra_columns": extra}
    errors, rows = [], []
    for i, raw in enumerate(reader, start=2):
        try:
            row = SurveyRow.model_validate({k: raw.get(k) for k in COLUMNS})
        except ValidationError as e:
            errors.append(f"{i}행: " + "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors(include_url=False)))
            continue
        if not row.has_outcome():
            errors.append(f"{i}행: 구매의향·가격 적절성·선호 제품·이유 중 하나 이상이 필요합니다")
            continue
        rows.append(row)
    ids = [(r.respondent_id, r.product_id) for r in rows]
    dup = len(ids) - len(set(ids))
    if dup:
        errors.append(f"같은 응답자·제품 조합이 중복된 행 {dup}개")
    return {"ok": not errors and bool(rows), "errors": errors[:20], "n_errors": len(errors), "n_rows": len(rows),
            "n_respondents": len({r.respondent_id for r in rows}),
            "products": sorted({r.product_id for r in rows}), "extra_columns": extra}
