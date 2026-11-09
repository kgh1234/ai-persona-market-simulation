"""Nemotron 레코드 → 기존 구조화 페르소나(src/persona_profiles.PersonaProfile) 변환 어댑터 (실험 모듈).

원칙
- 기존 모듈은 import 만 하고 수정하지 않는다. 이 모듈은 변환만 하고 파일을 저장하지 않는다
  (저장은 app.py 가 사용자가 '등록'을 누를 때 persona_profiles.save_user_profile 로 수행).
- 데이터에 있는 속성만 채운다: 연령대·성별·직업/생활(직업·지역·가구·주거). 음료 구매 빈도·가격 허용 범위·
  선호 음료·당류 고려·신제품 수용도 등은 Nemotron 에 없으므로 비워 둔다 (임의로 채우지 않음).
- 출처 표시: 데이터에서 온 속성은 '실제 조사 근거'가 아니므로 별도 출처 'NVIDIA 합성 데이터'(SRC_SYNTH)로 두고,
  사용자가 추가한 가정은 '가상 시나리오 가정'(SRC_ASSUME)으로 구분한다. evidence 에 Nemotron uuid·필드명과
  '공개 통계 분포 기반 합성 · 개인 실측 아님'을 적는다. (evidence 는 프롬프트로 전달되지 않음)
  [공개본 변경: 설명을 실제 코드 동작에 맞게 갱신, 코드 변경 없음]
- 서사(culinary_persona 등)는 gemma 모델이 생성한 문장이다. include_narrative=True 일 때만 '평소 구매 습관'
  자리에 넣고 출처를 'LLM 생성 서사'로 적는다. 기본값은 False.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from persona_profiles import SRC_ASSUME, SRC_SYNTH, Attr, PersonaProfile  # noqa: E402

from nemotron_source import ATTRIBUTION, DATASET_REVISION  # noqa: E402

SYNTH_NOTE = ("NVIDIA Nemotron-Personas-Korea 합성 페르소나를 형식 변환한 가상 응답자입니다. "
              "실제 소비자 응답이나 검증된 대표 표본이 아니며, 음료 관련 속성은 데이터에 없어 비워 두었습니다.")


def _age_band(age: int) -> str:  # 기존 AGE_BANDS 와 같은 표기
    return "10대 후반" if age < 20 else "50대 이상" if age >= 50 else f"{age // 10 * 10}대"


def _region(row: dict) -> str:
    d = str(row.get("district") or "")
    return d.replace("-", " ") if d else str(row.get("province") or "")


def to_profile(row: dict, text: dict | None = None, include_narrative: bool = False,
               user_assumptions: dict | None = None) -> PersonaProfile:
    """user_assumptions: {속성 키: 값} — 사용자가 추가한 가정. 출처 '가상 시나리오 가정'으로 따로 표시 (STEP 7)."""
    uid = (text or {}).get("uuid") or f"s{row['shard']}rg{row['row_group']}i{row['row_in_group']}"
    loc = f"shard {row['shard']} / row_group {row['row_group']} / row {row['row_in_group']}"
    ev = lambda fields: (f"Nemotron-Personas-Korea@{DATASET_REVISION[:7]} uuid={uid} ({loc}); 필드: {fields}; "  # noqa: E731
                         "공개 통계 분포 기반 합성 · 개인 실측 아님")
    sex = {"여자": "여성", "남자": "남성"}.get(row["sex"], "구분 없음")
    life = f"{row['occupation']}, {_region(row)} 거주, {row['family_type']}, {row['housing_type']}"
    attrs = {
        "age_band": Attr(value=_age_band(int(row["age"])), source=SRC_SYNTH, evidence=ev("age")),
        "gender": Attr(value=sex, source=SRC_SYNTH, evidence=ev("sex")),
        "lifestyle": Attr(value=f"{int(row['age'])}세, " + life, source=SRC_SYNTH,
                          evidence=ev("age, occupation, district, family_type, housing_type")),
    }
    if include_narrative and text and (text.get("culinary_persona") or "").strip():
        attrs["habits"] = Attr(value=text["culinary_persona"].strip(), source=SRC_SYNTH,
                               evidence=ev("culinary_persona") + " · LLM(gemma) 생성 서사")
    for k, v in (user_assumptions or {}).items():
        if k in attrs or not v or (isinstance(v, list) and not v):
            continue  # 데이터에서 온 속성은 덮어쓰지 않음
        attrs[k] = Attr(value=v, source=SRC_ASSUME, evidence="사용자가 추가한 가정 (Nemotron 데이터에 없는 속성)")
    occ = str(row["occupation"])
    occ = occ if len(occ) <= 12 else occ[:12] + "…"
    label = f"N·{int(row['age'])}세 {sex} {occ}"[:40]
    return PersonaProfile(profile_id=f"N_{str(uid)[:12]}", label=label, attrs=attrs, is_default=False,
                          created_at=f"Nemotron 선별 ({ATTRIBUTION[:40]}…)", note=SYNTH_NOTE)


def as_engine_persona(p: PersonaProfile) -> dict:
    """기존 persona_profiles.as_engine_persona 와 같은 형태 (profile 포함 → 엔진이 qf_v3p 경로 사용)."""
    import json
    return {"persona_id": p.profile_id, "label": p.label, "profile": json.loads(p.model_dump_json())}
