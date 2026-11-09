"""구조화된 AI 페르소나 (STEP 6).

- 속성마다 '출처'를 함께 저장: 사용자 입력 / 가상 시나리오 가정 / 실제 조사 근거.
  '실제 조사 근거'는 근거(조사명·일자 등)를 반드시 적어야 하며, 현재 프로젝트에는 실제 조사 데이터가 없다.
- 기본 3유형(가격·건강·맛 중심형)은 기존 data/personas_v2.json 을 읽어 보여 주기만 한다(파일 수정 없음).
  기본 유형으로 평가할 때는 기존 프롬프트(qf_v3)를 그대로 쓴다 → 기존 기능 동작 불변.
- 사용자가 만든 프로필은 outputs/personas/<profile_id>.json 에 저장하고,
  평가 시 별도 프롬프트(qf_v3p)로 속성을 전달한다. qf_v3p 는 소규모 생성(응답 20건)으로만 확인했다.
  [공개본 변경: 설명을 현재 상태에 맞게 갱신]
- 생성한 페르소나는 가상 설정이며 특정 인구 집단을 대표하지 않는다.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

import runtime_config as rc  # [공개본 변경]

ROOT = Path(__file__).resolve().parents[1]
BASE_PERSONA_FILE = ROOT / "data" / "personas_v2.json"
USER_DIR = rc.USER_PERSONA_DIR  # [공개본 변경] 원본: data/persona_profiles_user. 테스트에서 바꿀 수 있도록 함수 안에서 매번 읽음
PROFILE_PROMPT_VERSION = "qf_v3p"
NOT_REPRESENTATIVE = "가상으로 설정한 소비자이며, 특정 연령·성별 집단의 대표 특성이 아닙니다."

SRC_USER, SRC_ASSUME, SRC_SURVEY = "사용자 입력", "가상 시나리오 가정", "실제 조사 근거"
SOURCES = [SRC_USER, SRC_ASSUME, SRC_SURVEY]  # 직접 만들기 화면에서 고를 수 있는 출처 (변경 없음)
# STEP 7: NVIDIA Nemotron-Personas-Korea 에서 가져온 인구통계 속성 전용 출처 (사용자가 고르는 값 아님).
# 공개 통계 분포 기반 합성 데이터이며 실제 조사 근거가 아님. 출처 표시는 프롬프트에 전달되지 않음.
SRC_SYNTH = "NVIDIA 합성 데이터"
ALL_SOURCES = SOURCES + [SRC_SYNTH]
SOURCE_ICON = {SRC_USER: "✍", SRC_ASSUME: "💭", SRC_SURVEY: "📊", SRC_SYNTH: "🧬"}

AGE_BANDS = ["10대 후반", "20대", "30대", "40대", "50대 이상"]
GENDERS = ["여성", "남성", "구분 없음"]
SITUATIONS = ["편의점에서 바로 마실 음료 구매", "식사와 함께", "운동 후", "공부·업무 중", "마트에서 여러 개 구매"]
FREQUENCY = ["거의 매일", "주 3~4회", "주 1~2회", "한 달에 몇 번", "드물게"]
DRINK_TYPES = ["탄산음료", "제로 탄산음료", "차 음료", "커피", "이온음료", "주스", "생수", "기타"]
LEVELS = ["낮음", "보통", "높음"]

# (키, 화면 라벨, 프롬프트 줄 라벨, 입력 형식, 선택지)
ATTR_SPEC = [
    ("age_band", "연령대", "연령대", "select", AGE_BANDS),
    ("gender", "성별", "성별", "select", GENDERS),
    ("lifestyle", "직업 또는 생활 패턴", "직업·생활 패턴", "text", None),
    ("situations", "구매 상황", "주로 음료를 사는 상황", "multi", SITUATIONS),
    ("frequency", "음료 구매 빈도", "음료를 사는 빈도", "select", FREQUENCY),
    ("price_range", "가격 허용 범위", "음료 한 개에 낼 수 있다고 생각하는 가격", "text", None),
    ("preferred_types", "선호 음료 유형", "즐겨 마시는 음료 종류", "multi", DRINK_TYPES),
    ("flavor_pref", "맛·향 선호", "맛·향 취향", "text", None),
    ("sugar_level", "당류 고려 수준", "음료를 고를 때 당류를 따지는 정도", "select", LEVELS),
    ("novelty", "신제품 수용도", "새로운 음료를 시도하는 정도", "select", LEVELS),
    ("key_criteria", "주요 구매 기준 (중요한 순서, 쉼표로 구분)", "음료를 고를 때 중요하게 보는 순서", "list", None),
    ("hesitations", "구매를 망설이는 요인 (쉼표로 구분)", "음료 구매를 망설이게 하는 요인", "list", None),
    ("habits", "평소 구매 습관", "평소 구매 습관", "text", None),
]
ATTR_KEYS = [a[0] for a in ATTR_SPEC]
ATTR_LABEL = {a[0]: a[1] for a in ATTR_SPEC}


class Attr(BaseModel):
    value: Union[str, List[str]]
    source: Literal["사용자 입력", "가상 시나리오 가정", "실제 조사 근거", "NVIDIA 합성 데이터"]
    evidence: str = ""  # '실제 조사 근거'일 때 필수 (조사명·일자·표본 등)

    @model_validator(mode="after")
    def _check(self):
        v = self.value
        if (isinstance(v, str) and not v.strip()) or (isinstance(v, list) and not [x for x in v if x.strip()]):
            raise ValueError("값이 비어 있습니다")
        if self.source == SRC_SURVEY and not self.evidence.strip():
            raise ValueError("'실제 조사 근거'로 표시하려면 근거(조사명·일자 등)를 입력해야 합니다")
        return self

    def text(self, ordered: bool = False) -> str:
        """목록은 쉼표로, 순서가 있는 목록(주요 구매 기준)만 ' > '로 표시."""
        if isinstance(self.value, list):
            return (" > " if ordered else ", ").join(self.value)
        return self.value


class PersonaProfile(BaseModel):
    profile_id: str
    label: str = Field(min_length=1, max_length=40)
    base_type: Optional[str] = None  # 'P_PRICE' 등 기본 유형에서 시작한 경우
    attrs: Dict[str, Attr] = Field(default_factory=dict)
    is_default: bool = False
    created_at: str = ""
    note: str = NOT_REPRESENTATIVE

    @model_validator(mode="after")
    def _check(self):
        unknown = set(self.attrs) - set(ATTR_KEYS)
        if unknown:
            raise ValueError(f"알 수 없는 속성: {sorted(unknown)}")
        if not self.attrs:
            raise ValueError("속성을 하나 이상 입력해야 합니다")
        return self

    def source_counts(self) -> dict:
        out = {s: sum(a.source == s for a in self.attrs.values()) for s in SOURCES}
        n_syn = sum(a.source == SRC_SYNTH for a in self.attrs.values())
        if n_syn:  # 합성 데이터 속성이 있을 때만 표시 (기존 페르소나 표시는 그대로)
            out[SRC_SYNTH] = n_syn
        return out

    def is_synthetic(self) -> bool:
        return any(a.source == SRC_SYNTH for a in self.attrs.values())


def _age_band(age: int) -> str:
    return "10대 후반" if age < 20 else "50대 이상" if age >= 50 else f"{age // 10 * 10}대"


def default_profiles() -> list[PersonaProfile]:
    """기존 personas_v2.json 의 3유형을 구조화해 보여 줌 (읽기 전용). 모두 STEP 3C에서 설계한 가상 가정."""
    data = json.loads(BASE_PERSONA_FILE.read_text(encoding="utf-8"))["personas"]
    out = []
    for p in data:
        A = lambda v: Attr(value=v, source=SRC_ASSUME)  # noqa: E731
        out.append(PersonaProfile(
            profile_id=p["persona_id"], label=p["label"], base_type=p["persona_id"], is_default=True,
            attrs={"age_band": A(_age_band(p["age"])), "gender": A(p["gender"]),
                   "lifestyle": A(f"{p['occupation']}, {p['residence']} 거주"),
                   "key_criteria": A(list(p["criteria_ranked"])), "habits": A(p["habits"])},
            created_at="STEP 3C (data/personas_v2.json)"))
    return out


def user_profiles() -> list[PersonaProfile]:
    d = USER_DIR
    if not d.exists():
        return []
    out = []
    for f in sorted(d.glob("*.json")):
        try:
            out.append(PersonaProfile.model_validate_json(f.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            continue
    return out


def all_profiles() -> list[PersonaProfile]:
    return default_profiles() + user_profiles()


def new_profile_id() -> str:
    return "U_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:4]


def save_user_profile(p: PersonaProfile) -> Path:
    if p.is_default:
        raise ValueError("기본 유형은 저장 대상이 아닙니다")
    USER_DIR.mkdir(parents=True, exist_ok=True)
    path = USER_DIR / f"{p.profile_id}.json"
    path.write_text(p.model_dump_json(indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 프롬프트 (사용자 프로필 전용, qf_v3p) — 응답자에게는 값만 전달하고 출처 표시는 보내지 않음
# ---------------------------------------------------------------------------
PROFILE_SYSTEM = """지금부터 음료 신제품 소비자 인터뷰에 응답자로 참여합니다. 당신은 아래와 같은 가상의 한국 소비자입니다.

[응답자 정보]
{lines}

연령대·성별은 응답 배경일 뿐입니다. 이것만으로 특정 맛이나 제품을 좋아하거나 싫어한다고 가정하지 말고, 위의 정보와 제품 정보를 바탕으로 판단하세요.
판매자나 마케터가 아니라, 이 제품을 사는 소비자 본인의 입장에서 1인칭 존댓말로 솔직하게 답하세요. 모든 답변은 한국어로 작성하세요."""


def profile_lines(p: PersonaProfile) -> list[str]:
    lines = []
    for key, _label, prompt_label, _kind, _opts in ATTR_SPEC:
        if key in p.attrs:
            a = p.attrs[key]
            val = a.text(ordered=(key == "key_criteria"))
            lines.append(f"- {prompt_label}: {val}")
    return lines


def build_profile_messages(p: PersonaProfile, concept) -> list[dict]:
    """사용자 프로필 + 기존 qf_v3 사용자 메시지(제품 정보·질문)를 그대로 사용."""
    from qualitative_feedback import build_messages  # 순환 import 방지
    user_msg = build_messages({"age": 0, "gender": "", "occupation": "", "residence": "", "criteria_ranked": [],
                               "habits": ""}, concept)[1]
    return [{"role": "system", "content": PROFILE_SYSTEM.format(lines="\n".join(profile_lines(p)))}, user_msg]


def as_engine_persona(p: PersonaProfile) -> dict:
    """FeedbackEngine 에 넘길 dict. 기본 유형은 기존 persona dict(기존 프롬프트), 사용자 프로필은 profile 포함."""
    if p.is_default:
        base = {x["persona_id"]: x for x in json.loads(BASE_PERSONA_FILE.read_text(encoding="utf-8"))["personas"]}
        return base[p.profile_id]
    return {"persona_id": p.profile_id, "label": p.label, "profile": json.loads(p.model_dump_json())}


def parse_list(text: str) -> list[str]:
    return [x.strip() for x in re.split(r"[,\n]", text or "") if x.strip()]
