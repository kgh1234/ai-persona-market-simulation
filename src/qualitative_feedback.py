"""AI 가상 소비자 정성 피드백 엔진 (STEP 4 MVP).

- TinyTroupe 미사용. Ollama /api/chat 직접 호출 + format(JSON Schema) + Pydantic 검증 (v2 엔진과 같은 방식).
- 정량 점수(구매의향 등)는 STEP 3C에서 구분력이 확인되지 않아 **요청하지 않는다**.
- 모든 호출은 캐시 없이 실제 추론하며, Ollama 응답의 created_at / 토큰 수를 함께 기록한다.
- 검사 결과는 '검토 필요' 표시용이며 LLM 응답 내용은 절대 수정하지 않는다.
  자동 검사는 의미적 정합성을 보장하지 않는다.
"""
from __future__ import annotations

import csv
import difflib
import io
import json
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import requests
from pydantic import BaseModel, Field, ValidationError, field_validator

import runtime_config as rc  # [공개본 변경] Ollama 주소·결과 폴더를 환경변수로 지정

ROOT = Path(__file__).resolve().parents[1]
PERSONA_FILE = ROOT / "data" / "personas_v2.json"
RUNS_DIR = rc.RUNS_DIR  # [공개본 변경] 원본: results/step4/runs
DISCLAIMER = "AI 생성 가상 의견 · 실측 미검증"
PROMPT_VERSION = "qf_v3"  # qf_v1: STEP 4, qf_v2: STEP 4.5 가격 표기 분리, qf_v3: STEP 4.6 특징 분리·문항 역할 명확화 (이전 버전 프롬프트 기록은 공개본에 미포함)
# checks_v1: STEP 4 첫 테스트, v2: 100mL당 가격 오인 검사, v3: STEP 4.5 입력에 없는 특징·강도 불일치·의견 모순 검사
CHECKS_VERSION = "checks_v5"  # v4: STEP 4.6 절 단위 판정 등, v5: STEP 6.1 과일명 강도 왜곡·건강 효과 단정/추론 구분
DEFAULT_HOST = rc.OLLAMA_HOST  # [공개본 변경] 원본: 고정값 http://127.0.0.1:11435 → OLLAMA_HOST (없으면 11434)
DEFAULT_MODEL = "qwen2.5:7b"
OPTIONS = {"temperature": 0.7, "num_predict": 768}

# 한자·가나는 출력 단계에서 차단(라틴 문자는 제품명 표기 때문에 허용하고 사후 검사)
TEXT_PATTERN = r"^[가-힣A-Za-z0-9 .,!?~%()·:/+\-'&]+$"


# ---------------------------------------------------------------------------
# 입력 / 출력 스키마
# ---------------------------------------------------------------------------
class ProductConcept(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    category: str = Field(min_length=1, max_length=30)
    volume_ml: int = Field(gt=0, le=5000)
    price_krw: int = Field(gt=0, le=100000)
    sugar_g: Optional[float] = Field(default=None, ge=0, le=200)  # None = 미기재
    features: List[str] = Field(default_factory=list, max_length=8)
    description: str = Field(default="", max_length=300)

    @field_validator("features")
    @classmethod
    def _strip(cls, v):
        return [x.strip() for x in v if x and x.strip()]

    @property
    def unit_price(self) -> int:
        return round(self.price_krw / self.volume_ml * 100)

    def sugar_text(self) -> str:
        if self.sugar_g is None:
            return "정보 없음"
        return "0g (무설탕)" if self.sugar_g == 0 else f"{self.sugar_g:g}g"




class QualitativeFeedback(BaseModel):
    """LLM 생성 부분 (필드 순서 = 질문 순서)."""

    first_impression: str = Field(min_length=2, max_length=200, pattern=TEXT_PATTERN)
    positive_points: List[str] = Field(min_length=1, max_length=3)
    concerns: List[str] = Field(min_length=1, max_length=3)
    key_purchase_factors: List[str] = Field(min_length=1, max_length=3)

    @field_validator("positive_points", "concerns", "key_purchase_factors")
    @classmethod
    def _items(cls, v):
        for s in v:
            if not re.fullmatch(TEXT_PATTERN, s) or not (1 <= len(s) <= 120):  # STEP 4.6: 2→1 (예: 구매 기준 "맛")
                raise ValueError(f"invalid item: {s!r}")
        return v


def _llm_schema() -> dict:
    """Ollama format 용 JSON Schema (배열 항목에도 문자 제약을 명시)."""
    item = {"type": "string", "minLength": 1, "maxLength": 120, "pattern": TEXT_PATTERN}
    arr = {"type": "array", "items": item, "minItems": 1, "maxItems": 3}
    return {"type": "object",
            "properties": {"first_impression": {"type": "string", "minLength": 2, "maxLength": 200, "pattern": TEXT_PATTERN},
                           "positive_points": arr, "concerns": arr, "key_purchase_factors": arr},
            "required": ["first_impression", "positive_points", "concerns", "key_purchase_factors"]}


# ---------------------------------------------------------------------------
# 프롬프트
# ---------------------------------------------------------------------------
SYSTEM_TEMPLATE = """지금부터 음료 신제품 소비자 인터뷰에 응답자로 참여합니다. 당신은 아래와 같은 가상의 한국 소비자입니다.

[응답자 정보]
- {age}세 {gender}, {occupation}, {residence} 거주
- 음료를 고를 때 중요하게 보는 순서: {criteria_ranked}
- 평소 구매 습관: {habits}

판매자나 마케터가 아니라, 이 제품을 사는 소비자 본인의 입장에서 1인칭 존댓말로 솔직하게 답하세요. 모든 답변은 한국어로 작성하세요."""

USER_TEMPLATE = """다음은 출시를 검토 중인 음료 콘셉트입니다.

[제품 정보 — 아래에 적힌 내용만 사실입니다]
- 제품명: {name}
- 종류: {category}
- 용량: 1개 {volume_ml}mL
- 판매가격: 1개 {price_krw:,}원
- 참고 환산가: 100mL당 {unit_price}원 (판매가격이 아니라 비교용 환산 값)
- 당류: {sugar_text}
- 특징 (각 항목은 서로 별개이며, '강한' 같은 꾸미는 말은 그 항목에만 해당합니다):
{features_block}
{description_line}
[답변 규칙]
- 제품 정보에 없는 내용(맛의 세기, 성분, 칼로리, 원재료, 포장 등)은 사실처럼 말하지 마세요. 필요하면 "~인지 궁금하다"처럼 질문으로 말하세요.
- 각 질문에는 그 질문에 맞는 내용만 쓰세요.

[인터뷰 질문]
Q1. 이 제품을 처음 본 느낌을 한 문장으로 말해 주세요. (first_impression)
Q2. 제품 정보 중 마음에 드는 점과 그 이유를 1~3개 말해 주세요. (positive_points)
Q3. 걱정되거나 아쉬운 점만 1~3개 말해 주세요. 좋은 점은 여기에 쓰지 마세요. (concerns)
Q4. 이런 음료를 살지 정할 때 당신이 확인하는 기준을 1~3개, 짧은 명사구로 말해 주세요. 이 제품에 대한 평가나 감상은 쓰지 마세요. (key_purchase_factors)"""


def load_personas() -> list[dict]:
    return json.loads(PERSONA_FILE.read_text(encoding="utf-8"))["personas"]


def build_messages(persona: dict, c: ProductConcept) -> list[dict]:
    system = SYSTEM_TEMPLATE.format(age=persona["age"], gender=persona["gender"], occupation=persona["occupation"],
                                    residence=persona["residence"],
                                    criteria_ranked=" > ".join(persona["criteria_ranked"]), habits=persona["habits"])
    user = USER_TEMPLATE.format(name=c.name, category=c.category, volume_ml=c.volume_ml, price_krw=c.price_krw,
                                unit_price=c.unit_price, sugar_text=c.sugar_text(),
                                features_block="\n".join(f"  · {f}" for f in c.features) if c.features else "  · 정보 없음",
                                description_line=f"- 설명: {c.description}\n" if c.description.strip() else "")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# 자동 검사 (표시용 — 내용은 수정하지 않음)
# ---------------------------------------------------------------------------
_HANGUL = re.compile(r"[가-힣]")
_LATIN = re.compile(r"[A-Za-z]+")
_THIRD_PERSON = re.compile(r"사람들에게|분들에게|분들은|소비자라면|소비자들|고객들|타겟층|시장에서|"
                           r"마케팅|잘 팔릴|팔릴지|판매량|출시 전략|시장 반응")  # v4: 판매자·마케터 관점 추가
_NUM_UNIT = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(원|mL|ml|ML|밀리리터|g|그램)")


def _all_text(fb: QualitativeFeedback) -> str:
    return " ".join([fb.first_impression, *fb.positive_points, *fb.concerns, *fb.key_purchase_factors])


def run_checks(fb: QualitativeFeedback, c: ProductConcept) -> dict:
    text = _all_text(fb)
    input_text = " ".join([c.name, c.category, *c.features, c.description])
    flags: list[str] = []

    # 1) 한국어 여부 (한자·가나는 스키마 단계에서 차단됨)
    letters = len(_HANGUL.findall(text)) + sum(len(w) for w in _LATIN.findall(text))
    hangul_ratio = len(_HANGUL.findall(text)) / letters if letters else 0.0
    foreign = sorted({w for w in _LATIN.findall(text)
                      if w.lower() not in input_text.lower() and w.lower() not in ("ml", "g", "kcal")})
    if hangul_ratio < 0.9:
        flags.append(f"[언어] 한국어 비율 낮음 ({hangul_ratio:.0%})")
    if foreign:
        flags.append(f"[언어] 입력에 없는 영문 표현: {foreign}")

    # 2) 입력 제품 정보와의 일치 (수치)
    allowed = {"원": {c.price_krw, c.unit_price}, "mL": {c.volume_ml, 100},
               "g": ({c.sugar_g} if c.sugar_g is not None else set())}
    for num, unit in _NUM_UNIT.findall(text):
        v = float(num.replace(",", ""))
        key = "원" if unit == "원" else ("mL" if unit.lower() in ("ml", "밀리리터") else "g")
        if key == "g" and c.sugar_g is None:
            flags.append(f"[입력 불일치] 당류 미기재 제품인데 {num}{unit} 언급")
        elif v not in allowed[key] and not (key == "g" and re.search(re.escape(num) + r"\s*g", input_text)):
            flags.append(f"[입력 불일치] 입력과 다른 수치 언급: {num}{unit}")

    # 3) 입력 제품 정보와의 일치 (당류 서술)
    if c.sugar_g is not None and c.sugar_g > 0 and re.search(r"무설탕|설탕이 없|당이 없|당류가 없|제로 슈거|당 0", text):
        flags.append("[입력 불일치] 당 포함 제품인데 '무설탕/당 없음'으로 서술")
    if c.sugar_g == 0 and re.search(r"당이 많|당류가 많|설탕이 많|당 함량이 높|당류 함량이 높", text):
        flags.append("[입력 불일치] 무설탕 제품인데 '당이 많다'로 서술")
    if c.sugar_g is None and re.search(r"무설탕|당이 없|당류가 없", text) and not re.search(r"무설탕|제로|zero", input_text, re.I):
        flags.append("[입력 불일치] 당류 정보가 없는데 무설탕으로 단정")

    # 4) 응답자 관점 이탈
    if _THIRD_PERSON.search(text):
        flags.append("[관점] 응답자 본인이 아닌 3인칭/일반론 서술 가능성")

    # 5) 모순·근거 부족 휴리스틱
    for p in fb.positive_points:
        for q in fb.concerns:
            if difflib.SequenceMatcher(None, p, q).ratio() > 0.8:
                flags.append(f"[모순] 좋은 점과 우려 사항이 거의 같음: '{p}' / '{q}'")
    if len(set(fb.positive_points)) < len(fb.positive_points) or len(set(fb.concerns)) < len(fb.concerns):
        flags.append("[형식] 항목 중복")
    short = [s for s in [*fb.positive_points, *fb.concerns] if len(_HANGUL.findall(s)) < 6]
    if short:
        flags.append(f"[근거 부족] 근거가 짧은 항목: {short}")

    # 6) (checks v2, STEP 4 수동 검토 후 추가) 100mL당 가격을 제품 가격처럼 서술
    unit_pat = re.compile(r"(?<![\d,])" + f"{c.unit_price:,}".replace(",", ",?") + r"\s*원")
    for m in unit_pat.finditer(text):
        if not re.search(r"100\s*(mL|ml|밀리리터)\s*당", text[max(0, m.start() - 15):m.start()]):
            flags.append(f"[입력 불일치] 100mL당 가격({c.unit_price}원)을 제품 가격처럼 서술했을 가능성")
            break

    # 6-1) (checks v3.1) 깨진 문자열: 같은 기호 반복, 괄호만 연속, 한 글자 뒤 기호로 끊긴 단어
    for s_ in [fb.first_impression, *fb.positive_points, *fb.concerns, *fb.key_purchase_factors]:
        if re.search(r"([()~!?,/+\-])\1{2,}|\.{4,}|[()]\s*[()]|[가-힣]\s+[()]{1,}\s+[가-힣]", s_):  # '...'(말줄임) 허용
            flags.append(f"[형식] 깨진 문자열(기호 반복·끊김) — \"{s_}\"")

    # 7~9) (checks v3, STEP 4.5) 입력에 없는 특징 / 특징 강도 불일치 / 의견 모순
    flags += unsupported_claim_flags(fb, c)
    flags += intensity_mismatch_flags(fb, c)
    flags += polarity_flags(fb)
    flags += role_flags(fb)
    h_flags, h_notes = health_claim_checks(fb)
    flags += h_flags

    return {"checks_version": CHECKS_VERSION, "hangul_ratio": round(hangul_ratio, 3), "flags": flags,
            "notes": h_notes,  # v5: 상태에 영향 없는 검토 메모
            "status": "검토 필요" if flags else "자동 검사 통과 (의미 정합성 미보장)"}


# ---------------------------------------------------------------------------
# checks v3 (STEP 4.5) — 규칙 기반이므로 표현이 다르면 놓치거나 오탐할 수 있음
# ---------------------------------------------------------------------------
# 음료 제품 정보에 흔히 덧붙여지는 속성어. 입력(제품명·종류·특징·설명)에 없는데 응답이 사실처럼 말하면 표시.
CLAIM_TERMS = {
    "과일 종류": r"사과|레몬|라임|자몽|복숭아|포도|딸기|오렌지|망고|파인애플|유자|청포도|수박|블루베리|체리|키위|매실|귤",
    "감미료": r"알룰로스|스테비아|에리스리톨|수크랄로스|아스파탐|감미료",
    "기능성 성분": r"비타민|전해질|단백질|콜라겐|유산균|프로바이오틱|면역|다이어트 효과|소화에 좋|피로 회복|카테킨|항산화",
    "카페인": r"카페인",
    "칼로리": r"칼로리|kcal|열량",
    "원재료 표현": r"천연|유기농|국산|무첨가|무색소|향료|첨가물|착즙|과즙",
    "탄산": r"탄산|톡 ?쏘",
    "청량감": r"청량",
    "용기·표시": r"(?<![가-힣])캔(?![가-힣])|페트|유리병|패키지|포장|뒷면|라벨",
    "판매 조건": r"1\+1|2\+1|할인|행사",
}
# 정보가 없음을 인정하거나 묻는 표현 → 사실 주장으로 보지 않음
_ASK = re.compile(r"궁금|모르|모르겠|확인|정보가 없|정보가 부족|정보도 없|알 수 없|알려|나와 있지|적혀 있지|"
                  r"는지|인지|을지|ㄹ지|일지|할지|날지|될지|있을지|까요|을까|ㄹ까|일까|있으면|었으면|였으면|면 좋|알고 싶")
# 추측 표현 → 사실 단정과 구분해 '추측'으로 표시
_GUESS = re.compile(r"수 있|것 같|듯|아마|지도 모")


def _claim_sentences(fb: QualitativeFeedback) -> list[tuple[str, str]]:
    """(필드, 문장). 구매 기준(Q4)은 '내가 볼 기준'이라 제품 사실 주장에서 제외."""
    out = []
    for field, items in (("첫인상", [fb.first_impression]), ("좋은 점", fb.positive_points), ("우려", fb.concerns)):
        for it in items:
            for s in re.split(r"(?<=[.!?])\s+", it.strip()):
                if s:
                    out.append((field, s))
    return out


# v4: 질문·추측 여부를 문장 전체가 아니라 절 단위로 판단 ("맛이 강해서 | 먹을 수 있을지 의심" → 앞 절은 단정)
_CLAUSE_SPLIT = re.compile(r"(?<=[가-힣])(?:서|고|지만|는데|니까|으니)\s+|,\s*")


def _clauses(sentence: str) -> list[str]:
    return [x for x in _CLAUSE_SPLIT.split(sentence) if x and x.strip()]


def unsupported_claim_flags(fb: QualitativeFeedback, c: ProductConcept) -> list[str]:
    input_text = " ".join([c.name, c.category, *c.features, c.description])
    flags = []
    for field, s in _claim_sentences(fb):
        for cl in _clauses(s):
            if _ASK.search(cl):
                continue
            for label, pat in CLAIM_TERMS.items():
                m = re.search(pat, cl)
                if m and not re.search(pat, input_text):
                    kind = "추측" if _GUESS.search(cl) else "단정"
                    flags.append(f"[입력 불일치] 입력에 없는 특징({label}: '{m.group(0)}')을 {kind} — {field}: \"{s}\"")
    return list(dict.fromkeys(flags))


# v4: 속성을 범주로 묶음 (복숭아 향/과일 향/향 → 향, 과일 맛/녹차 맛/맛 → 맛, 청량감/탄산 → 청량)
_ATTR_CORE = r"(단맛|향|맛|청량감|탄산)"
_ATTR_CAT = {"단맛": "단맛", "향": "향", "맛": "맛", "청량감": "청량", "탄산": "청량"}
_MODS = {"강": r"강(?:한|해|하|렬)", "진": r"진(?:한|해|하)", "풍부": r"풍부", "은은": r"은은", "약": r"약(?:한|해|하)",
         "부드럽": r"부드러|부드럽"}
_COND = re.compile(r"(?:강하|진하|약하|은은하|풍부하|부드러우|달)면|어떨지|(?:강|진|약)한지|(?:강|진|약)할지")
_ADNOMINAL = {"강": "강한", "진": "진한", "풍부": "풍부한", "은은": "은은한", "약": "약한", "부드럽": "부드러운"}


def _attr_mod_pairs(text: str) -> set[tuple[str, str]]:
    """(속성 범주, 강도) 쌍. 서술형('향이 강해')과 관형형('강한 (복숭아) 향')만 인정."""
    pairs = set()
    for mk, mp in _MODS.items():
        # '의향', '입맛' 같은 다른 단어의 일부는 제외 (앞 글자가 한글이면 제외)
        for m in re.finditer(rf"(?<![가-힣]){_ATTR_CORE}\s*(?:이|가|은|는|도)?\s*(?:매우|아주|너무|좀|조금|꽤)?\s*{mp}", text):
            pairs.add((_ATTR_CAT[m.group(1)], mk))
        # 관형형: '강한 청량감' 또는 '강한 복숭아 향'. 첫 어절이 조사로 끝나면('청량감은') 그 어절에서 판단을 끝냄
        for m in re.finditer(rf"{_ADNOMINAL[mk]}\s+([가-힣]+)(?:\s+([가-힣]+))?", text):
            cat = _core_of(m.group(1))
            if cat is None and m.group(2) and not _ends_with_josa(m.group(1)):
                cat = _core_of(m.group(2))
            if cat:
                pairs.add((cat, mk))
    return pairs


_JOSA = ("", "이", "가", "은", "는", "을", "를", "도", "과", "와", "의", "에", "이나", "까지", "만")


def _core_of(word: str):
    for core in ("단맛", "청량감", "탄산", "향", "맛"):
        if word.startswith(core) and word[len(core):] in _JOSA:
            return _ATTR_CAT[core]
    return None


def _ends_with_josa(word: str) -> bool:
    return len(word) > 1 and word[-1] in "이가은는을를도과와의에만"


def intensity_mismatch_flags(fb: QualitativeFeedback, c: ProductConcept) -> list[str]:
    """입력의 '강한 청량감'을 '복숭아 향이 강하다'처럼 다른 속성에 옮겨 붙인 경우 등."""
    input_pairs = set()
    for part in [*c.features, c.description, c.name]:  # 특징끼리 이어 붙여 생기는 가짜 인접 방지
        input_pairs |= _attr_mod_pairs(part)
    # v5: 특징에 쓰인 맛·향 이름(예: '복숭아 향'의 '복숭아')에 강도를 붙인 표현도 '향' 범주로 봄 ('복숭아가 강하니까')
    flavor_words = sorted({w for f_ in c.features for w in re.findall(r"[가-힣]{2,}", f_)
                           if w not in ("강한", "은은한", "청량감", "탄산", "진한", "약한") and _core_of(w) is None}, key=len, reverse=True)
    flavor_re = (rf"(?<![가-힣])(?:{'|'.join(map(re.escape, flavor_words))})\s*(?:이|가|은|는|도)?\s*(?:매우|아주|너무|좀|조금|꽤)?\s*"
                 rf"(?:{'|'.join(_MODS.values())})") if flavor_words else None
    flags = []
    for field, s in _claim_sentences(fb):
        for cl in _clauses(s):
            if _ASK.search(cl) or _COND.search(cl) or (_GUESS.search(cl) and field == "우려"):
                continue  # 질문·가정('너무 강하면…')·우려 속 추측 절은 제외
            pairs = set(_attr_mod_pairs(cl))
            if flavor_re:
                for m in re.finditer(flavor_re, cl):
                    mk = next(k for k, mp in _MODS.items() if re.search(mp, m.group(0)))
                    pairs.add(("향", mk))
            for attr, mod in pairs:
                if (attr, mod) not in input_pairs:
                    flags.append(f"[입력 불일치] 입력에 없는 특징 강도 서술('{attr}'+'{mod}') — {field}: \"{s}\"")
    return list(dict.fromkeys(flags))


# v5: 입력값(당류 0g 등)만으로 건강 효과를 말하는 표현. 단정형은 검토 필요, 추측·느낌 표현은 검토 메모(상태 불변)
_HEALTH = re.compile(r"건강에\s*(?:좋|유익|도움)|건강(?:하|한|해)|몸에\s*좋|다이어트에\s*(?:좋|도움)|살이\s*(?:안|덜)\s*찌|혈당|체중\s*관리")
_HEDGE = re.compile(r"것\s*같|거\s*같|듯|수\s*있|지\s*않을까|[을ㄹ]까|느껴|보여|보이|기대|궁금")


def health_claim_checks(fb: QualitativeFeedback) -> tuple[list[str], list[str]]:
    flags, notes = [], []
    for field, s in _claim_sentences(fb):
        for cl in _clauses(s):
            if not _HEALTH.search(cl) or _ASK.search(cl):
                continue
            if _HEDGE.search(cl):
                notes.append(f"[건강 효과 추론] 입력값만으로 건강 효과를 추측 — {field}: \"{s}\"")
            else:
                flags.append(f"[건강 효과 단정] 입력값만으로 건강 효과를 단정 — {field}: \"{s}\"")
            break
    return list(dict.fromkeys(flags)), list(dict.fromkeys(notes))


_POS_IDIOM = re.compile(r"부담(?:이|은)?\s*(?:없|적)|걱정(?:이)?\s*없|장벽이\s*낮|문제(?:가)?\s*없|손색(?:이)?\s*없")  # v4
_POS = re.compile(r"좋|마음에 들|맘에 들|기대|매력|끌리|괜찮|만족|반갑|훌륭|맛있|신선|합리적|저렴")
_NEG = re.compile(r"걱정|아쉽|아쉬|부담|비싸|비싼|떨어|우려|싫|별로|모르|궁금|확신|의문|않|없|안 |못|부족|"
                  r"높아|높게|높은|높다|적어|작아|신경 쓰|망설|어렵|힘들|불편|단점|문제|아니|의심|약하|애매|글쎄")
_PRICE_POS = re.compile(r"가성비[가이]?\s*(?:좋|괜찮|훌륭|뛰어)|저렴|합리적|싸서|싼 편|가격[이도]?\s*(?:적당|괜찮|착하|좋)")
_PRICE_NEG = re.compile(r"비싸|비싼|가격[이도]?\s*(?:좀|조금|다소|약간)?\s*(?:높|부담)|가격 부담")
_SUGAR_LOW = re.compile(r"당[이류분]?\s*(?:가|이)?\s*(?:적|낮|없)|무설탕|저당")
_SUGAR_HIGH = re.compile(r"당[이류분량]?\s*(?:가|이|함량이)?\s*(?:많|높)")  # v4: 단맛(달다)은 당류와 다를 수 있어 제외


# v4: 구매 기준(Q4)에 평가·감상이 들어간 경우, 다른 칸 내용을 반복한 경우 / 반말 의문형
# 조건·질문형('좋아야', '좋으면', '좋은지')은 기준 표현이므로 제외
_FACTOR_EVAL = re.compile(r"좋(?!아야|은지|을지|은가|으면|다면|는지|은 ?편인지)|싫|걱정|마음에 들|맛있(?!는지|을지|으면)|기대(?!되는지)|매력")
_BANMAL_Q = re.compile(r"[가-힣](?:니|냐|거야|거지)\s*\?")


def role_flags(fb: QualitativeFeedback) -> list[str]:
    flags = []
    for f in fb.key_purchase_factors:
        if _FACTOR_EVAL.search(f):
            flags.append(f"[역할 혼동] 구매 기준 항목에 제품 평가·감상이 들어감 — \"{f}\"")
        elif any(difflib.SequenceMatcher(None, f, o).ratio() > 0.6 for o in fb.positive_points):
            # 우려를 기준으로 옮기는 것('맛이 실제 과일 같아야')은 자연스러워 비교 대상에서 제외
            flags.append(f"[역할 혼동] 구매 기준 항목이 좋은 점 내용을 반복 — \"{f}\"")
    for s in [fb.first_impression, *fb.positive_points, *fb.concerns, *fb.key_purchase_factors]:
        if _BANMAL_Q.search(s):
            flags.append(f"[문체] 존댓말이 아닌 반말 의문형 — \"{s}\"")
    return flags


def polarity_flags(fb: QualitativeFeedback) -> list[str]:
    flags = []
    for s in fb.concerns:
        if _POS.search(s) and not _NEG.search(_POS_IDIOM.sub(" ", s)):
            flags.append(f"[모순] 우려 항목에 우려·부정 표현 없이 긍정 서술만 있음 — \"{s}\"")
    for s in fb.positive_points:
        has_pos = _POS.search(s) or _POS_IDIOM.search(s)
        if _NEG.search(_POS_IDIOM.sub(" ", s)) and not has_pos:
            flags.append(f"[모순] 좋은 점 항목에 긍정 표현 없이 부정 서술만 있음 — \"{s}\"")
    pos_text = " ".join([fb.first_impression, *fb.positive_points])
    neg_text = " ".join(fb.concerns)
    if _PRICE_POS.search(pos_text) and _PRICE_NEG.search(neg_text):
        flags.append("[모순] 가격을 좋은 점(가성비·저렴)과 우려(비쌈·부담)로 동시에 평가 — 양가적 의견일 수 있어 검토 필요")
    if _SUGAR_LOW.search(pos_text) and _SUGAR_HIGH.search(neg_text):
        flags.append("[모순] 당을 좋은 점(적음)과 우려(많음)로 동시에 평가")
    return flags


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
class FeedbackEngine:
    def __init__(self, host: str = DEFAULT_HOST, model: str = DEFAULT_MODEL, timeout: int = 300):
        self.host, self.model, self.timeout = host.rstrip("/"), model, timeout

    def health(self) -> dict:
        try:
            v = requests.get(f"{self.host}/api/version", timeout=5).json()
            tags = requests.get(f"{self.host}/api/tags", timeout=5).json()
            return {"ok": True, "version": v.get("version"),
                    "model_available": any(m["name"] == self.model for m in tags.get("models", []))}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": str(e)}

    def evaluate_one(self, persona: dict, c: ProductConcept, messages: Optional[list] = None) -> dict:
        """messages 를 주면 그 프롬프트를 사용 (STEP 5 A/B 모듈). 생략 시 기존 qf 프롬프트."""
        messages = messages or build_messages(persona, c)
        out = {"persona_id": persona["persona_id"], "persona_label": persona["label"], "ok": False,
               "feedback": None, "raw_content": None, "errors": [], "checks": None,
               "request": {"messages": messages, "options": OPTIONS, "model": self.model}}
        t0 = time.time()
        try:
            r = requests.post(f"{self.host}/api/chat", timeout=self.timeout, json={
                "model": self.model, "messages": messages, "stream": False, "format": _llm_schema(), "options": OPTIONS})
        except requests.RequestException as e:
            out["errors"].append(f"request error: {e}")
            out["checks"] = {"status": "생성 실패", "flags": out["errors"]}
            return out
        out["wall_s"] = round(time.time() - t0, 2)
        if r.status_code != 200:
            out["errors"].append(f"HTTP {r.status_code}: {r.text[:300]}")
            out["checks"] = {"status": "생성 실패", "flags": out["errors"]}
            return out
        d = r.json()
        out["ollama"] = {"created_at": d.get("created_at"), "eval_count": d.get("eval_count"),
                         "prompt_eval_count": d.get("prompt_eval_count"), "done_reason": d.get("done_reason"),
                         "total_duration_s": round(d.get("total_duration", 0) / 1e9, 2)}
        out["raw_content"] = d.get("message", {}).get("content")
        try:
            fb = QualitativeFeedback.model_validate_json(out["raw_content"])
        except ValidationError as e:
            out["errors"].append(f"schema: {e.errors(include_url=False)}"[:800])
            out["checks"] = {"status": STATUS_SCHEMA_FAIL, "flags": out["errors"], "checks_version": CHECKS_VERSION}
            return out
        out["ok"] = True
        out["feedback"] = fb.model_dump()
        out["checks"] = run_checks(fb, c)
        return out

    def run(self, c: ProductConcept, personas: list[dict], progress=None) -> dict:
        run = {"run_id": datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6],
               "created_at": datetime.now().isoformat(timespec="seconds"), "disclaimer": DISCLAIMER,
               "prompt_version": PROMPT_VERSION, "model": self.model, "options": OPTIONS,
               "concept": c.model_dump(), "results": []}
        for i, p in enumerate(personas):
            if progress:
                progress(i, len(personas), p)
            if p.get("profile"):  # STEP 6: 사용자 구조화 프로필 → 별도 프롬프트(qf_v3p), 기본 유형은 기존 그대로
                from persona_profiles import PROFILE_PROMPT_VERSION, PersonaProfile, build_profile_messages
                prof = PersonaProfile.model_validate(p["profile"])
                res = self.evaluate_one(p, c, messages=build_profile_messages(prof, c))
                res["profile"] = p["profile"]  # 평가 당시 속성·출처 스냅샷
                res["prompt_version"] = PROFILE_PROMPT_VERSION
            else:
                res = self.evaluate_one(p, c)
            run["results"].append(res)  # 페르소나마다 독립 호출 (대화 공유 없음)
        run["finished_at"] = datetime.now().isoformat(timespec="seconds")
        return run


# ---------------------------------------------------------------------------
# 저장 / 불러오기 / CSV
# ---------------------------------------------------------------------------
def save_run(run: dict, runs_dir: Path = RUNS_DIR) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{run['run_id']}.json"
    path.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def list_runs(runs_dir: Path = RUNS_DIR) -> list[dict]:
    if not runs_dir.exists():
        return []
    runs = []
    for p in sorted(runs_dir.glob("*.json"), reverse=True):
        try:
            runs.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            continue
    return runs


STATUS_GEN_FAIL = "생성 실패"                    # 요청·HTTP 오류 등으로 응답 원문이 없음
STATUS_SCHEMA_FAIL = "검토 필요 (스키마 실패)"     # 원문은 있으나 현재 스키마 검증 실패


def effective_view(run: dict, r: dict) -> dict:
    """화면·CSV에 쓸 결과. (v4) 저장 파일과 LLM 원문은 수정하지 않는다.
    - 원문이 없으면: 생성 실패
    - 원문이 있으면: 현재 스키마로 다시 검증 → 실패 시 '검토 필요 (스키마 실패)', 통과 시 현재 검사 기준으로 검사
    반환: {"status", "flags", "checks_version", "feedback"(dict|None), "rechecked_from", "stored_status", "note"}"""
    stored = r.get("checks") or {}
    raw = r.get("raw_content")
    if r.get("ok") and r.get("feedback"):
        fb_dict = r["feedback"]
        if stored.get("checks_version") == CHECKS_VERSION:
            return {**stored, "feedback": fb_dict}
        cur = run_checks(QualitativeFeedback.model_validate(fb_dict), ProductConcept(**run["concept"]))
        return {**cur, "feedback": fb_dict, "rechecked_from": stored.get("checks_version", "checks_v1"),
                "stored_status": stored.get("status")}
    if not raw:
        return {"status": STATUS_GEN_FAIL, "flags": stored.get("flags") or r.get("errors", []),
                "checks_version": CHECKS_VERSION, "feedback": None}
    try:
        fb = QualitativeFeedback.model_validate_json(raw)
    except (ValidationError, ValueError) as e:
        msg = f"schema: {e.errors(include_url=False)}"[:800] if isinstance(e, ValidationError) else f"json: {e}"[:800]
        return {"status": STATUS_SCHEMA_FAIL, "flags": [f"[스키마] {msg}"], "checks_version": CHECKS_VERSION,
                "feedback": None, "stored_status": stored.get("status")}
    cur = run_checks(fb, ProductConcept(**run["concept"]))
    return {**cur, "feedback": fb.model_dump(), "rechecked_from": stored.get("checks_version", "생성 당시 스키마"),
            "stored_status": stored.get("status"),
            "note": "생성 당시 스키마 검증에 실패했으나 현재 스키마로 원문을 다시 검증해 통과함 (원문 수정 없음)"}


def effective_checks(run: dict, r: dict) -> dict:
    """하위 호환용: effective_view 에서 검사 결과만 반환."""
    return {k: v for k, v in effective_view(run, r).items() if k != "feedback"}


CSV_COLUMNS = ["표시", "run_id", "생성시각", "제품명", "종류", "용량(mL)", "가격(원)", "당류(g)", "특징", "설명",
               "소비자 유형", "첫인상", "좋은 점", "우려 사항", "구매 시 고려 기준", "검토 상태", "검토 사유", "검사 기준", "모델"]


def run_to_rows(run: dict) -> list[dict]:
    c = run["concept"]
    rows = []
    for r in run["results"]:
        chk = effective_view(run, r)
        fb = chk.get("feedback") or {}
        rows.append({
            "표시": DISCLAIMER, "run_id": run["run_id"], "생성시각": run["created_at"], "제품명": c["name"],
            "종류": c["category"], "용량(mL)": c["volume_ml"], "가격(원)": c["price_krw"],
            "당류(g)": "미기재" if c["sugar_g"] is None else c["sugar_g"], "특징": ", ".join(c["features"]),
            "설명": c.get("description", ""), "소비자 유형": r["persona_label"],
            "첫인상": fb.get("first_impression", ""), "좋은 점": " / ".join(fb.get("positive_points", [])),
            "우려 사항": " / ".join(fb.get("concerns", [])),
            "구매 시 고려 기준": " / ".join(fb.get("key_purchase_factors", [])),
            "검토 상태": chk.get("status", ""),
            "검토 사유": " | ".join(chk.get("flags", [])),
            "검사 기준": (chk.get("checks_version") or "") + (f" (생성 당시 {chk['rechecked_from']}에서 재검사)"
                                                             if chk.get("rechecked_from") else "")
                       + (f" · {chk['note']}" if chk.get("note") else ""),
            "모델": run["model"],
        })
    return rows


def rows_to_csv_bytes(rows: list[dict]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS)
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode("utf-8-sig")  # 엑셀 한글 호환
