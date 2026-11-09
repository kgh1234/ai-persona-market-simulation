"""목표 기반 AI 페르소나 A/B 콘셉트 비교 (STEP 5).

- 정성 피드백 엔진(qualitative_feedback)과 같은 모델·스키마·자동 검사(checks)를 사용한다.
- 각 제품은 **독립적으로** 평가한다: LLM 응답자에게는 제품 하나와 응답자 설정만 보여 주고,
  기획 질문·비교 대상·실험 목적은 알려주지 않는다. A와 B에 같은 응답자 설정과 같은 질문을 쓴다.
- 구매의향 점수·구매율·제품 우열 판정은 만들지 않는다.
- 분석(공통 관심사·차이)과 가설은 '자동 검사 통과' 응답만 근거로 쓰고, 규칙 기반으로 도출한다(추가 LLM 호출 없음).
- 타깃(연령·성별·구매 상황)은 가상 설정이며, 그것만으로 선호를 가정하지 않도록 프롬프트에서 명시하고 응답을 검사한다.
"""
from __future__ import annotations

import csv
import io
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field

from qualitative_feedback import (DISCLAIMER, OPTIONS, ProductConcept, QualitativeFeedback, STATUS_GEN_FAIL,
                                  STATUS_SCHEMA_FAIL, effective_view)

import runtime_config as rc  # noqa: E402  [공개본 변경]

ROOT = Path(__file__).resolve().parents[1]
AB_RUNS_DIR = rc.AB_RUNS_DIR  # [공개본 변경] 원본: results/step5/ab_runs
AB_PROMPT_VERSION = "ab_v2"  # ab_v1 + 공통 타깃 추가 특성 줄 (STEP 5.2). 이전 버전 프롬프트 기록은 공개본에 미포함
TARGET_NOTE = "가상 설정 · 실제 조사 근거 없음"

AGE_BANDS = ["10대 후반", "20대", "30대", "40대", "50대 이상"]
GENDERS = ["여성", "남성", "구분 없음"]
SITUATIONS = ["편의점에서 바로 마실 음료 구매", "식사와 함께", "운동 후", "공부·업무 중", "마트에서 여러 개 구매"]
ASPECT_CHOICES = ["맛·향", "가격", "당류·건강", "청량감", "용량", "구매 상황 적합성"]
# STEP 5.2: 조건 기반 판단 기준 (상품기획자가 명시적으로 선택. 자유 텍스트에서 자동 추정하지 않음)
CRITERION_NONE = "지정 안 함"
CRITERIA = [CRITERION_NONE, "당류가 낮은 제품 우선", "판매가격이 낮은 제품 우선", "100mL당 환산가가 낮은 제품 우선"]


class Goal(BaseModel):
    question: str = Field(min_length=1, max_length=300)
    aspects: List[str] = Field(min_length=1)
    criterion: str = CRITERION_NONE  # STEP 5.2


class Target(BaseModel):
    age_band: str
    gender: str
    situations: List[str] = Field(default_factory=list)
    persona_ids: List[str] = Field(min_length=1)  # 소비 성향 = 가격·건강·맛 중심형
    traits: str = Field(default="", max_length=100)  # STEP 5.2: 공통 추가 특성 (A/B 양쪽에 동일하게 전달)

    def describe(self) -> str:
        g = "" if self.gender == "구분 없음" else f" {self.gender}"
        return f"{self.age_band}{g}"


# ---------------------------------------------------------------------------
# 프롬프트 ab_v1
# ---------------------------------------------------------------------------
AB_SYSTEM = """지금부터 음료 신제품 소비자 인터뷰에 응답자로 참여합니다. 당신은 아래 조건의 가상 한국 소비자입니다.

[응답자 설정]
- 연령대·성별: {demo}
- 주로 음료를 사는 상황: {situations}
{traits_line}- 음료를 고를 때 중요하게 보는 순서: {criteria_ranked}
- 평소 구매 습관: {habits}

연령대·성별은 응답 배경일 뿐입니다. 이것만으로 특정 맛이나 제품을 좋아하거나 싫어한다고 가정하지 말고, 위의 구매 상황·특성·구매 기준과 제품 정보를 바탕으로 판단하세요.
이 소비자 본인의 입장에서 1인칭 존댓말로 솔직하게 답하세요. 모든 답변은 한국어로 작성하세요."""

AB_USER = """다음은 출시를 검토 중인 음료 콘셉트입니다.

[제품 정보]
- 제품명: {name}
- 종류: {category}
- 용량: 1개 {volume_ml}mL
- 판매가격: 1개 {price_krw:,}원
- 참고 환산가: 100mL당 {unit_price}원 (판매가격이 아니라 비교용 환산 값)
- 당류: {sugar_text}
- 특징: {features}
{description_line}
위 제품 정보에 적힌 내용만 사실로 보고, 적혀 있지 않은 점은 모른다고 생각하고 답하세요.

인터뷰 질문에 답해 주세요.
Q1. 이 제품을 처음 본 느낌을 한 문장으로 말해 주세요. (first_impression)
Q2. 좋게 느껴지는 점을 1~3개 말해 주세요. (positive_points)
Q3. 걱정되거나 아쉬운 점을 1~3개 말해 주세요. (concerns)
Q4. 이런 음료를 살지 정할 때 당신이 확인하는 기준을 1~3개, 짧은 명사구로 말해 주세요. (key_purchase_factors)"""


def build_ab_messages(persona: dict, target: Target, c: ProductConcept) -> list[dict]:
    system = AB_SYSTEM.format(demo=target.describe(),
                              situations=", ".join(target.situations) if target.situations else "특정하지 않음",
                              traits_line=f"- 추가 특성: {target.traits.strip()}\n" if target.traits.strip() else "",
                              criteria_ranked=" > ".join(persona["criteria_ranked"]), habits=persona["habits"])
    user = AB_USER.format(name=c.name, category=c.category, volume_ml=c.volume_ml, price_krw=c.price_krw,
                          unit_price=c.unit_price, sugar_text=c.sugar_text(),
                          features=", ".join(c.features) if c.features else "정보 없음",
                          description_line=f"- 설명: {c.description}\n" if c.description.strip() else "")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# 공통 조건 / 차이
# ---------------------------------------------------------------------------
FIELD_LABELS = [("name", "제품명"), ("category", "종류"), ("volume_ml", "용량(mL)"), ("price_krw", "판매가격(원)"),
                ("sugar_g", "당류(g)"), ("features", "특징"), ("description", "설명")]


def concept_diff(a: ProductConcept, b: ProductConcept) -> list[dict]:
    rows = []
    for key, label in FIELD_LABELS:
        va, vb = getattr(a, key), getattr(b, key)
        fmt = lambda v: ("미기재" if v is None else ", ".join(v) if isinstance(v, list) else (f"{v:g}" if isinstance(v, float) else str(v)))
        rows.append({"항목": label, "제품 A": fmt(va) or "-", "제품 B": fmt(vb) or "-",
                     "비교": "동일" if va == vb else "차이"})
    return rows


# ---------------------------------------------------------------------------
# 추가 검사: 인구통계 고정관념 (A/B 전용, 기존 checks 와 별도)
# ---------------------------------------------------------------------------
_STEREO = re.compile(r"(?:\d0대|이십대|삼십대|젊은|나이|여자|남자|여성|남성)(?:이|가|라서|이라서|이니까|라면|들은|들이|들에게|에게는|답게)")


def stereotype_flags(fb: QualitativeFeedback) -> list[str]:
    out = []
    for s in [fb.first_impression, *fb.positive_points, *fb.concerns, *fb.key_purchase_factors]:
        if _STEREO.search(s):
            out.append(f"[관점] 인구통계(연령·성별)를 근거로 한 판단 가능성 — \"{s}\"")
    return out


def ab_view(concept: dict, r: dict) -> dict:
    """effective_view(현재 스키마 재검증 + checks) + 고정관념 검사. 원문·파일 무수정."""
    v = effective_view({"concept": concept}, r)
    if v.get("feedback"):
        extra = stereotype_flags(QualitativeFeedback.model_validate(v["feedback"]))
        if extra:
            v = {**v, "flags": [*v.get("flags", []), *extra], "status": "검토 필요"}
    return v


def is_usable(v: dict) -> bool:
    """분석 근거로 쓸 수 있는 응답: 자동 검사 통과만."""
    return bool(v.get("feedback")) and not v.get("status", "").startswith("검토 필요") and v.get("status") != STATUS_GEN_FAIL


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def run_ab(engine, goal: Goal, target: Target, a: ProductConcept, b: ProductConcept, personas: dict,
           progress=None) -> dict:
    run = {"ab_run_id": datetime.now().strftime("AB_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6],
           "created_at": datetime.now().isoformat(timespec="seconds"), "disclaimer": DISCLAIMER,
           "prompt_version": AB_PROMPT_VERSION, "model": engine.model, "options": OPTIONS,
           "evaluation_design": "독립(모나딕) 평가: 응답자에게 제품 하나씩만 제시, 기획 질문·비교 대상 비공개",
           "goal": goal.model_dump(), "target": {**target.model_dump(), "note": TARGET_NOTE},
           "products": {"A": a.model_dump(), "B": b.model_dump()}, "diff": concept_diff(a, b), "results": []}
    jobs = [(pid, side) for pid in target.persona_ids for side in ("A", "B")]
    for i, (pid, side) in enumerate(jobs):
        p = personas[pid]
        c = a if side == "A" else b
        if progress:
            progress(i, len(jobs), p, side)
        res = engine.evaluate_one(p, c, messages=build_ab_messages(p, target, c))
        res["side"] = side
        run["results"].append(res)
    run["finished_at"] = datetime.now().isoformat(timespec="seconds")
    return run


def save_ab_run(run: dict, d: Path = AB_RUNS_DIR) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{run['ab_run_id']}.json"
    p.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def list_ab_runs(d: Path = AB_RUNS_DIR) -> list[dict]:
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("*.json"), reverse=True):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            continue
    return out


# ---------------------------------------------------------------------------
# 분석: 관점(aspect) 언급의 공통·차이 (규칙 기반, 자동 검사 통과 응답만)
# ---------------------------------------------------------------------------
ASPECT_PATTERNS = {
    "맛·향": r"맛|향|상큼|달콤|새콤|풍미",
    "가격": r"\d[\d,]*\s*원|가격|비싸|비싼|저렴|가성비|행사|할인|돈",
    "당류·건강": r"당류|당분|(?<![가-힣])당(?:이|을|은|도| 함량| 섭취| 0)|설탕|무설탕|제로|건강|칼로리|성분|원재료",  # STEP 6.1: 단독 '당 ' 제거
    "청량감": r"청량|탄산|시원|톡 ?쏘",
    "용량": r"용량|mL|ml|양이|양도",
    "구매 상황 적합성": r"운동|식사|편의점|사무실|공부|업무|여름|더운|갈증|출근",
}


# STEP 6.1: '100mL당 가격', '1개당 가격' 같은 단위당 표현은 가격 관점이므로, 관점 분류 전에 '단위당'을 가격 표현으로 바꿈
_PER_UNIT = re.compile(r"(?:\d[\d,]*\s*)?(?:mL|ml|ML|밀리리터|리터|L|개|병|캔|팩)\s*당")


def aspects_of(text: str, extra_flavor: list[str]) -> set[str]:
    text = _PER_UNIT.sub(" 단위 가격 ", text)
    text = re.sub(r"용량\s*대비\s*(?:가격|가성비)", " 가성비 ", text)  # STEP 6.2: '용량 대비 가성비'는 가격 관점
    found = {k for k, p in ASPECT_PATTERNS.items() if re.search(p, text)}
    if any(w and w in text for w in extra_flavor):
        found.add("맛·향")
    return found


def _flavor_words(c: dict) -> list[str]:
    words = []
    for f in c.get("features", []) + [c.get("name", "")]:
        words += [w for w in re.findall(r"[가-힣]{2,}", f) if w not in ("제로", "탄산음료", "음료", "강한", "은은한", "청량감", "탄산", "진한", "약한")]
    return words


def analyze(run: dict, include_flagged: bool = False) -> dict:
    """유형별 A/B 관점 언급 비교. include_flagged=True 는 '참고용' 표시이며 근거로 쓰지 않음."""
    prods = run["products"]
    flavor = {s: _flavor_words(prods[s]) for s in ("A", "B")}
    by = {}
    for r in run["results"]:
        v = ab_view(prods[r["side"]], r)
        by.setdefault(r["persona_label"], {})[r["side"]] = (r, v)
    per_persona = []
    for persona, sides in by.items():
        entry = {"persona": persona, "usable": {}, "status": {}, "aspects": {}, "quotes": {}}
        for side in ("A", "B"):
            r, v = sides.get(side, (None, {"status": STATUS_GEN_FAIL}))
            entry["status"][side] = v.get("status")
            ok = is_usable(v) or (include_flagged and v.get("feedback"))
            entry["usable"][side] = bool(is_usable(v))
            fb = v.get("feedback") or {}
            for pol, key in (("긍정", "positive_points"), ("우려", "concerns")):
                asp = set()
                for item in fb.get(key, []) if ok else []:
                    for a_ in aspects_of(item, flavor[side]):
                        asp.add(a_)
                        entry["quotes"].setdefault(f"{side}|{pol}|{a_}", []).append(item)
                entry["aspects"][f"{side}|{pol}"] = sorted(asp)
        both_ok = entry["usable"]["A"] and entry["usable"]["B"]
        entry["comparable"] = both_ok or (include_flagged and all(sides.get(s, (None, {}))[1].get("feedback") for s in ("A", "B")))
        if entry["comparable"]:
            for pol in ("긍정", "우려"):
                sa, sb = set(entry["aspects"][f"A|{pol}"]), set(entry["aspects"][f"B|{pol}"])
                entry[f"{pol}_공통"] = sorted(sa & sb)
                entry[f"{pol}_A만"] = sorted(sa - sb)
                entry[f"{pol}_B만"] = sorted(sb - sa)
        per_persona.append(entry)
    return {"include_flagged": include_flagged, "per_persona": per_persona,
            "n_comparable": sum(e["comparable"] for e in per_persona),
            "n_usable_responses": sum(e["usable"][s] for e in per_persona for s in ("A", "B")),
            "n_responses": len(run["results"])}


# ---------------------------------------------------------------------------
# 후속 실제 조사: AI 응답에서 도출한 가설(미검증) + 설문 문항 후보
# ---------------------------------------------------------------------------
def hypotheses(run: dict, analysis: dict) -> list[dict]:
    goal_aspects = run["goal"]["aspects"]
    a_name, b_name = run["products"]["A"]["name"], run["products"]["B"]["name"]
    tgt = Target(**{k: run["target"][k] for k in ("age_band", "gender", "situations", "persona_ids")}, traits=run["target"].get("traits", "")).describe()
    out = []
    for e in analysis["per_persona"]:
        if not e["comparable"]:
            continue
        for pol, verb in (("우려", "더 우려할"), ("긍정", "더 긍정적으로 볼")):
            for side, name, other in (("A", a_name, b_name), ("B", b_name, a_name)):
                for asp in e.get(f"{pol}_{side}만", []):
                    if asp not in goal_aspects and asp != "구매 상황 적합성":
                        continue
                    out.append({
                        "가설": f"{tgt} {e['persona']} 소비자는 '{name}'의 {asp} 측면을 '{other}'보다 {verb} 수 있다",
                        "근거": " / ".join(e["quotes"].get(f"{side}|{pol}|{asp}", [])[:2]),
                        "출처": f"AI 가상 응답 ({e['persona']}, 제품 {side})",
                        "상태": "AI 도출 가설 · 미검증"})
    if analysis["n_comparable"] == 0:
        # 비교 가능한 A/B 응답 쌍이 없으면 '차이 없음'도 주장할 수 없음
        out.append({"가설": "자동 검사를 통과한 A/B 응답 쌍이 없어 AI 의견으로 가설을 도출하지 않음 → "
                          "기본 설문 문항으로 두 콘셉트를 직접 확인 필요",
                    "근거": "-", "출처": f"비교 가능 유형 0/{len(analysis['per_persona'])}", "상태": "가설 없음 · 미검증"})
        return out
    for asp in goal_aspects:  # AI 응답에서 차이가 드러나지 않은 관점도 실제 조사에서 확인할 대상으로 남김
        if not any(asp in e.get(f"{p}_{s}만", []) for e in analysis["per_persona"] if e["comparable"]
                   for p in ("긍정", "우려") for s in ("A", "B")):
            out.append({"가설": f"AI 응답에서는 두 콘셉트의 {asp} 측면 반응 차이가 드러나지 않았음 → 실제 소비자에게서 차이가 있는지 확인 필요",
                        "근거": "-", "출처": "AI 가상 응답 전체", "상태": "AI 도출 가설 · 미검증"})
    return out


def survey_candidates(run: dict, hyps: list[dict]) -> list[dict]:
    a, b = run["products"]["A"], run["products"]["B"]
    rows = [
        {"구분": "선별", "문항": "최근 1개월 내 탄산음료를 구매한 적이 있습니까? / 연령대·성별·주 구매 상황", "응답 형식": "선택형",
         "비고": "타깃 정의와 일치하는 응답자 확보"},
        {"구분": "제품별(단독 제시, 제시 순서 무작위)", "문항": "이 제품을 구매할 의향은 어느 정도입니까?", "응답 형식": "5점 척도",
         "비고": "A·B 각각. 순서 효과를 줄이려 응답자 절반은 B 먼저"},
        {"구분": "제품별", "문항": "판매가격(1개 기준)이 적절하다고 느끼십니까?", "응답 형식": "5점 척도",
         "비고": f"A {a['price_krw']:,}원 / B {b['price_krw']:,}원"},
    ]
    for asp in run["goal"]["aspects"]:
        rows.append({"구분": "제품별", "문항": f"이 제품의 '{asp}' 측면은 얼마나 마음에 드십니까?", "응답 형식": "5점 척도",
                     "비고": "검증 목표에서 선택한 관점"})
    rows += [
        {"구분": "비교(두 제품 모두 본 뒤)", "문항": "두 제품 중 먼저 사서 마셔 보고 싶은 제품은 무엇입니까?",
         "응답 형식": "A / B / 둘 다 아님", "비고": "선호 제품"},
        {"구분": "비교", "문항": "그렇게 선택한 이유는 무엇입니까?", "응답 형식": "자유 응답", "비고": "선택 이유. AI 의견과 주제 비교에 사용"},
    ]
    for h in hyps:
        if h["근거"] != "-":
            rows.append({"구분": "가설 확인", "문항": f"[{h['출처']}] {h['가설']} — 관련 경험·생각을 말씀해 주세요",
                         "응답 형식": "5점 척도 + 자유 응답", "비고": "AI 도출 가설 기반 · 미검증"})
    return rows


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
AB_CSV_COLUMNS = ["표시", "ab_run_id", "생성시각", "검증 질문", "판단 기준", "확인 관점", "타깃(가상 설정)", "구매 상황", "추가 특성", "제품",
                  "제품명", "가격(원)", "용량(mL)", "당류(g)", "특징", "소비자 유형", "첫인상", "좋은 점", "우려 사항",
                  "구매 시 고려 기준", "검토 상태", "검토 사유", "분석 근거 사용", "프롬프트"]


def ab_rows(run: dict) -> list[dict]:
    rows = []
    for r in run["results"]:
        c = run["products"][r["side"]]
        v = ab_view(c, r)
        fb = v.get("feedback") or {}
        rows.append({
            "표시": DISCLAIMER, "ab_run_id": run["ab_run_id"], "생성시각": run["created_at"],
            "검증 질문": run["goal"]["question"], "확인 관점": ", ".join(run["goal"]["aspects"]),
            "타깃(가상 설정)": Target(**{k: run["target"][k] for k in ("age_band", "gender", "situations", "persona_ids")}, traits=run["target"].get("traits", "")).describe(),
            "구매 상황": ", ".join(run["target"]["situations"]), "추가 특성": run["target"].get("traits", ""),
            "판단 기준": run["goal"].get("criterion", CRITERION_NONE), "제품": r["side"], "제품명": c["name"],
            "가격(원)": c["price_krw"], "용량(mL)": c["volume_ml"], "당류(g)": "미기재" if c["sugar_g"] is None else c["sugar_g"],
            "특징": ", ".join(c["features"]), "소비자 유형": r["persona_label"],
            "첫인상": fb.get("first_impression", ""), "좋은 점": " / ".join(fb.get("positive_points", [])),
            "우려 사항": " / ".join(fb.get("concerns", [])), "구매 시 고려 기준": " / ".join(fb.get("key_purchase_factors", [])),
            "검토 상태": v.get("status", ""), "검토 사유": " | ".join(v.get("flags", [])),
            "분석 근거 사용": "예" if is_usable(v) else "아니오", "프롬프트": run["prompt_version"]})
    return rows


def to_csv_bytes(rows: list[dict], columns: list[str]) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns)
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode("utf-8-sig")


# ---------------------------------------------------------------------------
# STEP 5.1: 상품기획 결론 (규칙 기반, LLM 호출 없음)
# ---------------------------------------------------------------------------
CONCLUSION_HOLD = "판단 보류"            # 비교할 근거가 부족 (A·B 모두 자동 검사 통과한 유형 0)
CONCLUSION_REVIEW = "추가 검토 필요"      # AI 의견은 있으나 타당성 미확인 (자동 검사 통과 ≠ 내용 검증)
CONCLUSION_LIMITED = "제한적 탐색 결과"   # 수동 검토로 인정된 A·B 의견 쌍이 있음 (그래도 판매·구매율 예측 아님)
EMPIRICAL_DATA_AVAILABLE = False         # 현재 실측 소비자 데이터 없음
_SALES_Q = re.compile(r"팔릴|판매|매출|구매율|잘 나갈|흥행|히트")


def objective_diffs(a: dict, b: dict) -> list[dict]:
    """제품 정보에서 직접 계산한 조건 차이. 소비자 선호·판매 전망 해석 없음."""
    out = []
    pa, pb = a["price_krw"], b["price_krw"]
    if pa != pb:
        hi, lo = ("B", "A") if pb > pa else ("A", "B")
        diff = abs(pb - pa)
        out.append({"항목": "판매가격", "제품 A": f"{pa:,}원", "제품 B": f"{pb:,}원",
                    "차이(계산값)": f"{hi}가 {lo}보다 {diff:,}원 높음 ({diff / min(pa, pb) * 100:.1f}%)"})
    ua, ub = round(pa / a["volume_ml"] * 100), round(pb / b["volume_ml"] * 100)
    if ua != ub:
        out.append({"항목": "100mL당 환산가", "제품 A": f"{ua:,}원", "제품 B": f"{ub:,}원",
                    "차이(계산값)": f"{abs(ub - ua):,}원 차이"})
    if a["volume_ml"] != b["volume_ml"]:
        out.append({"항목": "용량", "제품 A": f"{a['volume_ml']}mL", "제품 B": f"{b['volume_ml']}mL",
                    "차이(계산값)": f"{abs(a['volume_ml'] - b['volume_ml'])}mL 차이"})
    sa, sb = a.get("sugar_g"), b.get("sugar_g")
    if sa != sb:
        if sa is None or sb is None:
            d = "한쪽 당류 미기재 → 비교 불가"
        else:
            d = f"{'A' if sa > sb else 'B'}가 {abs(sa - sb):g}g 많음"
        out.append({"항목": "당류", "제품 A": "미기재" if sa is None else f"{sa:g}g", "제품 B": "미기재" if sb is None else f"{sb:g}g",
                    "차이(계산값)": d})
    for key, label in (("category", "종류"), ("features", "특징"), ("description", "설명")):
        if a.get(key) != b.get(key):
            fa = ", ".join(a[key]) if isinstance(a[key], list) else a[key]
            fb_ = ", ".join(b[key]) if isinstance(b[key], list) else b[key]
            out.append({"항목": label, "제품 A": fa or "-", "제품 B": fb_ or "-", "차이(계산값)": "다름"})
    same = [lbl for key, lbl in (("category", "종류"), ("volume_ml", "용량"), ("features", "특징")) if a.get(key) == b.get(key)]
    if same:
        out.append({"항목": "동일 조건", "제품 A": "", "제품 B": "", "차이(계산값)": ", ".join(same) + " 동일"})
    return out


def planning_conclusion(run: dict, analysis: dict | None = None) -> dict:
    """저장된 실행 결과와 검사 상태만으로 결론 상태를 정함. 제품 추천·우열·판매 예측은 하지 않음."""
    an = analysis or analyze(run, include_flagged=False)
    per = an["per_persona"]
    n_total = len(per)
    comparable = [e for e in per if e["comparable"]]
    manual = run.get("manual_review") or {}  # {"<유형>|A": "인정", ...} — 현재 화면에서 기록하는 기능 없음
    manual_pairs = [e["persona"] for e in comparable
                    if manual.get(f"{e['persona']}|A") == "인정" and manual.get(f"{e['persona']}|B") == "인정"]
    if not comparable:
        status = CONCLUSION_HOLD
    elif not manual_pairs:
        status = CONCLUSION_REVIEW
    else:
        status = CONCLUSION_LIMITED
    a, b = run["products"]["A"], run["products"]["B"]
    asks_sales = bool(_SALES_Q.search(run["goal"]["question"]))
    target = f"'{a['name']}'(A)와 '{b['name']}'(B) 중 어느 제품이 " + ("더 잘 팔릴지" if asks_sales else "더 나은 콘셉트인지(또는 먼저 검증할지)")
    headline = f"{target} 판단할 수 없습니다."

    reasons = []
    excluded = [e for e in per if not e["comparable"]]
    reasons.append(f"A·B 응답이 모두 자동 검사를 통과해 비교할 수 있는 소비자 유형: {len(comparable)}/{n_total}"
                   + (f" ({', '.join(e['persona'] for e in comparable)})" if comparable else ""))
    for e in excluded:
        reasons.append(f"비교 제외 — {e['persona']}: A {e['status']['A']}, B {e['status']['B']}")
    if comparable:
        reasons.append("자동 검사 통과는 형식·일부 규칙 위반이 없다는 뜻일 뿐이며, 통과한 응답에도 실제 내용 오류가 있을 수 있음 "
                       "(의미 정합성 미보장)")
    reasons.append(f"수동 검토로 인정된 A·B 의견 쌍: {len(manual_pairs)}개")
    reasons.append("실측 소비자 데이터 없음 → 구매율·판매량은 예측하지 않으며, AI 의견만으로 제품 우열을 정하지 않음"
                   if not EMPIRICAL_DATA_AVAILABLE else "실측 데이터 있음")
    if status == CONCLUSION_LIMITED:
        reasons.append("제한적 탐색 결과도 '소비자 반응 탐색'일 뿐, 판매·구매율 예측이나 제품 추천이 아님")

    # AI 의견 요약 (조건 차이와 분리, 근거로 쓰지 않음)
    ai_notes = []
    for e in comparable:
        parts = []
        for pol in ("긍정", "우려"):
            if e.get(f"{pol}_A만"):
                parts.append(f"{pol}(A에서만): {', '.join(e[f'{pol}_A만'])}")
            if e.get(f"{pol}_B만"):
                parts.append(f"{pol}(B에서만): {', '.join(e[f'{pol}_B만'])}")
            if e.get(f"{pol}_공통"):
                parts.append(f"{pol}(공통): {', '.join(e[f'{pol}_공통'])}")
        ai_notes.append({"소비자 유형": e["persona"], "언급 관점": " / ".join(parts) or "—",
                         "상태": "수동 검토 인정" if e["persona"] in manual_pairs else "자동 검사만 통과 · 내용 미검증"})

    # 후속 실제 조사 방향 (조건 차이·검증 목표에서 도출. 근거가 아니라 확인할 질문)
    diffs = objective_diffs(a, b)
    follow = [f"타깃({Target(**{k: run['target'][k] for k in ('age_band', 'gender', 'situations', 'persona_ids')}, traits=run['target'].get('traits', '')).describe()}) "
              "실제 응답자를 모집해 A·B를 각각 단독 제시(제시 순서 무작위)하고 구매의향을 측정"]
    items = {d["항목"] for d in diffs}
    if "판매가격" in items:
        follow.append(f"가격 조건이 다르므로({a['price_krw']:,}원 vs {b['price_krw']:,}원) 제품별 가격 적절성 문항을 따로 측정해 "
                      "가격 차이가 구매의향에 어떻게 작용하는지 확인")
    if "당류" in items:
        follow.append("당류 조건이 다르므로 당류 정보를 보여 준 상태에서 제품별 평가와 '당류가 선택에 미친 영향'을 측정")
    independent = items - {"동일 조건", "제품명", "100mL당 환산가"}  # 환산가는 가격·용량에서 계산되는 값이라 별도 조건으로 세지 않음
    if len(independent) >= 2:
        follow.append("두 제품은 여러 조건(" + ", ".join(sorted(independent)) + ")이 함께 달라 "
                      "선호 차이가 나와도 원인을 하나로 특정할 수 없음 → 원인을 확인하려면 조건을 하나만 바꾼 추가 비교 필요")
    follow.append("두 제품을 모두 본 뒤 먼저 구매해 보고 싶은 제품(A/B/둘 다 아님)과 이유를 수집")
    follow.append("AI 가설·설문 후보(아래 5절)는 검증할 질문 목록이며 근거가 아님")
    return {"status": status, "question": run["goal"]["question"], "headline": headline, "reasons": reasons,
            "objective_diffs": diffs, "ai_notes": ai_notes, "followups": follow,
            "n_comparable": len(comparable), "n_personas": n_total, "manual_pairs": manual_pairs,
            "empirical_data": EMPIRICAL_DATA_AVAILABLE, "recommendation": None}



# ---------------------------------------------------------------------------
# STEP 5.2: 조건 기반 상품기획 제안 (목표 기준 + 객관적 제품 조건만 사용, AI 응답 미사용)
# ---------------------------------------------------------------------------
_HINTS = {"당류가 낮은 제품 우선": re.compile(r"당류|당 섭취|당을 줄|다이어트|저당|설탕|제로 슈거|혈당"),
          "판매가격이 낮은 제품 우선": re.compile(r"저렴|싼 |가격 부담|가성비|저가")}


# 타깃 '특성'으로 볼 만한 표현 (제품 속성 언급인 '당류' 단독은 제외) — 추가 특성 미전달 경고용
_TRAIT_HINT = re.compile(r"다이어트|당 섭취|당을 줄|당류를 줄|당류 섭취|혈당|체중|살 빼|운동하는|헬스")


def trait_hint(text: str) -> list[str]:
    return sorted(set(_TRAIT_HINT.findall(text)))


def criterion_hints(*texts: str) -> list[str]:
    """질문·추가 특성에 판단 기준과 관련된 표현이 있으면 안내만 함 (자동 적용하지 않음)."""
    t = " ".join(texts)
    return [c for c, p in _HINTS.items() if p.search(t)]


_JOSA_GA = {"당류": "가", "판매가격": "이", "100mL당 환산가": "가"}


def _fmt(v, unit):
    return f"{v:,.0f}" if unit == "원" else f"{v:g}"


def condition_proposal(a: dict, b: dict, criterion: str) -> dict:
    """명시된 판단 기준으로 제품 조건만 비교해 '우선 검증 후보'를 제안. 소비자 선호·판매 예측이 아님."""
    base = {"criterion": criterion, "derivable": False, "candidate": None, "reason": "", "tradeoffs": []}
    if criterion == CRITERION_NONE:
        return {**base, "reason": "판단 기준이 지정되지 않아 제품 조건만으로 우선 검증 후보를 정하지 않았습니다."}
    if criterion == "당류가 낮은 제품 우선":
        va, vb, unit, label = a.get("sugar_g"), b.get("sugar_g"), "g", "당류"
    elif criterion == "판매가격이 낮은 제품 우선":
        va, vb, unit, label = a["price_krw"], b["price_krw"], "원", "판매가격"
    else:
        va, vb = round(a["price_krw"] / a["volume_ml"] * 100), round(b["price_krw"] / b["volume_ml"] * 100)
        unit, label = "원", "100mL당 환산가"
    if va is None or vb is None:
        return {**base, "reason": f"한쪽 제품의 {label} 정보가 없어 이 기준으로 비교할 수 없습니다."}
    if va == vb:
        return {**base, "reason": f"두 제품의 {label}{_JOSA_GA[label]} 같아({_fmt(va, unit)}{unit}) 이 기준으로는 구분되지 않습니다."}
    cand, other = ("A", "B") if va < vb else ("B", "A")
    pc, po = (a, b) if cand == "A" else (b, a)
    reason = f"판단 기준 '{criterion}'에 따라 {label}{_JOSA_GA[label]} 더 낮은 제품 {cand}({pc['name']}, {_fmt(min(va, vb), unit)}{unit})를 " \
             f"우선 검증 후보로 제안합니다. 제품 {other}는 {_fmt(max(va, vb), unit)}{unit}입니다."
    tradeoffs = []
    if criterion != "판매가격이 낮은 제품 우선" and pc["price_krw"] > po["price_krw"]:
        tradeoffs.append(f"후보 {cand}의 판매가격이 {pc['price_krw'] - po['price_krw']:,}원 더 높습니다 "
                         f"({pc['price_krw']:,}원 vs {po['price_krw']:,}원). 기준 외 조건에서는 불리할 수 있습니다.")
    if criterion != "당류가 낮은 제품 우선" and None not in (pc.get("sugar_g"), po.get("sugar_g")) and pc["sugar_g"] > po["sugar_g"]:
        tradeoffs.append(f"후보 {cand}의 당류가 {pc['sugar_g'] - po['sugar_g']:g}g 더 많습니다.")
    if pc["volume_ml"] != po["volume_ml"]:
        tradeoffs.append(f"두 제품의 용량이 다릅니다({pc['volume_ml']}mL vs {po['volume_ml']}mL).")
    for key, lbl in (("category", "종류"), ("features", "특징"), ("description", "설명")):
        if pc.get(key) != po.get(key):
            tradeoffs.append(f"두 제품의 {lbl}도 달라, 기준 항목만의 차이로 볼 수 없습니다.")
    same_price = pc["price_krw"] == po["price_krw"] and criterion != "판매가격이 낮은 제품 우선"
    if same_price and not tradeoffs:
        reason += " 다른 입력 조건(가격·용량·종류·특징)은 같습니다."
    return {**base, "derivable": True, "candidate": cand, "reason": reason, "tradeoffs": tradeoffs}
