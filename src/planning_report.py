"""단일 콘셉트 평가 결과 → 상품기획 보고서 + 시뮬레이션 품질 정보 (STEP 6, 규칙 기반, LLM 호출 없음).

원칙
- 분석 근거는 '자동 검사 통과' 응답만 사용. 검토 필요·스키마 실패·생성 실패 응답은 '분석 제외'로 표시.
- 자동 검사 통과도 내용 정확성을 보장하지 않음 → 모든 AI 기반 문장은 'AI 가상 의견 · 미검증'.
- 비교 근거가 없으면 우열·결론을 내지 않고 '검증이 필요한 쟁점'만 제시.
- 실제 시장 선호·구매율·판매량을 예측하지 않음.
"""
from __future__ import annotations

from ab_comparison import ASPECT_PATTERNS, aspects_of
from qualitative_feedback import (CHECKS_VERSION, STATUS_GEN_FAIL, STATUS_SCHEMA_FAIL, effective_view)

# 개발 중 내부 점검에서 확인된 주의사항 (화면의 '시뮬레이션 품질 정보'에 표시)
# [공개본 변경] 내부 단계 번호와 공개본에 없는 개발 보고서 경로를 문구에서 제거 (내용·수치는 원본과 같음)
KNOWN_LIMITS = [
    "같은 입력으로 다시 실행해도 결과가 크게 달라질 수 있습니다 (개발 중 점검: 같은 조건 5회 실행 시 자동 검사 통과 0~5/6, "
    "수동 검토 오류 1/6~5/6).",
    "자동 검사를 통과한 응답에서도 수동 검토 시 내용 오류가 확인되었습니다 (개발 중 점검: 3건, 5/7건).",
    "두 제품을 한 번에 보여 주고 고르게 하면 먼저 제시된 제품을 고르는 경향이 컸습니다 (개발 중 점검: 7/8). "
    "그래서 이 서비스는 제품을 하나씩 따로 평가하며, 직접 선택 결과는 제공하지 않습니다.",
]


# STEP 6.1: 경고를 종류별로 구분 (짧은 응답 = 근거 부족, 입력에 없는 특징 단정 = 제품 정보 왜곡 의심)
FLAG_CATEGORIES = [("제품 정보 왜곡 의심", "[입력 불일치]"), ("근거 부족", "[근거 부족]"), ("건강 효과 단정", "[건강 효과 단정]")]


def flag_counts(flags: list[str], notes: list[str] | None = None) -> dict:
    out = {name: sum(f.startswith(pfx) for f in flags) for name, pfx in FLAG_CATEGORIES}
    out["기타 자동 경고"] = sum(not any(f.startswith(p) for _, p in FLAG_CATEGORIES) for f in flags)
    out["검토 메모(건강 효과 추론)"] = len(notes or [])
    return out


def reason_summary(flags: list[str]) -> str:
    c = flag_counts(flags)
    return ", ".join(f"{k} {v}" for k, v in c.items() if v and k != "검토 메모(건강 효과 추론)") or "-"


def usable(v: dict) -> bool:
    s = v.get("status", "")
    return bool(v.get("feedback")) and not s.startswith("검토 필요") and s != STATUS_GEN_FAIL


def _flavor_words(c: dict) -> list[str]:
    import re
    words = []
    for f in c.get("features", []) + [c.get("name", "")]:
        words += [w for w in re.findall(r"[가-힣]{2,}", f) if w not in ("제로", "탄산음료", "음료", "강한", "은은한", "청량감", "탄산", "진한", "약한")]
    return words


def quality_panel(run: dict, review: dict | None = None) -> dict:
    """응답별 검증 정보 + 요약. effective_view 를 그대로 사용(저장 파일·원문 무수정).
    STEP 6.2: 보고서 근거는 '사람이 승인한 문장'이므로 응답별 문장 검토 현황을 함께 표시."""
    import sentence_review as sr
    review = sr.load_review(run) if review is None else review
    all_units = sr.units(run)
    rows = []
    for i, r in enumerate(run["results"]):
        mine = [u for u in all_units if u["unit_id"].startswith(f"{i}|")]
        dec = [sr.decision_of(review, u["unit_id"]) for u in mine]
        v = effective_view(run, r)
        flags = v.get("flags", [])
        s = v.get("status", "")
        rows.append({
            "소비자": r["persona_label"],
            "JSON 검증": "실패(응답 없음)" if s == STATUS_GEN_FAIL else "실패" if s == STATUS_SCHEMA_FAIL else "통과",
            **(flag_counts(flags, v.get("notes")) if s not in (STATUS_GEN_FAIL, STATUS_SCHEMA_FAIL)
               else {k: "-" for k in [n for n, _ in FLAG_CATEGORIES] + ["기타 자동 경고", "검토 메모(건강 효과 추론)"]}),
            "응답 단위 자동 검사": "통과 (내용 미보장)" if usable(v) else "검토 필요",
            "문장 검토 (승인/제외/검토 전)": f"{dec.count(sr.APPROVED)} / {dec.count(sr.EXCLUDED)} / {dec.count(sr.PENDING)}"
                                       if mine else "-",
            "보고서 근거 사용": f"승인 문장 {dec.count(sr.APPROVED)}개" if sr.APPROVED in dec else "없음",
            "프롬프트": r.get("prompt_version", run.get("prompt_version")),
        })
    n = len(rows)
    return {"rows": rows, "n": n, "n_json_ok": sum(x["JSON 검증"] == "통과" for x in rows),
            "n_distortion": sum(isinstance(x["제품 정보 왜곡 의심"], int) and x["제품 정보 왜곡 의심"] > 0 for x in rows),
            "n_thin": sum(isinstance(x["근거 부족"], int) and x["근거 부족"] > 0 for x in rows),
            "n_usable": sum(x["보고서 근거 사용"] != "없음" for x in rows),
            "review_counts": {d: sum(sr.decision_of(review, u["unit_id"]) == d for u in all_units) for d in sr.DECISIONS},
            "single_run": True, "checks_version": CHECKS_VERSION, "known_limits": KNOWN_LIMITS}


def concept_summary(c: dict) -> list[str]:
    sugar = "미기재" if c.get("sugar_g") is None else ("0g (무설탕)" if c["sugar_g"] == 0 else f"{c['sugar_g']:g}g")
    unit = round(c["price_krw"] / c["volume_ml"] * 100)
    out = [f"{c['name']} — {c['category']}, 1개 {c['volume_ml']}mL, 판매가격 {c['price_krw']:,}원 (100mL당 {unit:,}원), 당류 {sugar}",
           f"특징: {', '.join(c.get('features') or []) or '입력 없음'}"]
    if c.get("description"):
        out.append(f"설명: {c['description']}")
    return out


def condition_checks(c: dict) -> list[str]:
    """제품 입력값만으로 알 수 있는 확인 사항 (AI 응답과 무관)."""
    out = []
    if c.get("sugar_g") is None:
        out.append("당류 정보가 비어 있어 건강·당류 관련 반응을 해석하기 어렵습니다. 당류 값을 정해 두세요.")
    if not c.get("description"):
        out.append("콘셉트 설명이 없어 맛·사용 상황에 대한 반응이 추측에 의존합니다. 한두 문장 설명을 추가해 보세요.")
    if len(c.get("features") or []) < 2:
        out.append("특징이 1개 이하라 차별점 판단 근거가 부족합니다.")
    out.append(f"판매가격 {c['price_krw']:,}원에 대한 실제 수용도는 AI로 판단할 수 없어 실제 조사에서 확인해야 합니다.")
    return out


HYP_STATUS = "AI 도출 · 사람이 문장 정합성 승인 · 실제 소비자 검증 전"


def build_report(run: dict, review: dict | None = None) -> dict:
    """STEP 6.2: 보고서의 긍정 요인·구매 장벽·가설 근거에는 **사람이 승인한 문장만** 사용.
    자동 검사 통과만으로는 사용하지 않으며, 검토 전 문장은 자동 승인하지 않음."""
    import sentence_review as sr
    c = run["concept"]
    flavor = _flavor_words(c)
    review = sr.load_review(run) if review is None else review
    all_units = sr.units(run)
    counts = {d: sum(sr.decision_of(review, u["unit_id"]) == d for u in all_units) for d in sr.DECISIONS}
    per, excluded = [], []
    for i, r in enumerate(run["results"]):
        v = effective_view(run, r)
        mine = [u for u in all_units if u["unit_id"].startswith(f"{i}|")]
        ok = [u for u in mine if sr.decision_of(review, u["unit_id"]) == sr.APPROVED]
        pos = [u["text"] for u in ok if u["field"] == "positive_points"]
        con = [u["text"] for u in ok if u["field"] == "concerns"]
        if not pos and not con:
            excluded.append({"소비자": r["persona_label"], "상태": v.get("status"),
                             "사유": (f"승인된 좋은 점·우려 문장 없음 (승인 {len(ok)} · 제외 "
                                    f"{sum(sr.decision_of(review, u['unit_id']) == sr.EXCLUDED for u in mine)} · 검토 전 "
                                    f"{sum(sr.decision_of(review, u['unit_id']) == sr.PENDING for u in mine)})")
                             if v.get("feedback") else v.get("status")})
            continue
        pos_asp, con_asp = {}, {}
        for item in pos:
            for a in aspects_of(item, flavor):
                pos_asp.setdefault(a, []).append(item)
        for item in con:
            for a in aspects_of(item, flavor):
                con_asp.setdefault(a, []).append(item)
        health_noted = {u["text"] for u in ok if u["auto_notes"]}
        per.append({"health_noted": health_noted, "소비자": r["persona_label"], "positives": pos, "barriers": con,
                    "first_impressions": [u["text"] for u in ok if u["field"] == "first_impression"],
                    "pos_aspects": pos_asp, "con_aspects": con_asp,
                    "profile_sources": (r.get("profile") or {}).get("attrs")})

    # 공통 쟁점 / 차이 (근거 응답이 2개 이상일 때만)
    common, diffs = [], []
    if len(per) >= 2:
        for asp in ASPECT_PATTERNS:
            who_con = [p["소비자"] for p in per if asp in p["con_aspects"]]
            who_pos = [p["소비자"] for p in per if asp in p["pos_aspects"]]
            if len(who_con) >= 2:
                common.append(f"{asp}: {len(who_con)}개 유형이 우려로 언급 ({', '.join(who_con)})")
            elif len(who_pos) >= 2 and not who_con:
                common.append(f"{asp}: {len(who_pos)}개 유형이 긍정 요인으로 언급 ({', '.join(who_pos)})")
            if who_con and len(who_con) < len(per):
                diffs.append(f"{asp}을(를) 우려로 언급한 유형: {', '.join(who_con)} / 언급하지 않은 유형: "
                             f"{', '.join(p['소비자'] for p in per if p['소비자'] not in who_con)}")
    # 가설 (AI 도출 · 미검증) — 읽기 쉽도록 소비자별 장벽 2개·동기 1개로 제한 (여러 유형이 언급한 관점 우선)
    def _rank(asp, key):
        return -sum(asp in q_[key] for q_ in per)
    hyps = []
    for p in per:
        for asp, items in sorted(p["con_aspects"].items(), key=lambda kv: _rank(kv[0], "con_aspects"))[:2]:
            hyps.append({"가설": f"'{p['소비자']}'(가상 설정) 소비자는 이 콘셉트의 {asp} 측면을 구매 장벽으로 느낄 수 있다",
                         "근거 문장(AI)": items[0], "상태": HYP_STATUS})
        for asp, items in sorted(p["pos_aspects"].items(), key=lambda kv: _rank(kv[0], "pos_aspects"))[:1]:
            hyps.append({"가설": f"'{p['소비자']}'(가상 설정) 소비자는 이 콘셉트의 {asp} 측면을 구매 동기로 볼 수 있다",
                         "근거 문장(AI)": items[0],
                         "상태": HYP_STATUS + (" · 근거 문장이 입력값만으로 건강 효과를 추측함(건강 효과는 검증 대상 아님)"
                                                     if items[0] in p["health_noted"] else "")})
    # 다음 단계
    nxt = list(condition_checks(c))
    issues = sorted({a for p in per for a in list(p["con_aspects"]) + list(p["pos_aspects"])})
    nxt.append("실제 조사 문항 후보: 제품 단독 제시 후 구매의향(5점), 가격 적절성(5점)"
               + (f", 관점별 평가({', '.join(issues)})" if issues else "") + ", 구매하고 싶은/망설여지는 이유(자유 응답)")
    if counts[sr.PENDING]:
        nxt.insert(0, f"검토 전 문장 {counts[sr.PENDING]}개가 있습니다. '문장별 검토'에서 승인·제외를 정하면 보고서에 반영됩니다.")
    if len(per) < 2:
        nxt.insert(0, f"승인된 문장이 있는 소비자 유형이 {len(per)}개뿐이라 유형 간 비교를 하지 않았습니다.")
    # [공개본 변경] 결과 폴더가 프로젝트 밖(APP_OUTPUT_DIR)이어도 오류가 나지 않도록 표시용 경로 함수 사용
    import runtime_config as rc
    return {"review_counts": counts, "review_file": rc.display_path(sr.review_path(run["run_id"])),
            "concept_summary": concept_summary(c), "per_persona": per, "excluded": excluded,
            "common_issues": common, "differences": diffs, "hypotheses": hyps, "next_steps": nxt,
            "n_usable": len(per), "n_total": len(run["results"]), "comparison_possible": len(per) >= 2,
            "product_ranking": None}
