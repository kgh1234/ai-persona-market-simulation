"""AI 가상 소비자 정성 피드백 MVP (STEP 4) + 목표 기반 A/B 콘셉트 비교 (STEP 5)
+ 구조화 페르소나·상품기획 보고서·품질 정보·실제 조사 CSV 형식 검증 (STEP 6).

실행: OLLAMA_HOST=127.0.0.1:11434 streamlit run app.py   (127.0.0.1:8501 전용, README 참고)
- LLM은 로컬 Ollama(qwen2.5:7b)만 사용. 외부 LLM API 없음. TinyTroupe 미사용.
- [공개본 변경] Ollama 주소는 OLLAMA_HOST, 실행 결과 폴더는 APP_OUTPUT_DIR(기본 outputs/)로 지정 (src/runtime_config.py).
- 정량 점수·순위·구매율 그래프는 제공하지 않음 (STEP 3C에서 구분력 미확인).
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import ab_comparison as ab  # noqa: E402
import persona_profiles as pp  # noqa: E402
import planning_report as prpt  # noqa: E402
import sentence_review as srv  # noqa: E402
import survey_data as sdata  # noqa: E402
from qualitative_feedback import (CHECKS_VERSION, DISCLAIMER, FeedbackEngine, ProductConcept,  # noqa: E402
                                  STATUS_GEN_FAIL, STATUS_SCHEMA_FAIL, effective_view, list_runs, load_personas, run_to_rows, rows_to_csv_bytes,
                                  save_run)
import runtime_config as rc  # noqa: E402  [공개본 변경]

rc.ensure_output_dirs()  # [공개본 변경] 새 환경에서도 outputs/ 하위 폴더를 만들고 시작

st.set_page_config(page_title="AI 가상 소비자 정성 피드백 (MVP)", layout="wide")

# STEP 6.3: PPT 시연 모드 (화면 표현만 변경, 기본 OFF). 계산·저장·검토 데이터는 동일
DEMO = st.sidebar.toggle("PPT 시연 모드", value=False, key="demo_mode",
                         help="상품기획 결과 중심으로 간결하게 표시합니다. 품질 검토 정보는 '품질 검토 상세'에서 그대로 확인할 수 있습니다.")
DEMO_LINE = "AI 페르소나 기반 가상 평가 · 실제 소비자 검증 전"
# STEP 6.4: 시연 모드를 켤 때마다 '품질 검토 상세 보기'는 모두 꺼진 상태로 시작 (이전에 켰던 상태가 남지 않게)
if DEMO and not st.session_state.get("_demo_prev", False):
    for _k in [k for k in st.session_state.keys() if str(k).startswith("demo_detail_")]:
        st.session_state[_k] = False
st.session_state["_demo_prev"] = DEMO
if DEMO:  # 차분한 중립색 (시연 화면 전용 최소 CSS)
    st.markdown("""
<style>
.demo-card {border: 1px solid rgba(128,128,128,0.35); border-radius: 8px; padding: 12px 16px; margin-bottom: 8px;}
.demo-muted {color: rgba(128,128,128,0.95); font-size: 14px;}
</style>
""", unsafe_allow_html=True)

engine = FeedbackEngine()
personas = load_personas()
persona_by_label = {p["label"]: p for p in personas}


def nbox(fn, text: str, neutral: bool | None = None):
    """STEP 6.5: 시연 모드에서는 노랑·빨강·초록 상자 대신 중립 테두리 카드로 같은 내용을 표시 (OFF는 원래 상자).
    품질 검토 상세 화면에서는 neutral=False 로 원래 상자를 그대로 사용."""
    if DEMO if neutral is None else neutral:
        with st.container(border=True):
            st.markdown(text)
    else:
        fn(text)


DETAIL_TITLE = "#### 품질 검토 상세 (검증용 화면 · 시연 화면 아님)"

# ---------------------------------------------------------------------------
# STEP 7: NVIDIA 합성 페르소나 불러오기 (실험) — experiments/nemotron_personas 의 선별·변환 코드를 그대로 사용
# ---------------------------------------------------------------------------
NEMO_DIR = ROOT / "experiments" / "nemotron_personas"
NEMO_NOTICE = ("공개 통계 분포를 바탕으로 만든 **인구통계 합성 프로필**입니다. 실제 소비자 데이터나 검증된 대표 표본이 아닙니다. "
               "음료 구매 빈도·가격 허용 범위·맛 선호 등은 데이터에 없어 비워 둡니다.")
NEMO_ATTRIBUTION = ("데이터 출처: **NVIDIA, Nemotron-Personas-Korea** "
                    "([Hugging Face](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea)) · "
                    "라이선스 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · "
                    "변경 사항: 일부 레코드를 선별해 연령대·성별·직업/생활 속성으로 형식만 변환 (서사 텍스트는 사용하지 않음). "
                    "NVIDIA와 무관한 실험 기능입니다.")


def _nemo_modules():
    if str(NEMO_DIR) not in sys.path:
        sys.path.insert(0, str(NEMO_DIR))
    import adapter as nad  # noqa: E402
    import nemotron_source as ns  # noqa: E402
    import selection as nsel  # noqa: E402
    return ns, nsel, nad


@st.cache_resource(show_spinner="NVIDIA 합성 페르소나의 인구통계 열을 불러오는 중… (처음 한 번 약 1분)")
def _nemo_demographics():
    ns, _, _ = _nemo_modules()
    return ns.load_demographics().to_pandas()


def _nemo_cached() -> bool:
    return all((rc.NEMOTRON_CACHE_DIR / f"demo_shard{i}.parquet").exists() for i in range(9))  # [공개본 변경] 캐시 위치


def run_has_synthetic(run: dict) -> bool:
    return any(a.get("source") == pp.SRC_SYNTH for r in run.get("results", [])
               for a in ((r.get("profile") or {}).get("attrs") or {}).values())


def synth_run_note(run: dict):
    if run_has_synthetic(run):
        st.caption("🧬 일부 응답자는 NVIDIA Nemotron-Personas-Korea(CC BY 4.0) 인구통계 합성 프로필을 바탕으로 한 가상 응답자입니다. "
                   "실제 소비자 응답이 아닙니다.")


def badge():
    st.markdown(f":orange-background[**{DISCLAIMER}**]")


# 최소 CSS: 한국어 단어 단위 줄바꿈(keep-all), 본문 15px, 표 세로 정렬·좁은 화면 가로 스크롤
st.markdown("""
<style>
[data-testid="stTable"] { overflow-x: auto; }
[data-testid="stTable"] table { font-size: 15px; width: 100%; }
[data-testid="stTable"] th, [data-testid="stTable"] td {
  word-break: keep-all; overflow-wrap: anywhere; white-space: normal; vertical-align: top; line-height: 1.5; }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {
  word-break: keep-all; overflow-wrap: anywhere; font-size: 15px; line-height: 1.6; }
[data-testid="stAlertContainer"] p, [data-testid="stAlertContainer"] li { font-size: 14px; }
</style>
""", unsafe_allow_html=True)

REVIEW_PREFIX = "⚠ 검토 필요 · "
SECTIONS = (("positive_points", "좋은 점"), ("concerns", "우려 사항"), ("key_purchase_factors", "구매 시 고려 기준"))


def is_flagged(chk: dict) -> bool:
    return chk.get("status", "").startswith("검토 필요") or chk.get("status") == STATUS_GEN_FAIL


def status_label(chk: dict) -> str:
    """상태 표시 문자열. 생성 실패(원문 없음)와 스키마 실패(원문 있음)·검토 필요를 구분."""
    s = chk.get("status", "")
    if s == STATUS_GEN_FAIL:
        return "❌ 생성 실패"
    if s == STATUS_SCHEMA_FAIL:
        return "⛔ 검토 필요 (스키마 실패)"
    if s.startswith("검토 필요"):
        return f"⚠ 검토 필요 ({len(chk.get('flags', []))}건)"
    return "✅ 자동 검사 통과"


def short(text: str, n: int = 32) -> str:
    """요약용: n자 이내에서 어절 경계로 자름 (원문은 카드와 원문 펼침에 그대로 있음)."""
    text = " ".join(str(text).split())
    if len(text) <= n:
        return text
    cut = text[:n].rsplit(" ", 1)[0]
    return (cut if len(cut) >= n // 2 else text[:n]) + "…"


def summary_table(run: dict, views: list) -> pd.DataFrame:
    """짧은 요약 표: 상태 + 항목 수와 첫 항목 요약. 전체 의견은 상세 카드에서 확인."""
    rows = []
    for r, v in zip(run["results"], views):
        fb = v.get("feedback") or {}
        row = {"소비자 유형": r["persona_label"], "검토 상태": status_label(v)}
        for key, title in SECTIONS:
            items = fb.get(key, [])
            if not fb:
                row[title] = "—"
            elif key == "key_purchase_factors":
                row[title] = ", ".join(short(x, 16) for x in items)
            else:
                row[title] = f"{short(items[0])} (총 {len(items)}개)" if items else "—"
        rows.append(row)
    df = pd.DataFrame(rows).set_index("소비자 유형")
    df.index.name = None  # st.table 이 인덱스 이름을 별도 머리글 줄로 그리지 않도록
    return df


def render_status(chk: dict):
    """상태와 검토 사유를 카드 상단에만 표시 (표 전체에 배경색을 쓰지 않음)."""
    s = chk.get("status", "")
    flags = chk.get("flags", [])
    reasons = "\n".join(f"- {f}" for f in flags)
    if s == STATUS_GEN_FAIL:
        st.error("**❌ 생성 실패** — 응답 원문이 없습니다 (요청·통신 오류).\n\n" + reasons)
    elif s == STATUS_SCHEMA_FAIL:
        st.error("**⛔ 검토 필요 (스키마 실패)** — 응답 원문은 있으나 형식 검증에 실패해 의견으로 표시하지 않습니다.\n\n" + reasons)
    elif s.startswith("검토 필요"):
        st.warning(f"**⚠ 검토 필요** — 자동 검사 경고 {len(flags)}건. 검토 전에는 소비자 의견으로 인용하지 마세요.\n\n" + reasons)
    else:
        st.success("**✅ 자동 검사 통과** (의미 정합성은 보장하지 않음)")


def render_feedback(fb: dict):
    st.markdown(f"**첫인상**  \n{fb.get('first_impression', '')}")
    for key, title in SECTIONS:
        st.markdown(f"**{title}**")
        st.markdown("\n".join(f"- {x}" for x in fb.get(key, [])) or "—")


def render_quality(qp: dict, expanded: bool = False):
    """시뮬레이션 품질 정보 (STEP 6). 자동 검사 통과를 내용 정확성 보장으로 표현하지 않음."""
    head = (f"보고서 근거(승인 문장)가 있는 소비자 {qp['n_usable']}/{qp['n']}" if "review_counts" in qp
            else f"분석 근거로 쓸 수 있는 응답 {qp['n_usable']}/{qp['n']}")
    with st.expander(f"시뮬레이션 품질 정보 — {head}", expanded=expanded):
        review_line = (f"- 문장 검토: 승인 **{qp['review_counts']['승인']}** · 제외 **{qp['review_counts']['제외']}** · "
                       f"검토 전 **{qp['review_counts']['검토 전']}**  \n" if "review_counts" in qp
                       else "- 의미 오류 검토: **사람 검토 기록 없음**  \n")
        st.markdown(f"- JSON 형식 검증 통과 **{qp['n_json_ok']}/{qp['n']}**  \n"
                    f"- 제품 정보 왜곡 의심(입력에 없는 특징 단정) **{qp['n_distortion']}/{qp['n']}** · "
                    f"근거 부족(설명이 너무 짧음) **{qp['n_thin']}/{qp['n']}** — 짧은 응답은 왜곡으로 세지 않음  \n"
                    + review_line +
                    f"- 자동 검사({qp['checks_version']})는 표현 규칙만 확인하며 내용 정확성을 보장하지 않음  \n"
                    f"- 반복 안정성: **이번 결과는 1회 실행**이라 다시 실행하면 달라질 수 있음")
        st.table(pd.DataFrame(qp["rows"]).set_index("소비자").rename_axis(None))
        st.markdown("**지난 실험에서 확인된 주의사항**")
        st.markdown("\n".join(f"- {x}" for x in qp["known_limits"]))


def render_planning_report(run: dict, review: dict):
    """상품기획 보고서 (STEP 6.2). 긍정 요인·구매 장벽·가설 근거는 사람이 승인한 문장만 사용."""
    rep_ = prpt.build_report(run, review)
    st.markdown("### 상품기획 보고서")
    st.caption("AI 가상 의견을 정리한 탐색용 보고서입니다. 실제 소비자 조사 결과가 아니며, 구매율·판매량을 예측하지 않습니다.")
    st.markdown("**① 제품 콘셉트 요약** (입력값)")
    st.markdown("\n".join(f"- {x}" for x in rep_["concept_summary"]))
    rc = rep_["review_counts"]
    st.markdown(f"**② 소비자별 긍정 요인과 구매 장벽** — 사람이 승인한 AI 문장만 (승인 {rc['승인']} · 제외 {rc['제외']} · 검토 전 {rc['검토 전']})")
    st.caption(srv.APPROVAL_MEANING)
    if rep_["per_persona"]:
        cols = st.columns(len(rep_["per_persona"]))
        for col, p in zip(cols, rep_["per_persona"]):
            with col:
                with st.container(border=True):
                    st.markdown(f"**{p['소비자']}**")
                    if p.get("profile_sources"):
                        cnt = {s_: sum(a["source"] == s_ for a in p["profile_sources"].values()) for s_ in pp.ALL_SOURCES}
                        st.caption("페르소나 속성 출처: " + " · ".join(f"{pp.SOURCE_ICON[k]} {k} {v}" for k, v in cnt.items()
                                                                  if v or k in pp.SOURCES))
                    st.markdown("긍정 요인\n" + "\n".join(f"- {x}" for x in p["positives"]))
                    st.markdown("구매 장벽\n" + "\n".join(f"- {x}" for x in p["barriers"]))
    else:
        st.info("승인된 문장이 없어 소비자별 요인을 정리하지 않았습니다. 아래 '문장별 검토'에서 문장을 확인해 승인·제외를 정하세요. "
                "검토 전 문장은 자동으로 승인하지 않습니다.")
    if rep_["excluded"]:
        st.caption("보고서에 반영되지 않은 소비자: " + " / ".join(f"{e['소비자']}({e['사유']})" for e in rep_["excluded"]))
    st.markdown("**③ 공통으로 확인할 쟁점 · ④ 소비자 간 반응 차이** (언급된 관점 기준, 선호·우열 아님)")
    if not rep_["comparison_possible"]:
        st.info("승인된 문장이 있는 소비자 유형이 2개 미만이라 공통 쟁점과 반응 차이를 정리하지 않았습니다. 제품 우열도 판단하지 않습니다.")
    else:
        st.markdown("\n".join(f"- 공통: {x}" for x in rep_["common_issues"]) or "- 공통으로 언급된 관점 없음")
        st.markdown("\n".join(f"- 차이: {x}" for x in rep_["differences"]) or "- 차이 없음")
    st.markdown("**⑤ 실제 소비자 조사에서 검증할 가설** — 승인된 AI 문장에서 도출 · 실제 소비자 검증 전")
    if rep_["hypotheses"]:
        st.table(pd.DataFrame(rep_["hypotheses"]))
    else:
        st.caption("승인된 문장이 없어 가설을 만들지 않았습니다.")
    st.markdown("**⑥ 상품기획자가 다음 단계에서 확인할 사항**")
    st.markdown("\n".join(f"- {x}" for x in rep_["next_steps"]))


def render_sentence_review(run: dict, review: dict, live: bool):
    """문장(항목)별 검토 (STEP 6.2). 결과는 outputs/reviews/<run_id>.json 에 저장 [공개본 변경: 경로 표기], 원본 실행 JSON은 수정하지 않음."""
    us = srv.units(run)
    if not us:
        return
    icon = {srv.AUTO_DISTORT: "⚠", srv.AUTO_WARN: "⚠", srv.AUTO_HEALTH: "⚠", srv.AUTO_THIN_OK: "ℹ", srv.AUTO_THIN: "ℹ", srv.AUTO_OK: "·"}
    with st.expander(f"문장별 검토 — 승인 {sum(srv.decision_of(review, u['unit_id']) == srv.APPROVED for u in us)} · "
                     f"제외 {sum(srv.decision_of(review, u['unit_id']) == srv.EXCLUDED for u in us)} · "
                     f"검토 전 {sum(srv.decision_of(review, u['unit_id']) == srv.PENDING for u in us)}", expanded=False):
        st.caption(srv.APPROVAL_MEANING + " 승인: 입력 정보와 정합한 AI 문장 / 제외: 제품 정보 왜곡·의미 오류 / 검토 전: 아직 확인하지 않음. "
                   "자동 표시는 참고용이며, 자동 검사가 놓친 오류가 있을 수 있습니다.")
        with st.form(f"review_form_{run['run_id']}_{live}"):
            decisions = {}
            for pl in dict.fromkeys(u["persona_label"] for u in us):
                st.markdown(f"**{pl}**")
                for u in [x for x in us if x["persona_label"] == pl]:
                    c1, c2 = st.columns([3, 2])
                    with c1:
                        st.markdown(f"{u['field_label']} · {u['text']}")
                        note = " · 건강 효과 추론 포함" if u["auto_notes"] else ""
                        st.caption(f"{icon[u['auto_label']]} 자동 표시: {u['auto_label']}{note}")
                    with c2:
                        cur = srv.decision_of(review, u["unit_id"])
                        decisions[u["unit_id"]] = st.radio("검토", srv.DECISIONS, index=srv.DECISIONS.index(cur), horizontal=True,
                                                           key=f"rv_{run['run_id']}_{live}_{u['unit_id']}", label_visibility="collapsed")
            if st.form_submit_button("검토 결과 저장"):
                path = srv.save_review(run, decisions)
                st.session_state[f"review_saved_{run['run_id']}"] = rc.display_path(path)  # [공개본 변경] 저장 경로 표시
                st.rerun()
        if st.session_state.get(f"review_saved_{run['run_id']}"):
            saved_name = st.session_state[f"review_saved_{run['run_id']}"]
            st.success(f"검토 결과를 저장했습니다 ({saved_name}). 원본 실행 기록은 바뀌지 않습니다.")


def render_run(run: dict, live: bool):
    synth_run_note(run)  # STEP 7: 합성 페르소나가 없는 기존 기록에서는 아무것도 표시하지 않음
    if DEMO:
        render_demo_run(run, live)
    else:
        render_run_full(run, live)


def _persona_brief(run: dict, r: dict) -> str:
    """평가한 페르소나 요약 (표시용). 사용자 페르소나는 저장된 속성 스냅샷, 기본 유형은 기존 구매 기준."""
    prof = r.get("profile")
    if prof:
        parts = []
        for key in ("age_band", "situations", "sugar_level", "price_range", "key_criteria"):
            a = prof.get("attrs", {}).get(key)
            if a:
                v = a["value"]
                parts.append(f"{pp.ATTR_LABEL[key].split(' (')[0]}: {' > '.join(v) if key == 'key_criteria' and isinstance(v, list) else (', '.join(v) if isinstance(v, list) else v)}")
        return " · ".join(parts) or "속성 없음"
    base = {p["persona_id"]: p for p in personas}.get(r["persona_id"])
    return f"중요하게 보는 기준: {' > '.join(base['criteria_ranked'])}" if base else ""


def render_demo_run(run: dict, live: bool):
    """PPT 시연 화면 (STEP 6.3). build_report / effective_view / 문장 검토 결과를 그대로 사용하고 표시만 간결하게 함.
    사람이 승인한 문장만 보고서에 쓰며, 승인 문장이 없으면 내용을 채우지 않음."""
    c = run["concept"]
    review = srv.load_review(run)
    rep_ = prpt.build_report(run, review)
    rc = rep_["review_counts"]
    if not live:
        st.caption(f"저장된 평가 기록 · {run['created_at']}")

    st.markdown("### 1. 이번에 검토한 신제품 콘셉트")
    sugar = "미기재" if c["sugar_g"] is None else ("0g (무설탕)" if c["sugar_g"] == 0 else f"{c['sugar_g']:g}g")
    st.markdown(f"**{c['name']}**  \n{c['category']} · {c['volume_ml']}mL · {c['price_krw']:,}원 · 당류 {sugar}  \n"
                f"특징: {', '.join(c['features']) or '-'}" + (f"  \n{c['description']}" if c.get("description") else ""))

    st.markdown("### 2. 평가에 참여한 AI 페르소나")
    cols = st.columns(len(run["results"]))
    for col, r in zip(cols, run["results"]):
        with col:
            with st.container(border=True):
                st.markdown(f"**{r['persona_label']}**")
                st.caption(_persona_brief(run, r))

    st.markdown("### 3. 소비자별로 좋게 본 점과 망설인 점")
    st.caption(f"사람이 내용을 확인한 AI 문장만 정리했습니다 (확인 완료 {rc['승인']}개 · 확인 전 {rc['검토 전']}개 · 제외 {rc['제외']}개).")
    if rep_["per_persona"]:
        cols = st.columns(len(rep_["per_persona"]))
        for col, p in zip(cols, rep_["per_persona"]):
            with col:
                with st.container(border=True):
                    st.markdown(f"**{p['소비자']}**")
                    st.markdown("좋게 본 점\n" + ("\n".join(f"- {x}" for x in p["positives"]) or "- (확인된 문장 없음)"))
                    st.markdown("망설인 점\n" + ("\n".join(f"- {x}" for x in p["barriers"]) or "- (확인된 문장 없음)"))
    else:
        st.markdown('<div class="demo-card demo-muted">아직 사람이 확인한 문장이 없어 정리할 내용이 없습니다. '
                    '아래 \'품질 검토 상세\'에서 문장을 확인할 수 있습니다.</div>', unsafe_allow_html=True)

    st.markdown("### 4. 공통 관심사와 소비자별 차이")
    if rep_["comparison_possible"]:
        st.markdown("\n".join(f"- 공통: {x}" for x in rep_["common_issues"]) or "- 공통으로 언급된 관심사 없음")
        st.markdown("\n".join(f"- 차이: {x}" for x in rep_["differences"]) or "- 소비자별 차이 없음")
    else:
        st.markdown('<div class="demo-card demo-muted">확인된 문장이 있는 소비자가 2명 미만이라 비교하지 않았습니다.</div>',
                    unsafe_allow_html=True)

    st.markdown("### 5. 후속 소비자 조사에서 확인할 가설과 질문")
    if rep_["hypotheses"]:
        st.table(pd.DataFrame([{"확인할 가설": h["가설"], "근거가 된 AI 문장": h["근거 문장(AI)"]} for h in rep_["hypotheses"]]))
    else:
        st.markdown('<div class="demo-card demo-muted">확인된 문장이 없어 가설을 만들지 않았습니다.</div>', unsafe_allow_html=True)
    questions = [x for x in rep_["next_steps"] if not any(k in x for k in ("검토 전 문장", "승인된 문장", "확인한 문장"))]
    st.markdown("\n".join(f"- {x}" for x in questions))

    # 품질 검토 상세: 기존 검증 화면 전체를 그대로 (중첩 펼침을 피하려고 토글 사용)
    st.divider()
    qp = prpt.quality_panel(run, review)
    detail_key = f"demo_detail_{run['run_id']}_{live}"
    shown_key = f"_demo_shown_run_{live}"
    if st.session_state.get(shown_key) != run["run_id"]:  # STEP 6.4: 다른 기록으로 바뀌면 상세 보기는 꺼진 상태로 시작
        st.session_state[detail_key] = False
        st.session_state[shown_key] = run["run_id"]
    if st.toggle(f"품질 검토 상세 보기 — 응답 {qp['n']}개 중 자동 검사 '검토 필요' "
                 f"{sum(x['응답 단위 자동 검사'] == '검토 필요' for x in qp['rows'])}개 · 문장 확인 전 {rc['검토 전']}개",
                 key=detail_key):  # 기본값은 꺼짐 (상태는 위에서 세션으로 관리)
        st.markdown(DETAIL_TITLE)
        render_run_full(run, live, from_detail=True)


def render_run_full(run: dict, live: bool, from_detail: bool = False):
    if DEMO and not from_detail:  # STEP 6.4: 시연 모드 기본 화면에서는 검증용 상세(경고·상태·원문·카드)를 그리지 않음
        return
    c = run["concept"]
    if not live:
        st.info(f"저장된 기록입니다. 지금 새로 생성된 결과가 아닙니다. (생성 시각 {run['created_at']}, run_id `{run['run_id']}`)")
    st.subheader(f"{c['name']}")
    sugar = "미기재" if c["sugar_g"] is None else f"{c['sugar_g']:g}g"
    unit = round(c["price_krw"] / c["volume_ml"] * 100)
    st.caption(f"{c['category']} · 1개 {c['volume_ml']}mL · 판매가격 {c['price_krw']:,}원 (참고: 100mL당 {unit:,}원) · "
               f"당류 {sugar} · 특징: {', '.join(c['features']) or '-'}"
               + (f" · 설명: {c['description']}" if c.get("description") else ""))
    badge()
    st.caption(f"모델 {run['model']} (로컬) · 프롬프트 {run['prompt_version']} · 생성 {run['created_at']} · "
               "각 소비자 유형은 서로 독립된 호출로 생성됨 · 가상 페르소나이며 실제 소비자 세그먼트가 아님")

    review = srv.load_review(run)
    render_planning_report(run, review)
    render_quality(prpt.quality_panel(run, review))
    render_sentence_review(run, review, live)
    st.divider()
    views = [effective_view(run, r) for r in run["results"]]
    n_fail = sum(v.get("status") == STATUS_GEN_FAIL for v in views)
    n_schema = sum(v.get("status") == STATUS_SCHEMA_FAIL for v in views)
    n_flag = sum(is_flagged(v) for v in views) - n_fail - n_schema
    n_ok = len(views) - n_flag - n_fail - n_schema
    st.markdown("#### 소비자 유형별 요약")
    st.markdown(f"✅ 자동 검사 통과 **{n_ok}** · ⚠ 검토 필요 **{n_flag}** · ⛔ 스키마 실패 **{n_schema}** · "
                f"❌ 생성 실패 **{n_fail}**")
    st.caption(f"검사 기준 {CHECKS_VERSION} · 자동 검사는 의미 정합성을 보장하지 않음 · "
               "표는 첫 항목 요약이며 전체 의견과 검토 사유는 아래 카드에서 확인")
    st.table(summary_table(run, views))

    st.markdown("#### 소비자 유형별 상세 의견")
    cols = st.columns(len(run["results"]))  # 좁은 화면에서는 자동으로 세로 배치
    for col, r, v in zip(cols, run["results"], views):
        with col:
            with st.container(border=True):
                st.markdown(f"##### {r['persona_label']}")
                render_status(v)
                if v.get("rechecked_from"):
                    st.caption(f"현재 검사 기준({CHECKS_VERSION})으로 재검사함 · 생성 당시({v['rechecked_from']}) 상태: {v['stored_status']}")
                if v.get("note"):
                    st.caption(v["note"])
                if v.get("feedback"):
                    render_feedback(v["feedback"])
                with st.expander("원문 (수정되지 않은 LLM 출력)"):
                    st.caption(f"Ollama created_at: {(r.get('ollama') or {}).get('created_at')} · "
                               f"출력 토큰 {(r.get('ollama') or {}).get('eval_count')} · {r.get('wall_s')}s")
                    st.code(r.get("raw_content") or "\n".join(r.get("errors", [])), language="json", wrap_lines=True)

    st.download_button("이 결과 CSV 다운로드", data=rows_to_csv_bytes(run_to_rows(run)),
                       file_name=f"feedback_{run['run_id']}.csv", mime="text/csv", key=f"dl_{run['run_id']}_{live}")


# ---------------------------------------------------------------------------
if DEMO:
    st.title("AI 소비자 페르소나 신제품 콘셉트 검토")
    st.caption(DEMO_LINE)
else:
    st.title("AI 가상 소비자 정성 피드백 (MVP)")
    st.warning(f"**{DISCLAIMER}** — 이 화면의 모든 의견은 로컬 LLM이 가상 소비자 조건으로 생성한 문장이며, "
               "실제 소비자 조사 결과가 아닙니다. 구매의향 점수·순위·구매율은 제공하지 않습니다(현재 모델의 점수 구분력 미확인).")

with st.sidebar:
    st.header("시스템 상태")
    h = engine.health()
    if h["ok"] and h.get("model_available"):
        st.success(f"Ollama {h['version']} 연결됨 · {engine.model}")
    elif h["ok"]:
        st.error(f"Ollama 연결됨, 모델 {engine.model} 없음")
    else:
        st.error("Ollama 서버에 연결할 수 없습니다. Ollama 실행 여부와 OLLAMA_HOST 설정을 확인하세요.")  # [공개본 변경] 안내 문구
    st.caption(f"Ollama 서버 {engine.host} · 외부 LLM API 미사용")  # [공개본 변경] 주소를 OLLAMA_HOST로 바꿀 수 있어 '로컬 전용' 표현 제거
    st.header("가상 소비자 유형")
    for p in personas:
        st.markdown(f"**{p['label']}** — {' > '.join(p['criteria_ranked'])}")
    st.caption("가상으로 설계한 조건이며 실제 조사로 검증된 세그먼트가 아님")

PROVINCES = ["서울", "경기", "인천", "부산", "대구", "광주", "대전", "울산", "세종", "강원", "충청북", "충청남",
             "전북", "전라남", "경상북", "경상남", "제주"]  # 데이터의 실제 시도 표기


def render_nemotron_import():
    """STEP 7: 'NVIDIA 합성 페르소나 불러오기 (실험)'. 선별·변환은 experiments/nemotron_personas 코드 그대로.
    인구통계 속성만 '🧬 NVIDIA 합성 데이터'로, 사용자가 추가한 가정은 '💭 가상 시나리오 가정'으로 구분. 가중치 미적용."""
    st.markdown("#### NVIDIA 합성 페르소나 불러오기 (실험)")
    if st.session_state.get("nemo_msg"):
        st.success(st.session_state.pop("nemo_msg"))
    st.caption(NEMO_NOTICE)
    st.caption(NEMO_ATTRIBUTION)
    if not _nemo_cached():
        st.caption("처음 사용할 때 데이터 전체(약 1.98GB)를 받지 않고 인구통계 열만 원격으로 읽습니다 (약 9MB, 약 1분).")
    with st.form("nemo_form"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("연령 (데이터는 19세 이상)", 19, 99, (20, 39), key="nemo_age")
            sex = st.selectbox("성별", ["무관", "여자", "남자"], key="nemo_sex")
            provs = st.multiselect("지역 (시도, 비우면 전국)", PROVINCES, key="nemo_prov")
        with c2:
            occ_in = st.text_input("직업에 포함할 단어 (쉼표로 구분, 선택)", key="nemo_occ_in", placeholder="예: 사무원, 간호사")
            occ_ex = st.text_input("직업에서 제외할 단어 (쉼표로 구분, 선택)", key="nemo_occ_ex", placeholder="예: 무직")
            n = st.number_input("선별 인원 (최대 3명)", min_value=1, max_value=3, value=3, step=1, key="nemo_n")
            seed = st.number_input("선별 순서 번호 (같은 조건·번호면 같은 결과)", min_value=0, max_value=9999, value=0, step=1, key="nemo_seed")
        go = st.form_submit_button("조건에 맞는 합성 페르소나 찾기")
    if go:
        try:
            ns, nsel, _ = _nemo_modules()
            df = _nemo_demographics()
            t = nsel.Target(age_min=int(age[0]), age_max=int(age[1]), sex=None if sex == "무관" else sex, provinces=list(provs),
                            occupation_include=pp.parse_list(occ_in), occupation_exclude=pp.parse_list(occ_ex))
            picked, diag = nsel.select(df, t, n=int(n), seed=int(seed))
            rows = picked.to_dict("records")
            ids = ns.fetch_text([(int(r["shard"]), int(r["row_group"]), int(r["row_in_group"])) for r in rows], columns=["uuid"]) if rows else {}
            for r in rows:
                r["uuid"] = ids[(int(r["shard"]), int(r["row_group"]), int(r["row_in_group"]))]["uuid"]
            st.session_state["nemo_cands"] = {"rows": rows, "diag": diag}
        except Exception as e:  # noqa: BLE001
            st.session_state.pop("nemo_cands", None)
            st.error(f"합성 페르소나를 불러오지 못했습니다 (네트워크 또는 데이터 접근 문제일 수 있음): {e}")
    cands = st.session_state.get("nemo_cands")
    if not cands:
        return
    d = cands["diag"]
    if not cands["rows"]:
        st.info(f"조건({d['target']})을 만족하는 합성 레코드가 없습니다. 조건을 넓혀 보세요.")
        return
    st.markdown(f"조건: {d['target']} · 조건을 만족하는 합성 레코드 {d['pool']:,}개 중 **{d['selected']}명** 선별")
    st.caption("연령대×성별 칸별로 데이터셋 안의 분포에 비례해 인원을 나누고, 선별 순서 번호로 정렬해 고릅니다. "
               "공식 기준 통계로 보정하는 가중치는 적용하지 않습니다.")
    st.table(pd.DataFrame([{"데이터 ID (uuid)": r["uuid"], "나이": int(r["age"]), "성별": r["sex"], "직업": r["occupation"],
                            "지역": str(r["district"]).replace("-", " "), "가구 형태": r["family_type"], "주거": r["housing_type"]}
                           for r in cands["rows"]]))
    labels_ = {f"{int(r['age'])}세 {r['sex']} · {r['occupation']} · {str(r['district']).replace('-', ' ')} · {r['uuid'][:8]}": r
               for r in cands["rows"]}
    with st.form("nemo_add_form"):
        add_pick = st.multiselect("등록할 페르소나", list(labels_), default=list(labels_), key="nemo_add_pick")
        st.markdown("**사용자가 추가하는 가정 (선택)** — 비워 두면 공란으로 두고 응답자에게 전달하지 않습니다. "
                    f"입력하면 위에서 고른 페르소나 모두에 '{pp.SOURCE_ICON[pp.SRC_ASSUME]} {pp.SRC_ASSUME}'으로 따로 표시됩니다.")
        a1, a2 = st.columns(2)
        with a1:
            as_sit = st.multiselect(pp.ATTR_LABEL["situations"], pp.SITUATIONS, key="nemo_as_situations")
            as_freq = st.selectbox(pp.ATTR_LABEL["frequency"], ["(입력 안 함)"] + pp.FREQUENCY, key="nemo_as_frequency")
        with a2:
            as_price = st.text_input(pp.ATTR_LABEL["price_range"], key="nemo_as_price_range")
            as_sugar = st.selectbox(pp.ATTR_LABEL["sugar_level"], ["(입력 안 함)"] + pp.LEVELS, key="nemo_as_sugar_level")
        add = st.form_submit_button("선택한 페르소나 등록")
    if add:
        _, _, nad = _nemo_modules()
        ua = {"situations": as_sit, "frequency": None if as_freq == "(입력 안 함)" else as_freq,
              "price_range": as_price.strip(), "sugar_level": None if as_sugar == "(입력 안 함)" else as_sugar}
        done = []
        try:
            for lab in add_pick:
                r = labels_[lab]
                prof = nad.to_profile(r, {"uuid": r["uuid"]}, include_narrative=False, user_assumptions=ua)
                prof.created_at = pd.Timestamp.now().isoformat(timespec="seconds")
                pp.save_user_profile(prof)
                done.append(prof.label)
            if done:  # 목록·평가 대상 선택 상자에 바로 반영되도록 다시 그림
                st.session_state["nemo_msg"] = (f"등록했습니다: {', '.join(done)}. "
                                                "위 '등록된 페르소나'와 '새 콘셉트 평가' 탭에서 확인·선택할 수 있습니다.")
                st.rerun()
            else:
                st.info("등록할 페르소나를 선택하세요.")
        except Exception as e:  # noqa: BLE001
            st.error(f"등록하지 못했습니다: {e}")


tab_persona, tab_new, tab_ab, tab_hist, tab_cmp, tab_survey = st.tabs(
    ["페르소나 설정", "새 콘셉트 평가", "목표 기반 A/B 비교", "실행 기록", "콘셉트 비교", "실제 조사 데이터 (준비)"])

with tab_new:
    with st.form("concept_form"):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("제품명", key="name", placeholder="예: 제로 과일 탄산음료")
            category = st.text_input("종류", key="category", placeholder="예: 탄산음료")
            volume = st.number_input("용량 (mL, 1개 기준)", min_value=1, max_value=5000, value=350, step=10, key="volume")
            price = st.number_input("판매가격 (원, 1개 기준)", min_value=1, max_value=100000, value=1800, step=100, key="price")
        with c2:
            sugar_known = st.checkbox("당류 정보 입력", value=True, key="sugar_known")
            sugar = st.number_input("당류 (g)", min_value=0.0, max_value=200.0, value=0.0, step=1.0, key="sugar")
            features = st.text_input("특징 (쉼표로 구분)", key="features", placeholder="예: 과일 향, 강한 청량감")
            description = st.text_area("추가 설명 (선택)", key="description", max_chars=300)
        profiles = pp.all_profiles()
        prof_by_label = {(p.label if p.is_default else f"{p.label} · NVIDIA 합성 {p.profile_id[-4:]}" if p.is_synthetic()
                          else f"{p.label} · 사용자 페르소나 {p.profile_id[-4:]}"): p for p in profiles}
        selected = st.multiselect("평가할 소비자 (기본 3유형 + '페르소나 설정' 탭에서 만들거나 불러온 페르소나)", list(prof_by_label),
                                  default=[l for l, p in prof_by_label.items() if p.is_default], key="personas")
        st.caption("기본 3유형과 직접 만든 페르소나를 함께 평가할 수 있습니다." if DEMO else
                   "사용자 페르소나는 속성을 그대로 응답자 설정으로 전달합니다(프롬프트 qf_v3p, 소규모 생성으로만 확인). "  # [공개본 변경] 실제 상태에 맞게 수정
                   "기본 3유형은 기존 프롬프트(qf_v3)를 그대로 씁니다.")
        submitted = st.form_submit_button("가상 소비자 의견 생성", type="primary")

    if submitted:
        try:
            concept = ProductConcept(name=name.strip(), category=category.strip(), volume_ml=int(volume),
                                     price_krw=int(price), sugar_g=float(sugar) if sugar_known else None,
                                     features=[x for x in features.split(",")], description=description.strip())
        except Exception as e:  # noqa: BLE001
            st.error(f"입력 값을 확인해 주세요: {e}")
            concept = None
        if concept and not selected:
            st.error("소비자 유형을 하나 이상 선택해 주세요.")
        elif concept:
            bar = st.progress(0.0, text="생성 준비 중")

            def prog(i, n, p):
                bar.progress(i / n, text=f"{p['label']} 의견 생성 중 ({i + 1}/{n}) — 로컬 LLM 실제 추론")

            run = engine.run(concept, [pp.as_engine_persona(prof_by_label[l]) for l in selected], progress=prog)
            bar.progress(1.0, text="완료")
            path = save_run(run)
            st.session_state["last_run"] = run
            st.success(f"새로 생성 완료 · 저장: {rc.display_path(path)}")  # [공개본 변경] 저장 경로 표시

    if st.session_state.get("last_run"):
        render_run(st.session_state["last_run"], live=True)

with tab_hist:
    runs = list_runs()
    if not runs:
        st.write("저장된 실행 기록이 없습니다.")
    else:
        labels = {f"{r['created_at']} · {r['concept']['name']} · {len(r['results'])}개 유형 · {r['run_id']}": r for r in runs}
        pick = st.selectbox("실행 기록 선택", list(labels), key="hist_pick")
        render_run(labels[pick], live=False)

def render_compare_full(sel: list):
    """기존 콘셉트 비교 화면 (STEP 6.5에서 함수로 분리, 내용 동일)."""
    st.info("저장된 기록을 비교합니다. 각 콘셉트는 서로 독립적으로 평가되었으며, 콘셉트 간 우열이나 점수를 나타내지 않습니다. "
            f"'⚠ 검토 필요'·'⛔ 스키마 실패'·'❌ 생성 실패'는 자동 검사({CHECKS_VERSION}) 결과입니다.")
    badge()
    all_personas = []
    for r in sel:
        for x in r["results"]:
            if x["persona_label"] not in all_personas:
                all_personas.append(x["persona_label"])
    for p in all_personas:
        st.markdown(f"#### {p}")
        cols = st.columns(len(sel))  # 좁은 화면에서는 자동으로 세로 배치
        for col, r in zip(cols, sel):
            with col:
                with st.container(border=True):
                    st.markdown(f"**{r['concept']['name']}** ({r['concept']['price_krw']:,}원) · `{r['run_id'][-6:]}`")
                    x = {y["persona_label"]: y for y in r["results"]}.get(p)
                    if not x:
                        st.caption("이 기록에는 해당 유형이 없습니다.")
                        continue
                    v = effective_view(r, x)
                    if is_flagged(v) or v.get("status") == STATUS_GEN_FAIL:
                        st.markdown(f"**{status_label(v)}** — 검토 사유는 실행 기록 탭의 카드에서 확인")
                    else:
                        st.markdown(f"**{status_label(v)}**")
                    fb = v.get("feedback") or {}
                    for key, title in SECTIONS:
                        st.markdown(f"**{title}**")
                        st.markdown("\n".join(f"- {y}" for y in fb.get(key, [])) or "—")
    rows = [row for r in sel for row in run_to_rows(r)]
    st.download_button("비교 결과 CSV 다운로드", data=rows_to_csv_bytes(rows), file_name="feedback_compare.csv",
                       mime="text/csv", key="dl_cmp")


def render_compare_demo(sel: list):
    """STEP 6.5: 콘셉트 비교 시연 화면 — 사람이 승인한 문장만, 상태 배지 없음. 승인 문장이 없으면 채우지 않음."""
    st.caption("각 콘셉트는 서로 독립적으로 평가되었으며, 콘셉트 간 우열이나 점수를 나타내지 않습니다. 사람이 확인한 AI 문장만 표시합니다.")
    approved = {r["run_id"]: srv.approved_units(r) for r in sel}
    all_personas = list(dict.fromkeys(x["persona_label"] for r in sel for x in r["results"]))
    for p in all_personas:
        st.markdown(f"#### {p}")
        cols = st.columns(len(sel))
        for col, r in zip(cols, sel):
            with col:
                with st.container(border=True):
                    st.markdown(f"**{r['concept']['name']}** ({r['concept']['price_krw']:,}원)")
                    us = [u for u in approved[r["run_id"]] if u["persona_label"] == p]
                    pos = [u["text"] for u in us if u["field"] == "positive_points"]
                    con = [u["text"] for u in us if u["field"] == "concerns"]
                    if not pos and not con:
                        st.caption("사람이 확인한 문장이 아직 없습니다.")
                        continue
                    st.markdown("좋게 본 점\n" + ("\n".join(f"- {x}" for x in pos) or "- (확인된 문장 없음)"))
                    st.markdown("망설인 점\n" + ("\n".join(f"- {x}" for x in con) or "- (확인된 문장 없음)"))


with tab_cmp:
    runs = list_runs()
    labels = {f"{r['created_at']} · {r['concept']['name']} · {r['run_id']}": r for r in runs}
    picks = st.multiselect("비교할 실행 기록 (2~3개)", list(labels), max_selections=3, key="cmp_pick")
    if len(picks) >= 2 and DEMO:
        render_compare_demo([labels[k] for k in picks])
        st.divider()
        if st.toggle("품질 검토 상세 보기 — 자동 검사 상태와 검토 전 문장을 포함한 원래 비교 화면", key="demo_detail_cmp"):
            st.markdown(DETAIL_TITLE)
            render_compare_full([labels[k] for k in picks])
    elif len(picks) >= 2:
        render_compare_full([labels[k] for k in picks])
    else:
        st.write("두 개 이상의 실행 기록을 선택하세요.")


# ---------------------------------------------------------------------------
# STEP 5: 목표 기반 A/B 콘셉트 비교
# ---------------------------------------------------------------------------
def render_conclusion(run: dict, live: bool, key_suffix: str = "", neutral: bool | None = None):
    """결과 화면 맨 위 (STEP 5.2): ① 질문 ② 조건 기반 우선 검증 후보 ③ AI 평가 신뢰 상태
    ④ 실제 소비자 선호 판단 가능 여부 ⑤ 후속 조사 질문. 규칙 기반이며 LLM 호출 없음."""
    c = ab.planning_conclusion(run)
    g, t = run["goal"], run["target"]
    a, b = run["products"]["A"], run["products"]["B"]
    st.markdown("### 상품기획 결론")

    # ① 질문
    st.markdown("**① 상품기획 질문**")
    st.markdown(f"> {c['question']}")
    traits = t.get("traits", "")
    st.caption(f"타깃(가상 설정): {ab.Target(**{k: t[k] for k in ('age_band', 'gender', 'situations', 'persona_ids')}, traits=traits).describe()}"
               f" · 구매 상황: {', '.join(t['situations']) or '특정하지 않음'}"
               f" · 추가 특성: {traits or '없음'}")
    if not traits and ab.trait_hint(c["question"]):
        nbox(st.warning, f"질문에 타깃 특성으로 보이는 표현({', '.join(ab.trait_hint(c['question']))})이 있지만, 이 실행에서는 **추가 특성 필드가 비어 있어 "
                   "AI 응답자에게 전달되지 않았습니다**" + (" (ab_v1 기록)" if run["prompt_version"] == "ab_v1" else "") + ".", neutral)

    # ② 조건 기반 우선 검증 후보
    st.markdown("**② 조건 기반 우선 검증 후보** — 판단 기준 + 제품 조건만 사용 (AI 응답 미사용)")
    stored = g.get("criterion", ab.CRITERION_NONE)
    crit = st.selectbox("판단 기준", ab.CRITERIA, index=ab.CRITERIA.index(stored) if stored in ab.CRITERIA else 0,
                        key=f"crit_{run['ab_run_id']}_{live}{key_suffix}")
    if crit != stored:
        st.caption(f"결과 보기용으로 선택한 기준입니다 (실행 시 저장된 기준: {stored}). 실행 기록은 바뀌지 않습니다.")
    hints = [h for h in ab.criterion_hints(c["question"], traits) if h != crit]
    if crit == ab.CRITERION_NONE and hints:
        st.caption(f"질문·추가 특성에 '{hints[0]}'과 관련된 표현이 있습니다. 해당 목표라면 판단 기준을 직접 선택하세요 (자동 적용하지 않음).")
    prop = ab.condition_proposal(a, b, crit)
    if prop["derivable"]:
        nbox(st.success, f"**우선 검증 후보: 제품 {prop['candidate']}** — {prop['reason']}", neutral)
        for x in prop["tradeoffs"]:
            st.markdown(f"- ⚖ {x}")
    else:
        nbox(st.info, f"**우선 검증 후보 없음** — {prop['reason']}", neutral)
    st.caption("이 제안은 상품기획자가 정한 기준을 제품 조건에 적용한 것일 뿐이며, 실제 소비자 선호를 입증하거나 판매량을 예측한 것이 아닙니다.")
    with st.expander("제품 조건 차이 (입력값에서 계산)"):
        st.table(pd.DataFrame(c["objective_diffs"]).set_index("항목").rename_axis(None))

    # ③ AI 페르소나 평가 신뢰 상태
    st.markdown("**③ AI 페르소나 평가 결과의 신뢰 상태** — 모델이 생성한 의견")
    _fn = {ab.CONCLUSION_HOLD: st.warning, ab.CONCLUSION_REVIEW: st.info, ab.CONCLUSION_LIMITED: st.info}[c["status"]]
    box = lambda text: nbox(_fn, text, neutral)  # noqa: E731
    icon = {ab.CONCLUSION_HOLD: "⏸", ab.CONCLUSION_REVIEW: "🔍", ab.CONCLUSION_LIMITED: "🧭"}[c["status"]]
    box(f"**{icon} {c['status']}** — 비교 가능 유형 {c['n_comparable']}/{c['n_personas']} (A·B 모두 자동 검사 통과 기준). "
        "AI 의견으로 제품을 추천하지 않습니다.")
    with st.expander("판단 근거와 AI 의견 요약"):
        st.markdown("\n".join(f"- {x}" for x in c["reasons"]))
        if c["ai_notes"]:
            st.markdown("AI 가상 의견 요약 (참고용 · 근거 아님)")
            st.table(pd.DataFrame(c["ai_notes"]).set_index("소비자 유형").rename_axis(None))

    # ④ 실제 소비자 선호·판매 판단 가능 여부
    st.markdown("**④ 실제 소비자 선호·판매 판단 가능 여부** — 실제 소비자 조사 결과")
    nbox(st.error, f"**판단 불가** — {c['headline']} 실측 소비자 데이터가 없어 선호·구매율·판매량을 판단하거나 예측하지 않습니다.", neutral)

    # ⑤ 후속 조사 질문
    st.markdown("**⑤ 후속 실제 조사에서 확인할 질문**")
    follow = list(c["followups"])
    if prop["derivable"]:
        x = f"'{crit}' 기준의 후보 {prop['candidate']}가 해당 목표를 가진 실제 타깃 소비자에게 실제로 더 선호되는지 확인"
        if prop["tradeoffs"]:
            x += " (기준 외 조건 차이, 예: 가격이 수용되는지 포함)"
        follow.insert(1, x)
    st.markdown("\n".join(f"- {x}" for x in follow))


def render_ab_quality(run: dict):
    """A/B 시뮬레이션 품질 정보 (STEP 6)."""
    rows = []
    for r in run["results"]:
        v = ab.ab_view(run["products"][r["side"]], r)
        s_ = v.get("status", "")
        flags = v.get("flags", [])
        rows.append({"소비자": f"{r['persona_label']} · 제품 {r['side']}",
                     "JSON 검증": "실패(응답 없음)" if s_ == STATUS_GEN_FAIL else "실패" if s_ == STATUS_SCHEMA_FAIL else "통과",
                     **prpt.flag_counts(flags if v.get("feedback") else [], v.get("notes")),
                     "의미 오류 검토": "미검토 (사람 검토 기록 없음)",
                     "분석 근거 사용": "사용 (내용 미검증)" if ab.is_usable(v) else "제외"})
    qp = {"rows": rows, "n": len(rows), "n_json_ok": sum(x["JSON 검증"] == "통과" for x in rows),
          "n_distortion": sum(x["제품 정보 왜곡 의심"] > 0 for x in rows), "n_thin": sum(x["근거 부족"] > 0 for x in rows),
          "n_usable": sum(x["분석 근거 사용"].startswith("사용") for x in rows), "checks_version": CHECKS_VERSION,
          "known_limits": prpt.KNOWN_LIMITS}
    render_quality(qp)


def render_ab(run: dict, live: bool):
    if DEMO:
        render_ab_demo(run, live)
    else:
        render_ab_full(run, live)


def render_ab_demo(run: dict, live: bool):
    """STEP 6.5: A/B 시연 화면 — 결론(중립 카드)과 제품 조건만. 소비자별 AI 카드·원문·품질 표·자동 검사 기반 분석은 상세 보기로."""
    if not live:
        st.caption(f"저장된 A/B 평가 기록 · {run['created_at']}")
    render_conclusion(run, live)
    st.markdown("#### 비교한 두 제품의 조건")
    st.table(pd.DataFrame(run["diff"]).set_index("항목").rename_axis(None))
    st.caption("제품별 AI 의견과 품질 검토 결과는 아래 '품질 검토 상세 보기'에서 확인할 수 있습니다.")
    st.divider()
    dkey = f"demo_detail_ab_{run['ab_run_id']}_{live}"
    shown = f"_demo_shown_ab_{live}"
    if st.session_state.get(shown) != run["ab_run_id"]:
        st.session_state[dkey] = False
        st.session_state[shown] = run["ab_run_id"]
    n_flag = sum(ab.ab_view(run["products"][r["side"]], r).get("status", "").startswith("검토 필요") for r in run["results"])
    if st.toggle(f"품질 검토 상세 보기 — 응답 {len(run['results'])}개 중 자동 검사 '검토 필요' {n_flag}개", key=dkey):
        st.markdown(DETAIL_TITLE)
        render_ab_full(run, live, from_detail=True)


def render_ab_full(run: dict, live: bool, from_detail: bool = False):
    if not live:
        st.info(f"저장된 A/B 기록입니다. 지금 새로 생성된 결과가 아닙니다. (생성 시각 {run['created_at']}, `{run['ab_run_id']}`)")
    g, t = run["goal"], run["target"]
    tgt = ab.Target(**{k: t[k] for k in ("age_band", "gender", "situations", "persona_ids")}, traits=t.get("traits", ""))
    render_conclusion(run, live, key_suffix="_detail" if from_detail else "", neutral=False if from_detail else None)
    render_ab_quality(run)
    st.divider()
    st.markdown("#### 1. 검증 목표")
    st.markdown(f"**질문** {g['question']}")
    st.markdown(f"**비교 대상** A: {run['products']['A']['name']} / B: {run['products']['B']['name']}  \n"
                f"**확인할 관점** {', '.join(g['aspects'])}")
    st.markdown(f"**타깃(가상 설정)** {tgt.describe()} · 구매 상황: {', '.join(t['situations']) or '특정하지 않음'} · "
                f"추가 특성: {t.get('traits') or '없음'} · 소비 성향: {', '.join(persona_by_id[p]['label'] for p in t['persona_ids'])}")
    st.caption(f"{ab.TARGET_NOTE} · 연령·성별만으로 선호를 가정하지 않도록 프롬프트에 명시함 · {run['evaluation_design']} · "
               f"모델 {run['model']} · 프롬프트 {run['prompt_version']}")
    badge()
    st.warning("이 화면은 구매의향 점수·구매율·판매량을 예측하지 않으며, 제품의 우열을 판정하지 않습니다. "
               "AI 가상 의견을 바탕으로 **실제 조사에서 확인할 점**을 정리하는 용도입니다.")

    st.markdown("#### 2. 제품 A/B 조건")
    st.table(pd.DataFrame(run["diff"]).set_index("항목").rename_axis(None))

    st.markdown("#### 3. 소비자 유형별 A/B 의견")
    views = {(r["persona_label"], r["side"]): (r, ab.ab_view(run["products"][r["side"]], r)) for r in run["results"]}
    personas_in_run = list(dict.fromkeys(r["persona_label"] for r in run["results"]))
    for side in ("A", "B"):
        vs = [v for (pl, sd), (_, v) in views.items() if sd == side]
        st.caption(f"제품 {side}: ✅ 통과 {sum(ab.is_usable(v) for v in vs)} · "
                   f"⚠ 검토 필요 {sum(v.get('status','').startswith('검토 필요') and v.get('status') != STATUS_SCHEMA_FAIL for v in vs)} · "
                   f"⛔ 스키마 실패 {sum(v.get('status') == STATUS_SCHEMA_FAIL for v in vs)} · "
                   f"❌ 생성 실패 {sum(v.get('status') == STATUS_GEN_FAIL for v in vs)}")
    for pl in personas_in_run:
        st.markdown(f"##### {pl}")
        cols = st.columns(2)
        for col, side in zip(cols, ("A", "B")):
            with col:
                with st.container(border=True):
                    st.markdown(f"**제품 {side} · {run['products'][side]['name']}**")
                    if (pl, side) not in views:
                        st.caption("응답 없음")
                        continue
                    r, v = views[(pl, side)]
                    render_status(v)
                    if v.get("note"):
                        st.caption(v["note"])
                    if v.get("feedback"):
                        render_feedback(v["feedback"])
                    with st.expander("원문 (수정되지 않은 LLM 출력)"):
                        st.code(r.get("raw_content") or "\n".join(r.get("errors", [])), language="json", wrap_lines=True)

    st.markdown("#### 4. 공통 관심사와 서로 다른 의견")
    inc = st.checkbox("검토 필요 응답도 참고용으로 포함해 보기 (분석 근거 아님)", value=False,
                      key=f"ab_inc_{run['ab_run_id']}_{live}")
    an = ab.analyze(run, include_flagged=inc)
    st.caption(f"근거로 쓸 수 있는 응답(자동 검사 통과) {an['n_usable_responses']}/{an['n_responses']} · "
               f"A·B 모두 비교 가능한 유형 {an['n_comparable']}/{len(an['per_persona'])} · "
               "응답 문장에 언급된 관점(맛·향, 가격 등)을 키워드로 분류한 것이며 선호나 우열을 뜻하지 않음")
    rows = []
    for e in an["per_persona"]:
        if not e["comparable"]:
            rows.append({"소비자 유형": e["persona"], "구분": "비교 제외",
                         "공통": f"A: {e['status']['A']} / B: {e['status']['B']}", "A에서만": "—", "B에서만": "—"})
            continue
        for pol in ("긍정", "우려"):
            rows.append({"소비자 유형": e["persona"], "구분": pol, "공통": ", ".join(e[f"{pol}_공통"]) or "—",
                         "A에서만": ", ".join(e[f"{pol}_A만"]) or "—", "B에서만": ", ".join(e[f"{pol}_B만"]) or "—"})
    st.table(pd.DataFrame(rows).set_index("소비자 유형").rename_axis(None))
    if inc:
        st.warning("검토 필요 응답이 포함된 참고 화면입니다. 이 표를 근거로 인용하지 마세요.")

    st.markdown("#### 5. 후속 실제 조사 제안")
    st.error("아래 가설과 문항 후보는 **AI 가상 응답에서 규칙 기반으로 도출**한 것이며, 실제 소비자 데이터로 검증되지 않았습니다.")
    an_base = ab.analyze(run, include_flagged=False)  # 가설은 항상 '자동 검사 통과' 응답만 근거로 사용
    hyps = ab.hypotheses(run, an_base)
    st.markdown("**검증할 가설 (AI 도출 · 미검증)**")
    st.table(pd.DataFrame(hyps) if hyps else pd.DataFrame([{"가설": "근거로 쓸 수 있는 응답이 없어 가설을 도출하지 않음"}]))
    st.markdown("**실제 설문 문항 후보**")
    surv = ab.survey_candidates(run, hyps)
    st.table(pd.DataFrame(surv))

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("A/B 의견 CSV 다운로드", data=ab.to_csv_bytes(ab.ab_rows(run), ab.AB_CSV_COLUMNS),
                           file_name=f"ab_feedback_{run['ab_run_id']}.csv", mime="text/csv",
                           key=f"dl_ab_{run['ab_run_id']}_{live}")
    with c2:
        plan = [{"유형": "가설", **h, "응답 형식": "", "비고": ""} for h in hyps] + \
               [{"유형": "설문 문항", "가설": r_["문항"], "근거": "", "출처": r_["구분"], "상태": "AI 도출 · 미검증",
                 "응답 형식": r_["응답 형식"], "비고": r_["비고"]} for r_ in surv]
        st.download_button("가설·설문 후보 CSV 다운로드",
                           data=ab.to_csv_bytes(plan, ["유형", "가설", "근거", "출처", "상태", "응답 형식", "비고"]),
                           file_name=f"ab_survey_plan_{run['ab_run_id']}.csv", mime="text/csv",
                           key=f"dl_abplan_{run['ab_run_id']}_{live}")


persona_by_id = {p["persona_id"]: p for p in personas}

with tab_ab:
    st.markdown("##### 1. 검증 목표 설정")
    ab_q = st.text_area("상품기획 질문", key="ab_question", max_chars=300,
                        value="20대 여성 소비자를 대상으로 제로 복숭아 탄산음료와 제로 레몬 탄산음료 중 어떤 콘셉트를 우선 검증해야 할까?")
    ab_crit = st.selectbox("조건 기반 판단 기준 (선택)", ab.CRITERIA, index=0, key="ab_criterion",
                           help="상품기획 목표를 기준으로 제품 조건만 비교해 우선 검증 후보를 제안합니다. AI 응답은 쓰지 않습니다.")
    ab_aspects = st.multiselect("확인할 관점", ab.ASPECT_CHOICES, default=["맛·향", "가격", "당류·건강", "청량감"],
                                key="ab_aspects")
    st.caption("질문은 결과 화면과 기록에만 쓰이며, AI 응답자에게는 보여 주지 않습니다 (독립 평가 유지).")

    st.markdown("##### 2. 타깃 소비자 설정 (가상 설정)")
    t1, t2 = st.columns(2)
    with t1:
        ab_age = st.selectbox("연령대", ab.AGE_BANDS, index=1, key="ab_age")
        ab_gender = st.selectbox("성별", ab.GENDERS, index=0, key="ab_gender")
    with t2:
        ab_sit = st.multiselect("구매 상황", ab.SITUATIONS, default=[ab.SITUATIONS[0]], key="ab_situations")
        ab_personas = st.multiselect("소비 성향 (페르소나)", [p["label"] for p in personas],
                                     default=[p["label"] for p in personas], key="ab_personas")
    ab_traits = st.text_input("추가 특성 (A/B 양쪽 응답자에게 동일하게 전달)", key="ab_traits", max_chars=100,
                              placeholder="예: 당류 섭취를 줄이려고 함")
    st.caption(f"{ab.TARGET_NOTE}. 연령·성별만으로 특정 맛을 선호한다고 가정하지 않도록 지시합니다. "
               "질문 문장은 AI 응답자에게 전달되지 않으므로, 타깃 특성은 '추가 특성'에 입력해야 반영됩니다.")
    _h = ab.criterion_hints(ab_q, ab_traits)
    if _h and ab_crit == ab.CRITERION_NONE:
        st.caption(f"참고: 질문·추가 특성에 '{_h[0]}'과 관련된 표현이 있습니다. 해당 목표라면 판단 기준을 선택하세요 (자동 적용하지 않음).")

    st.markdown("##### 3. 제품 A/B 입력")
    defaults = {"A": ("제로 복숭아 탄산음료", "복숭아 향, 강한 청량감"), "B": ("제로 레몬 탄산음료", "레몬 향, 강한 청량감")}
    inputs = {}
    pc = st.columns(2)
    for col, side in zip(pc, ("A", "B")):
        with col:
            with st.container(border=True):
                st.markdown(f"**제품 {side}**")
                inputs[side] = dict(
                    name=st.text_input("제품명", value=defaults[side][0], key=f"ab_{side}_name"),
                    category=st.text_input("종류", value="탄산음료", key=f"ab_{side}_category"),
                    volume=st.number_input("용량 (mL, 1개 기준)", 1, 5000, 350, 10, key=f"ab_{side}_volume"),
                    price=st.number_input("판매가격 (원, 1개 기준)", 1, 100000, 1800, 100, key=f"ab_{side}_price"),
                    sugar_known=st.checkbox("당류 정보 입력", value=True, key=f"ab_{side}_sugar_known"),
                    sugar=st.number_input("당류 (g)", 0.0, 200.0, 0.0, 1.0, key=f"ab_{side}_sugar"),
                    features=st.text_input("특징 (쉼표로 구분)", value=defaults[side][1], key=f"ab_{side}_features"),
                    description=st.text_area("추가 설명 (선택)", key=f"ab_{side}_description", max_chars=300))
    concepts, err = {}, None
    for side, v in inputs.items():
        try:
            concepts[side] = ProductConcept(name=v["name"].strip(), category=v["category"].strip(), volume_ml=int(v["volume"]),
                                            price_krw=int(v["price"]), sugar_g=float(v["sugar"]) if v["sugar_known"] else None,
                                            features=v["features"].split(","), description=v["description"].strip())
        except Exception as e:  # noqa: BLE001
            err = f"제품 {side} 입력 값을 확인해 주세요: {e}"
    if err:
        st.error(err)
    else:
        st.markdown("**공통 조건과 차이**")
        st.table(pd.DataFrame(ab.concept_diff(concepts["A"], concepts["B"])).set_index("항목").rename_axis(None))

    n_calls = len(ab_personas) * 2
    st.markdown("##### 4. AI 페르소나 시뮬레이션")
    st.caption(f"선택한 소비 성향 {len(ab_personas)}개 × 제품 2개 = LLM 호출 {n_calls}건. 각 제품을 따로 제시해 평가합니다.")
    if st.button("A/B 가상 소비자 의견 생성", type="primary", key="ab_run_btn"):
        if err:
            st.error("제품 입력을 먼저 고쳐 주세요.")
        elif not ab_personas or not ab_aspects or not ab_q.strip():
            st.error("질문, 확인할 관점, 소비 성향을 하나 이상 입력해 주세요.")
        else:
            goal = ab.Goal(question=ab_q.strip(), aspects=ab_aspects, criterion=ab_crit)
            label_to_id = {p["label"]: p["persona_id"] for p in personas}
            target = ab.Target(age_band=ab_age, gender=ab_gender, situations=ab_sit,
                               persona_ids=[label_to_id[l] for l in ab_personas], traits=ab_traits.strip())
            bar = st.progress(0.0, text="생성 준비 중")

            def ab_prog(i, n, p, side):
                bar.progress(i / n, text=f"{p['label']} · 제품 {side} 의견 생성 중 ({i + 1}/{n}) — 로컬 LLM 실제 추론")

            run = ab.run_ab(engine, goal, target, concepts["A"], concepts["B"], persona_by_id, progress=ab_prog)
            bar.progress(1.0, text="완료")
            path = ab.save_ab_run(run)
            st.session_state["last_ab_run"] = run
            st.success(f"새로 생성 완료 · 저장: {rc.display_path(path)}")  # [공개본 변경] 저장 경로 표시
    if st.session_state.get("last_ab_run"):
        render_ab(st.session_state["last_ab_run"], live=True)

    st.divider()
    st.markdown("##### 저장된 A/B 기록")
    ab_runs = ab.list_ab_runs()
    if not ab_runs:
        st.write("저장된 A/B 기록이 없습니다.")
    else:
        ab_labels = {f"{r['created_at']} · A {r['products']['A']['name']} / B {r['products']['B']['name']} · {r['ab_run_id']}": r
                     for r in ab_runs}
        ab_pick = st.selectbox("A/B 기록 선택", ["(선택 안 함)"] + list(ab_labels), key="ab_hist_pick")
        if ab_pick != "(선택 안 함)":
            render_ab(ab_labels[ab_pick], live=False)


# ---------------------------------------------------------------------------
# STEP 6: 페르소나 설정 탭
# ---------------------------------------------------------------------------
with tab_persona:
    st.markdown("### 페르소나 설정")
    st.info(f"{pp.NOT_REPRESENTATIVE} 속성마다 출처를 표시합니다: "
            f"{pp.SOURCE_ICON[pp.SRC_USER]} 사용자 입력 · {pp.SOURCE_ICON[pp.SRC_ASSUME]} 가상 시나리오 가정 · "
            f"{pp.SOURCE_ICON[pp.SRC_SURVEY]} 실제 조사 근거(근거 입력 필수) · "
            f"{pp.SOURCE_ICON[pp.SRC_SYNTH]} NVIDIA 합성 데이터(인구통계 합성 프로필). 현재 프로젝트에는 실제 조사 데이터가 없습니다.")
    st.markdown("#### 등록된 페르소나")
    for prof in pp.all_profiles():
        cnt = prof.source_counts()
        kind_ = "기본 유형" if prof.is_default else "NVIDIA 합성 페르소나 (실험)" if prof.is_synthetic() else "사용자 페르소나"
        title = f"{prof.label} — {kind_} · " + \
                " · ".join(f"{pp.SOURCE_ICON[k]} {v}" for k, v in cnt.items())
        with st.expander(title):
            st.table(pd.DataFrame([{"속성": pp.ATTR_LABEL[k], "값": a.text(ordered=(k == "key_criteria")), "출처": f"{pp.SOURCE_ICON[a.source]} {a.source}",
                                    "근거": a.evidence or "-"} for k, a in prof.attrs.items()]).set_index("속성").rename_axis(None))
            if DEMO:
                st.caption("기본 유형 (가상 설계)" if prof.is_default else f"만든 시각 {prof.created_at[:16]}")
            else:
                st.caption(("기본 유형: data/personas_v2.json(프로젝트에서 설계한 가상 조건)을 읽어 표시 · 평가 시 기존 프롬프트 사용"  # [공개본 변경] 내부 단계 번호 제거
                            if prof.is_default else f"ID {prof.profile_id} · 만든 시각 {prof.created_at} · 평가 시 프롬프트 qf_v3p"))
            if prof.is_synthetic():
                st.caption(NEMO_NOTICE)
                st.caption(NEMO_ATTRIBUTION)

    st.markdown("#### 새 페르소나 만들기")
    st.caption("필요한 속성만 입력하세요. 비워 둔 속성은 응답자에게 전달되지 않습니다.")
    base_opts = ["(처음부터)"] + [p.label for p in pp.default_profiles()]
    base_pick = st.selectbox("기본 유형에서 시작 (선택 시 구매 기준·구매 습관만 '가상 시나리오 가정'으로 채움)", base_opts,
                             key="pf_base")
    base_prof = next((p for p in pp.default_profiles() if p.label == base_pick), None)
    with st.form("persona_form"):
        pf_label = st.text_input("페르소나 이름", key="pf_label", max_chars=40, placeholder="예: 운동 후 제로음료를 찾는 직장인 (가상)")
        inputs = {}
        for key, label, _pl, kind, opts in pp.ATTR_SPEC:
            pre = base_prof.attrs.get(key) if (base_prof and key in ("key_criteria", "habits")) else None  # 인구통계는 채우지 않음
            c1, c2, c3 = st.columns([3, 1.3, 1.7])
            with c1:
                if kind == "select":
                    val = st.selectbox(label, ["(입력 안 함)"] + opts, key=f"pf_{key}")
                    val = None if val == "(입력 안 함)" else val
                elif kind == "multi":
                    val = st.multiselect(label, opts, key=f"pf_{key}")
                elif kind == "list":
                    val = pp.parse_list(st.text_input(label, value=(", ".join(pre.value) if pre else ""), key=f"pf_{key}"))
                else:
                    val = st.text_input(label, value=(pre.text() if pre else ""), key=f"pf_{key}").strip()
            with c2:
                src = st.selectbox("출처", pp.SOURCES, index=1 if pre else 0, key=f"pf_{key}_src")
            with c3:
                ev = st.text_input("근거 (실제 조사일 때 필수)", key=f"pf_{key}_ev")
            inputs[key] = (val, src, ev)
        pf_submit = st.form_submit_button("페르소나 저장")
    if pf_submit:
        try:
            attrs = {k: pp.Attr(value=v, source=src, evidence=ev) for k, (v, src, ev) in inputs.items() if v}
            prof = pp.PersonaProfile(profile_id=pp.new_profile_id(), label=pf_label.strip(),
                                     base_type=base_prof.profile_id if base_prof else None, attrs=attrs,
                                     created_at=pd.Timestamp.now().isoformat(timespec="seconds"))
            path = pp.save_user_profile(prof)
            st.success(f"저장했습니다: {prof.label} ({path.name}). '새 콘셉트 평가' 탭에서 선택할 수 있습니다.")
        except Exception as e:  # noqa: BLE001
            st.error(f"저장하지 못했습니다: {e}")

    render_nemotron_import()


# ---------------------------------------------------------------------------
# STEP 6: 실제 조사 데이터 탭 (형식 검증만, 분석·비교는 계획 단계)
# ---------------------------------------------------------------------------
with tab_survey:
    st.markdown("### 실제 소비자 조사 데이터 (준비 단계)")
    if DEMO:
        st.caption("실제 소비자 조사 결과를 연결하기 위한 준비 화면입니다 (현재는 입력 형식 검증만 제공).")
    else:
        st.warning("현재 프로젝트에는 실제 조사 데이터가 없습니다. 이 화면은 **입력 형식 준비와 형식 검증**만 제공하며, "
                   "업로드한 데이터를 저장하거나 AI 결과와 비교·보정하지 않습니다 (계획 단계).")
    st.markdown("**입력 필드**")
    st.table(pd.DataFrame([{"컬럼": c_, "필수": "필수" if req else "선택", "설명": d_} for c_, req, d_ in sdata.FIELDS])
             .set_index("컬럼").rename_axis(None))
    st.download_button("빈 입력 템플릿 CSV 다운로드", data=sdata.template_csv_bytes(), file_name="survey_template.csv",
                       mime="text/csv", key="dl_survey_tpl")
    up = st.file_uploader("조사 결과 CSV 형식 검증 (UTF-8)", type=["csv"], key="survey_upload")
    if up is not None:
        res_ = sdata.validate_csv(up.getvalue())
        (st.success if res_["ok"] else st.error)(
            f"형식 검증 {'통과' if res_['ok'] else '실패'} — 유효한 행 {res_['n_rows']}개, 응답자 {res_.get('n_respondents', 0)}명, "
            f"제품 {', '.join(res_.get('products', [])) or '-'}")
        for e_ in res_["errors"]:
            st.markdown(f"- {e_}")
        st.caption("검증 결과만 보여 줍니다. 데이터는 저장하지 않습니다.")
    st.markdown("**실측 데이터 활용 절차 (계획)**")
    st.markdown("- 조사 전에 비교 지표·성공 기준·페르소나·프롬프트 버전을 고정\n"
                "- 실측 데이터를 **보정용과 검증용으로 분리**(응답자 단위, 가능하면 제품 단위도)\n"
                "- 보정 전 AI 결과와 실측을 비교 → 보정용 데이터로만 조정 → 검증용 데이터로 확인\n"
                "- 검증을 통과한 지표만 화면에 표시 (상세: docs/survey_data_spec.md, README 'Future Work')")  # [공개본 변경] 공개본에 없는 개발 보고서 경로 제거
