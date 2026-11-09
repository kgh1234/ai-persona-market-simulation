# 원본 대비 변경 내역 (CHANGES_FROM_ORIGINAL)

이 저장소는 비공개 개발 환경에서 실제로 실행한 프로젝트 가운데, 최종 Streamlit 서비스 실행에 필요한 파일만 골라 만든 공개용 복사본입니다.
원본 코드와 다른 부분을 아래에 모두 기록합니다. 코드 안의 변경 위치에는 `[공개본 변경]` 주석을 달았습니다.

## 1. 변경 목적

| 목적 | 내용 |
|---|---|
| 다른 컴퓨터에서 실행 | 원본은 Ollama 주소가 개발 서버 전용 포트(`127.0.0.1:11435`)로 고정되어 있었습니다. 공개본은 `OLLAMA_HOST` 환경변수로 지정하고, 지정하지 않으면 Ollama 기본 주소 `127.0.0.1:11434`를 씁니다. |
| 소스와 실행 결과 분리 | 원본은 실행 결과를 `results/step4/runs`, `results/step5/ab_runs`, `results/reviews`, `data/persona_profiles_user`, `experiments/nemotron_personas/cache`에 나눠 저장했습니다. 공개본은 모두 `outputs/` 아래에 저장하며, `APP_OUTPUT_DIR`로 바꿀 수 있습니다. |
| 빈 환경에서 시작 | 앱이 시작할 때 `outputs/` 하위 폴더를 만들고, NVIDIA 데이터 캐시 폴더는 상위 폴더까지 함께 만듭니다. |
| 화면 문구 정리 | 공개본에 없는 개발 보고서 경로와 내부 단계 번호(STEP)를 화면 문구에서 뺐습니다. 실제 상태와 맞지 않는 문구도 고쳤습니다. 수치와 의미는 바꾸지 않았습니다. |

**바꾸지 않은 것:** 프롬프트(`qf_v3`, `qf_v3p`, `ab_v2`), 응답 스키마, 자동 품질 점검(`checks_v5`), 문장 검토, 보고서 생성 로직, 화면 구성, 모델과 생성 옵션(`qwen2.5:7b`, temperature 0.7)은 원본과 같습니다.

**확인 방법**

1. **전체 기록 비교:** 개발 환경에 저장된 전체 기록(콘셉트 평가 20건(합성 페르소나 실험 2건 포함), A/B 비교 17건, 검토 2건)에 원본 모듈과 공개본 모듈을 각각 적용했습니다.
   - 비교한 결과: 프롬프트, 품질 점검 결과, 문장 단위 자동 표시, 보고서, A/B 결론
   - 차이는 화면 문구 정리로 바뀐 주의사항 문장과 검토 파일 표시 경로뿐이었습니다.
2. **기능 테스트 재실행:** 오프라인 기능 테스트를 다시 실행해 모두 통과했습니다. 테스트에서는 저장된 LLM 응답을 재생했고, 외부 네트워크는 차단했습니다.

## 2. 파일별 상태

### 원본과 동일 (sha256 일치)

| 파일 |
|---|
| `.streamlit/config.toml` |
| `data/personas_v2.json` |
| `src/survey_data.py` |
| `experiments/nemotron_personas/selection.py` |

### 원본에서 수정

| 파일 | 변경 요약 |
|---|---|
| `app.py` | `runtime_config` import, 시작할 때 `outputs/` 폴더 생성, NVIDIA 캐시 위치, 저장 경로 표시 문구, Ollama 연결 실패 안내 문구, 사이드바 서버 표시('로컬 전용' → '외부 LLM API 미사용'). 화면 문구 3곳도 고쳤습니다: 사용자 페르소나 프롬프트 상태('실제 생성 검증 전' → '소규모 생성으로만 확인'), 기본 유형 설명의 내부 단계 번호 제거, 조사 데이터 탭의 개발 보고서 경로 제거. 모듈 설명 주석도 수정했습니다. |
| `src/qualitative_feedback.py` | `DEFAULT_HOST` → `OLLAMA_HOST` 환경변수(기본 11434), `RUNS_DIR` → `outputs/runs`, 공개본에 없는 프롬프트 기록 파일을 가리키던 주석 수정 |
| `src/ab_comparison.py` | `AB_RUNS_DIR` → `outputs/ab_runs`, 공개본에 없는 프롬프트 기록 파일을 가리키던 주석 수정 |
| `src/persona_profiles.py` | `USER_DIR` → `outputs/personas`. 모듈 설명 주석(저장 경로, `qf_v3p` 확인 상태) 수정 |
| `src/sentence_review.py` | `REVIEW_DIR` → `outputs/reviews`, 주석의 저장 경로 |
| `src/planning_report.py` | 검토 파일 경로 표시를 `relative_to` 대신 `display_path`로 처리(결과 폴더가 프로젝트 밖이어도 오류가 나지 않게). '시뮬레이션 품질 정보'의 주의사항 문장에서 내부 단계 번호와 개발 보고서 경로 제거(수치는 같음) |
| `experiments/nemotron_personas/nemotron_source.py` | 캐시와 `HF_HOME`을 `outputs/cache/` 아래로 옮김, `mkdir(parents=True)` |
| `experiments/nemotron_personas/adapter.py` | **주석만 수정**(출처 표기와 저장 주체 설명을 실제 코드 동작에 맞게 고침). 코드 변경 없음 |

### 원본 문서에서 가져와 수정

| 파일 | 변경 요약 |
|---|---|
| `docs/survey_data_spec.md` | 조사 데이터 탭이 참조하는 입력 명세입니다. 제목의 내부 단계 번호를 빼고, 공개본에 없는 템플릿 파일 경로를 앱의 템플릿 다운로드 버튼 안내로 바꿨습니다. 필드와 검증 규칙은 원본과 같고, `src/survey_data.py`와 일치함을 확인했습니다. |

### 공개본에서 새로 추가

| 파일 | 내용 |
|---|---|
| `src/runtime_config.py` | 환경변수 `OLLAMA_HOST`, `APP_OUTPUT_DIR` 해석과 결과 폴더 경로 |
| `data/examples/product_concepts.json` | 입력 예시용 가상 제품 콘셉트. 원본 실행 기록과 A/B 탭 기본값에서 실제로 쓴 콘셉트이며, 앱이 자동으로 읽지 않음 |
| `examples/sample_outputs/runs/20261104_023549_56a74f.json`, `examples/sample_outputs/reviews/20261104_023549_56a74f.json` | 원본 실행 기록과 문장 검토 기록을 **내용 변경 없이** 복사 (sha256이 원본과 같음, 검토 기록 안의 실행 기록 해시와도 일치) |
| `requirements.txt` | 실행에 필요한 Python 패키지 (개발 환경 검증 버전) |
| `.gitignore`, `README.md`, `NOTICE.md`, `docs/CHANGES_FROM_ORIGINAL.md`, `docs/images/.gitkeep` | 공개용 문서와 설정 |
| `.gitattributes` | `examples/sample_outputs/**`에 Git 줄바꿈 변환을 적용하지 않음 (샘플의 sha256 유지) |

## 3. 공개본에 포함하지 않은 원본 구성요소

| 구성요소 | 제외 이유 |
|---|---|
| 이전 버전 엔진과 실험 코드(`persona_evaluator*.py`, `paired_comparison.py`, `ab_prompt_en.py`, 단계별 테스트 스크립트, 이전 버전 보관본) | 최종 앱이 import하지 않는 과거 실험 코드 |
| 이전 프롬프트 기록 (`prompts/*.json`) | 최종 앱은 코드 안의 프롬프트를 쓰며, 이 파일들을 읽지 않음 |
| 단계별 개발 보고서 (`docs/step*_report.md`) | 개발 일지이며 개발 서버 정보가 들어 있음. 공개에 필요한 내용(설계 근거, 향후 검증 절차)은 README에 요약 |
| 실행 기록, 로그, 검토 결과, 사용자 페르소나 | 개발 중 생성된 결과물. 단, 검토를 마친 실행 기록 1건과 그 검토 기록은 `examples/sample_outputs/`에 샘플로 포함 |
| 개발 서버 운영 스크립트와 설정 | 개발 서버의 경로, GPU, 포트에 의존 |
| 가상환경, Ollama 실행 파일, 모델 가중치 | 대용량이며 재배포 대상이 아님 |
| NVIDIA 데이터셋 캐시, 외부 저장소 사본 (TinyTroupe, 참고 저장소) | 원본 데이터와 외부 코드는 재배포하지 않음 ([NOTICE.md](../NOTICE.md) 참고) |
| TinyTroupe 설정 (`config.ini`) | 최종 서비스에서 TinyTroupe를 쓰지 않음 |

## 4. 전체 diff

경로 앞의 `original/`은 개발 환경 원본, `public/`은 이 저장소를 뜻합니다.

### 4.1 코드

```diff
--- original/app.py
+++ public/app.py
@@ -1,8 +1,9 @@
 """AI 가상 소비자 정성 피드백 MVP (STEP 4) + 목표 기반 A/B 콘셉트 비교 (STEP 5)
 + 구조화 페르소나·상품기획 보고서·품질 정보·실제 조사 CSV 형식 검증 (STEP 6).
 
-실행: scripts/start_ollama.sh && scripts/start_app.sh   (127.0.0.1:8501 전용)
-- 로컬 Ollama(qwen2.5:7b)만 사용. 외부 API 없음. TinyTroupe 미사용.
+실행: OLLAMA_HOST=127.0.0.1:11434 streamlit run app.py   (127.0.0.1:8501 전용, README 참고)
+- LLM은 로컬 Ollama(qwen2.5:7b)만 사용. 외부 LLM API 없음. TinyTroupe 미사용.
+- [공개본 변경] Ollama 주소는 OLLAMA_HOST, 실행 결과 폴더는 APP_OUTPUT_DIR(기본 outputs/)로 지정 (src/runtime_config.py).
 - 정량 점수·순위·구매율 그래프는 제공하지 않음 (STEP 3C에서 구분력 미확인).
 """
 import sys
@@ -21,6 +22,9 @@
 from qualitative_feedback import (CHECKS_VERSION, DISCLAIMER, FeedbackEngine, ProductConcept,  # noqa: E402
                                   STATUS_GEN_FAIL, STATUS_SCHEMA_FAIL, effective_view, list_runs, load_personas, run_to_rows, rows_to_csv_bytes,
                                   save_run)
+import runtime_config as rc  # noqa: E402  [공개본 변경]
+
+rc.ensure_output_dirs()  # [공개본 변경] 새 환경에서도 outputs/ 하위 폴더를 만들고 시작
 
 st.set_page_config(page_title="AI 가상 소비자 정성 피드백 (MVP)", layout="wide")
 
@@ -87,7 +91,7 @@
 
 
 def _nemo_cached() -> bool:
-    return all((NEMO_DIR / "cache" / f"demo_shard{i}.parquet").exists() for i in range(9))
+    return all((rc.NEMOTRON_CACHE_DIR / f"demo_shard{i}.parquet").exists() for i in range(9))  # [공개본 변경] 캐시 위치
 
 
 def run_has_synthetic(run: dict) -> bool:
@@ -251,7 +255,7 @@
 
 
 def render_sentence_review(run: dict, review: dict, live: bool):
-    """문장(항목)별 검토 (STEP 6.2). 결과는 results/reviews/<run_id>.json 에 저장, 원본 실행 JSON은 수정하지 않음."""
+    """문장(항목)별 검토 (STEP 6.2). 결과는 outputs/reviews/<run_id>.json 에 저장 [공개본 변경: 경로 표기], 원본 실행 JSON은 수정하지 않음."""
     us = srv.units(run)
     if not us:
         return
@@ -277,11 +281,11 @@
                                                            key=f"rv_{run['run_id']}_{live}_{u['unit_id']}", label_visibility="collapsed")
             if st.form_submit_button("검토 결과 저장"):
                 path = srv.save_review(run, decisions)
-                st.session_state[f"review_saved_{run['run_id']}"] = str(path.name)
+                st.session_state[f"review_saved_{run['run_id']}"] = rc.display_path(path)  # [공개본 변경] 저장 경로 표시
                 st.rerun()
         if st.session_state.get(f"review_saved_{run['run_id']}"):
             saved_name = st.session_state[f"review_saved_{run['run_id']}"]
-            st.success(f"검토 결과를 저장했습니다 (results/reviews/{saved_name}). 원본 실행 기록은 바뀌지 않습니다.")
+            st.success(f"검토 결과를 저장했습니다 ({saved_name}). 원본 실행 기록은 바뀌지 않습니다.")
 
 
 def render_run(run: dict, live: bool):
@@ -447,8 +451,8 @@
     elif h["ok"]:
         st.error(f"Ollama 연결됨, 모델 {engine.model} 없음")
     else:
-        st.error("Ollama 서버에 연결할 수 없습니다. scripts/start_ollama.sh 실행 필요")
-    st.caption(f"서버 {engine.host} (로컬 전용, 외부 API 미사용)")
+        st.error("Ollama 서버에 연결할 수 없습니다. Ollama 실행 여부와 OLLAMA_HOST 설정을 확인하세요.")  # [공개본 변경] 안내 문구
+    st.caption(f"Ollama 서버 {engine.host} · 외부 LLM API 미사용")  # [공개본 변경] 주소를 OLLAMA_HOST로 바꿀 수 있어 '로컬 전용' 표현 제거
     st.header("가상 소비자 유형")
     for p in personas:
         st.markdown(f"**{p['label']}** — {' > '.join(p['criteria_ranked'])}")
@@ -566,7 +570,7 @@
         selected = st.multiselect("평가할 소비자 (기본 3유형 + '페르소나 설정' 탭에서 만들거나 불러온 페르소나)", list(prof_by_label),
                                   default=[l for l, p in prof_by_label.items() if p.is_default], key="personas")
         st.caption("기본 3유형과 직접 만든 페르소나를 함께 평가할 수 있습니다." if DEMO else
-                   "사용자 페르소나는 속성을 그대로 응답자 설정으로 전달합니다(프롬프트 qf_v3p, 실제 생성 검증 전). "
+                   "사용자 페르소나는 속성을 그대로 응답자 설정으로 전달합니다(프롬프트 qf_v3p, 소규모 생성으로만 확인). "  # [공개본 변경] 실제 상태에 맞게 수정
                    "기본 3유형은 기존 프롬프트(qf_v3)를 그대로 씁니다.")
         submitted = st.form_submit_button("가상 소비자 의견 생성", type="primary")
 
@@ -590,7 +594,7 @@
             bar.progress(1.0, text="완료")
             path = save_run(run)
             st.session_state["last_run"] = run
-            st.success(f"새로 생성 완료 · 저장: results/step4/runs/{path.name}")
+            st.success(f"새로 생성 완료 · 저장: {rc.display_path(path)}")  # [공개본 변경] 저장 경로 표시
 
     if st.session_state.get("last_run"):
         render_run(st.session_state["last_run"], live=True)
@@ -972,7 +976,7 @@
             bar.progress(1.0, text="완료")
             path = ab.save_ab_run(run)
             st.session_state["last_ab_run"] = run
-            st.success(f"새로 생성 완료 · 저장: results/step5/ab_runs/{path.name}")
+            st.success(f"새로 생성 완료 · 저장: {rc.display_path(path)}")  # [공개본 변경] 저장 경로 표시
     if st.session_state.get("last_ab_run"):
         render_ab(st.session_state["last_ab_run"], live=True)
 
@@ -1010,7 +1014,7 @@
             if DEMO:
                 st.caption("기본 유형 (가상 설계)" if prof.is_default else f"만든 시각 {prof.created_at[:16]}")
             else:
-                st.caption(("기본 유형: data/personas_v2.json(STEP 3C 가상 설계)을 읽어 표시 · 평가 시 기존 프롬프트 사용"
+                st.caption(("기본 유형: data/personas_v2.json(프로젝트에서 설계한 가상 조건)을 읽어 표시 · 평가 시 기존 프롬프트 사용"  # [공개본 변경] 내부 단계 번호 제거
                             if prof.is_default else f"ID {prof.profile_id} · 만든 시각 {prof.created_at} · 평가 시 프롬프트 qf_v3p"))
             if prof.is_synthetic():
                 st.caption(NEMO_NOTICE)
@@ -1086,4 +1090,4 @@
     st.markdown("- 조사 전에 비교 지표·성공 기준·페르소나·프롬프트 버전을 고정\n"
                 "- 실측 데이터를 **보정용과 검증용으로 분리**(응답자 단위, 가능하면 제품 단위도)\n"
                 "- 보정 전 AI 결과와 실측을 비교 → 보정용 데이터로만 조정 → 검증용 데이터로 확인\n"
-                "- 검증을 통과한 지표만 화면에 표시 (상세: docs/step5_goal_based_ab_report.md 6절, docs/survey_data_spec.md)")
+                "- 검증을 통과한 지표만 화면에 표시 (상세: docs/survey_data_spec.md, README 'Future Work')")  # [공개본 변경] 공개본에 없는 개발 보고서 경로 제거
--- original/src/qualitative_feedback.py
+++ public/src/qualitative_feedback.py
@@ -22,14 +22,16 @@
 import requests
 from pydantic import BaseModel, Field, ValidationError, field_validator
 
+import runtime_config as rc  # [공개본 변경] Ollama 주소·결과 폴더를 환경변수로 지정
+
 ROOT = Path(__file__).resolve().parents[1]
 PERSONA_FILE = ROOT / "data" / "personas_v2.json"
-RUNS_DIR = ROOT / "results" / "step4" / "runs"
+RUNS_DIR = rc.RUNS_DIR  # [공개본 변경] 원본: results/step4/runs
 DISCLAIMER = "AI 생성 가상 의견 · 실측 미검증"
-PROMPT_VERSION = "qf_v3"  # qf_v1: STEP 4, qf_v2: STEP 4.5 가격 표기 분리, qf_v3: STEP 4.6 특징 분리·문항 역할 명확화 (prompts/qf_v*.json)
+PROMPT_VERSION = "qf_v3"  # qf_v1: STEP 4, qf_v2: STEP 4.5 가격 표기 분리, qf_v3: STEP 4.6 특징 분리·문항 역할 명확화 (이전 버전 프롬프트 기록은 공개본에 미포함)
 # checks_v1: STEP 4 첫 테스트, v2: 100mL당 가격 오인 검사, v3: STEP 4.5 입력에 없는 특징·강도 불일치·의견 모순 검사
 CHECKS_VERSION = "checks_v5"  # v4: STEP 4.6 절 단위 판정 등, v5: STEP 6.1 과일명 강도 왜곡·건강 효과 단정/추론 구분
-DEFAULT_HOST = "http://127.0.0.1:11435"
+DEFAULT_HOST = rc.OLLAMA_HOST  # [공개본 변경] 원본: 고정값 http://127.0.0.1:11435 → OLLAMA_HOST (없으면 11434)
 DEFAULT_MODEL = "qwen2.5:7b"
 OPTIONS = {"temperature": 0.7, "num_predict": 768}
 
--- original/src/ab_comparison.py
+++ public/src/ab_comparison.py
@@ -23,9 +23,11 @@
 from qualitative_feedback import (DISCLAIMER, OPTIONS, ProductConcept, QualitativeFeedback, STATUS_GEN_FAIL,
                                   STATUS_SCHEMA_FAIL, effective_view)
 
+import runtime_config as rc  # noqa: E402  [공개본 변경]
+
 ROOT = Path(__file__).resolve().parents[1]
-AB_RUNS_DIR = ROOT / "results" / "step5" / "ab_runs"
-AB_PROMPT_VERSION = "ab_v2"  # ab_v1(prompts/ab_v1.json) + 공통 타깃 추가 특성 줄 (STEP 5.2)
+AB_RUNS_DIR = rc.AB_RUNS_DIR  # [공개본 변경] 원본: results/step5/ab_runs
+AB_PROMPT_VERSION = "ab_v2"  # ab_v1 + 공통 타깃 추가 특성 줄 (STEP 5.2). 이전 버전 프롬프트 기록은 공개본에 미포함
 TARGET_NOTE = "가상 설정 · 실제 조사 근거 없음"
 
 AGE_BANDS = ["10대 후반", "20대", "30대", "40대", "50대 이상"]
--- original/src/persona_profiles.py
+++ public/src/persona_profiles.py
@@ -4,8 +4,9 @@
   '실제 조사 근거'는 근거(조사명·일자 등)를 반드시 적어야 하며, 현재 프로젝트에는 실제 조사 데이터가 없다.
 - 기본 3유형(가격·건강·맛 중심형)은 기존 data/personas_v2.json 을 읽어 보여 주기만 한다(파일 수정 없음).
   기본 유형으로 평가할 때는 기존 프롬프트(qf_v3)를 그대로 쓴다 → 기존 기능 동작 불변.
-- 사용자가 만든 프로필은 data/persona_profiles_user/<profile_id>.json 에 저장하고,
-  평가 시 별도 프롬프트(qf_v3p)로 속성을 전달한다. qf_v3p 는 실제 생성으로 검증하지 않았다.
+- 사용자가 만든 프로필은 outputs/personas/<profile_id>.json 에 저장하고,
+  평가 시 별도 프롬프트(qf_v3p)로 속성을 전달한다. qf_v3p 는 소규모 생성(응답 20건)으로만 확인했다.
+  [공개본 변경: 설명을 현재 상태에 맞게 갱신]
 - 생성한 페르소나는 가상 설정이며 특정 인구 집단을 대표하지 않는다.
 """
 from __future__ import annotations
@@ -19,9 +20,11 @@
 
 from pydantic import BaseModel, Field, model_validator
 
+import runtime_config as rc  # [공개본 변경]
+
 ROOT = Path(__file__).resolve().parents[1]
 BASE_PERSONA_FILE = ROOT / "data" / "personas_v2.json"
-USER_DIR = ROOT / "data" / "persona_profiles_user"  # 테스트에서 다른 경로로 바꿀 수 있도록 함수 안에서 매번 읽음
+USER_DIR = rc.USER_PERSONA_DIR  # [공개본 변경] 원본: data/persona_profiles_user. 테스트에서 바꿀 수 있도록 함수 안에서 매번 읽음
 PROFILE_PROMPT_VERSION = "qf_v3p"
 NOT_REPRESENTATIVE = "가상으로 설정한 소비자이며, 특정 연령·성별 집단의 대표 특성이 아닙니다."
 
--- original/src/planning_report.py
+++ public/src/planning_report.py
@@ -11,14 +11,14 @@
 from ab_comparison import ASPECT_PATTERNS, aspects_of
 from qualitative_feedback import (CHECKS_VERSION, STATUS_GEN_FAIL, STATUS_SCHEMA_FAIL, effective_view)
 
-# STEP 5.3·5.4 실험에서 확인된 주의사항 (보고서 근거 파일 명시)
+# 개발 중 내부 점검에서 확인된 주의사항 (화면의 '시뮬레이션 품질 정보'에 표시)
+# [공개본 변경] 내부 단계 번호와 공개본에 없는 개발 보고서 경로를 문구에서 제거 (내용·수치는 원본과 같음)
 KNOWN_LIMITS = [
-    "같은 입력으로 다시 실행해도 결과가 크게 달라질 수 있습니다 (STEP 5.3: 같은 조건 5회 실행 시 자동 검사 통과 0~5/6, "
-    "수동 검토 오류 1/6~5/6 — docs/step5_3_prompt_language_report.md).",
-    "자동 검사를 통과한 응답에서도 수동 검토 시 내용 오류가 확인되었습니다 (STEP 5.3 3건, STEP 5.4 5/7건).",
-    "두 제품을 한 번에 보여 주고 고르게 하면 먼저 제시된 제품을 고르는 경향이 컸습니다 (STEP 5.4: 7/8). "
-    "그래서 이 서비스는 제품을 하나씩 따로 평가하며, 직접 선택 결과는 제공하지 않습니다 "
-    "(docs/step5_4_paired_comparison_report.md).",
+    "같은 입력으로 다시 실행해도 결과가 크게 달라질 수 있습니다 (개발 중 점검: 같은 조건 5회 실행 시 자동 검사 통과 0~5/6, "
+    "수동 검토 오류 1/6~5/6).",
+    "자동 검사를 통과한 응답에서도 수동 검토 시 내용 오류가 확인되었습니다 (개발 중 점검: 3건, 5/7건).",
+    "두 제품을 한 번에 보여 주고 고르게 하면 먼저 제시된 제품을 고르는 경향이 컸습니다 (개발 중 점검: 7/8). "
+    "그래서 이 서비스는 제품을 하나씩 따로 평가하며, 직접 선택 결과는 제공하지 않습니다.",
 ]
 
 
@@ -181,7 +181,9 @@
         nxt.insert(0, f"검토 전 문장 {counts[sr.PENDING]}개가 있습니다. '문장별 검토'에서 승인·제외를 정하면 보고서에 반영됩니다.")
     if len(per) < 2:
         nxt.insert(0, f"승인된 문장이 있는 소비자 유형이 {len(per)}개뿐이라 유형 간 비교를 하지 않았습니다.")
-    return {"review_counts": counts, "review_file": str(sr.review_path(run["run_id"]).relative_to(sr.ROOT)),
+    # [공개본 변경] 결과 폴더가 프로젝트 밖(APP_OUTPUT_DIR)이어도 오류가 나지 않도록 표시용 경로 함수 사용
+    import runtime_config as rc
+    return {"review_counts": counts, "review_file": rc.display_path(sr.review_path(run["run_id"])),
             "concept_summary": concept_summary(c), "per_persona": per, "excluded": excluded,
             "common_issues": common, "differences": diffs, "hypotheses": hyps, "next_steps": nxt,
             "n_usable": len(per), "n_total": len(run["results"]), "comparison_possible": len(per) >= 2,
--- original/src/sentence_review.py
+++ public/src/sentence_review.py
@@ -6,7 +6,7 @@
   짧지만 입력값과 일치하는 표현('무설탕', '강한 청량감')은 사실 오류로 분류하지 않음
 - 사람 검토: 승인 / 제외 / 검토 전 (기본값 '검토 전', 자동 승인 없음)
   '승인'은 AI가 만든 문장이 입력 정보와 정합한지 사람이 확인했다는 뜻이며, 실제 소비자 의견이 아님
-- 저장: results/reviews/<run_id>.json (원본 실행 JSON은 수정하지 않음). 문장 원문을 함께 저장해,
+- 저장: outputs/reviews/<run_id>.json (원본 실행 JSON은 수정하지 않음). 문장 원문을 함께 저장해,
   원문이 달라지면 해당 검토는 적용하지 않음
 """
 from __future__ import annotations
@@ -17,10 +17,11 @@
 from datetime import datetime
 from pathlib import Path
 
+import runtime_config as rc  # [공개본 변경]
 from qualitative_feedback import ProductConcept, QualitativeFeedback, effective_view, run_checks
 
 ROOT = Path(__file__).resolve().parents[1]
-REVIEW_DIR = ROOT / "results" / "reviews"  # 테스트에서 바꿀 수 있도록 함수 안에서 매번 읽음
+REVIEW_DIR = rc.REVIEW_DIR  # [공개본 변경] 원본: results/reviews. 테스트에서 바꿀 수 있도록 함수 안에서 매번 읽음
 APPROVED, EXCLUDED, PENDING = "승인", "제외", "검토 전"
 DECISIONS = [PENDING, APPROVED, EXCLUDED]
 APPROVAL_MEANING = ("'승인'은 AI가 생성한 문장이 입력한 제품 정보와 맞는지 사람이 확인했다는 뜻입니다. "
--- original/experiments/nemotron_personas/nemotron_source.py
+++ public/experiments/nemotron_personas/nemotron_source.py
@@ -7,17 +7,23 @@
   2) 선별된 행이 속한 행 그룹에서만 uuid·서사 열을 읽음
 - 접근 방식은 Power-xc/power-persona-sim 의 'remote / shard' 경로 아이디어를 참고해 새로 작성(코드 복사 아님).
   (참고 저장소는 DuckDB httpfs 사용, 여기서는 기존 환경에 있는 pyarrow + huggingface_hub 만 사용)
-- 캐시: experiments/nemotron_personas/cache/ (HF_HOME 도 이 폴더 아래로 고정, 홈 디렉터리에 쓰지 않음)
+- 캐시: outputs/cache/nemotron/ (HF_HOME 도 outputs/cache/hf_home 으로 고정, 홈 디렉터리에 쓰지 않음)
 """
 from __future__ import annotations
 
 import os
+import sys
 import time
 from pathlib import Path
 
 HERE = Path(__file__).resolve().parent
-CACHE = HERE / "cache"
-os.environ.setdefault("HF_HOME", str(HERE / "hf_home"))
+# [공개본 변경] 캐시를 소스 폴더가 아닌 실행 결과 폴더(outputs/)에 저장. 원본: HERE/"cache", HERE/"hf_home"
+if str(HERE.parents[1] / "src") not in sys.path:
+    sys.path.insert(0, str(HERE.parents[1] / "src"))
+import runtime_config as rc  # noqa: E402
+
+CACHE = rc.NEMOTRON_CACHE_DIR
+os.environ.setdefault("HF_HOME", str(rc.HF_HOME_DIR))
 os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
 
 import pyarrow as pa  # noqa: E402
@@ -50,7 +56,7 @@
 
 def load_demographics(shards=range(N_SHARDS)) -> pa.Table:
     """인구통계 열 + 위치(shard, row_group, row_in_group). 샤드별로 캐시."""
-    CACHE.mkdir(exist_ok=True)
+    CACHE.mkdir(parents=True, exist_ok=True)  # [공개본 변경] parents=True
     tables = []
     for s in shards:
         cp = CACHE / f"demo_shard{s}.parquet"
@@ -78,7 +84,7 @@
 
 def fetch_text(locs: list[tuple[int, int, int]], columns=TEXT_COLS) -> dict[tuple[int, int, int], dict]:
     """선별된 위치의 서사 열만 읽음. 같은 행 그룹은 한 번만 읽고, 필요한 행만 캐시에 남김."""
-    CACHE.mkdir(exist_ok=True)
+    CACHE.mkdir(parents=True, exist_ok=True)  # [공개본 변경] parents=True
     out: dict = {}
     by_group: dict = {}
     for s, rg, i in locs:
--- original/experiments/nemotron_personas/adapter.py
+++ public/experiments/nemotron_personas/adapter.py
@@ -1,13 +1,14 @@
 """Nemotron 레코드 → 기존 구조화 페르소나(src/persona_profiles.PersonaProfile) 변환 어댑터 (실험 모듈).
 
 원칙
-- 기존 모듈은 import 만 하고 수정하지 않는다. 변환 결과는 data/persona_profiles_user 에 저장하지 않는다
-  (저장하면 운영 Streamlit 화면의 페르소나 목록에 나타나기 때문).
+- 기존 모듈은 import 만 하고 수정하지 않는다. 이 모듈은 변환만 하고 파일을 저장하지 않는다
+  (저장은 app.py 가 사용자가 '등록'을 누를 때 persona_profiles.save_user_profile 로 수행).
 - 데이터에 있는 속성만 채운다: 연령대·성별·직업/생활(직업·지역·가구·주거). 음료 구매 빈도·가격 허용 범위·
   선호 음료·당류 고려·신제품 수용도 등은 Nemotron 에 없으므로 비워 둔다 (임의로 채우지 않음).
-- 출처 표시: 기존 형식의 출처는 3종(사용자 입력 / 가상 시나리오 가정 / 실제 조사 근거)뿐이다.
-  합성 데이터는 '실제 조사 근거'가 아니므로 '가상 시나리오 가정'으로 두고, evidence 에 Nemotron uuid·필드명과
+- 출처 표시: 데이터에서 온 속성은 '실제 조사 근거'가 아니므로 별도 출처 'NVIDIA 합성 데이터'(SRC_SYNTH)로 두고,
+  사용자가 추가한 가정은 '가상 시나리오 가정'(SRC_ASSUME)으로 구분한다. evidence 에 Nemotron uuid·필드명과
   '공개 통계 분포 기반 합성 · 개인 실측 아님'을 적는다. (evidence 는 프롬프트로 전달되지 않음)
+  [공개본 변경: 설명을 실제 코드 동작에 맞게 갱신, 코드 변경 없음]
 - 서사(culinary_persona 등)는 gemma 모델이 생성한 문장이다. include_narrative=True 일 때만 '평소 구매 습관'
   자리에 넣고 출처를 'LLM 생성 서사'로 적는다. 기본값은 False.
 """
```

### 4.2 문서 (`docs/survey_data_spec.md`)

```diff
--- original/docs/survey_data_spec.md
+++ public/docs/survey_data_spec.md
@@ -1,12 +1,17 @@
-# 실제 소비자 조사 데이터 입력 명세 (STEP 6, 설계 단계)
+# 실제 소비자 조사 데이터 입력 명세 (준비 단계)
 
-- 목적: 후속 실제 설문 결과를 CSV로 받아, 이후 단계에서 AI 가상 의견과 비교·검증하기 위한 공통 형식
-- 현재 구현 범위: 필드 정의(`src/survey_data.py`), **헤더만 있는 빈 템플릿**(`data/templates/survey_template.csv`), 앱의 '실제 조사 데이터 (준비)' 탭에서 **형식 검증**만 수행
-- 구현하지 않음(계획): 업로드 데이터 저장, AI 결과와의 비교, 보정, 통계 분석. 임의의 설문 결과를 만들지 않음
+- **목적:** 이후 실제 설문 결과를 CSV로 받아 AI 가상 의견과 비교·검증할 때 쓸 공통 형식
+- **현재 구현 범위**
+  - 필드 정의: `src/survey_data.py`
+  - 앱 '실제 조사 데이터 (준비)' 탭의 **빈 입력 템플릿 CSV 다운로드**(헤더만 있음)
+  - 업로드한 CSV의 **형식 검증**
+- **구현하지 않음 (계획):** 업로드 데이터 저장, AI 결과와의 비교, 보정, 통계 분석. 임의의 설문 결과를 만들지 않습니다.
 
 ## 형식
-- UTF-8 CSV (엑셀에서 'CSV UTF-8'로 저장), 1행 = 응답자 1명 × 평가 제품 1개 (긴 형식)
-- 두 제품 비교 조사라면 응답자당 제품 행 2개 + 각 행에 `preferred_product` 같은 값
+
+- UTF-8 CSV입니다(엑셀에서는 'CSV UTF-8'로 저장).
+- 1행은 응답자 1명 × 평가 제품 1개입니다(긴 형식).
+- 두 제품을 비교하는 조사라면 응답자마다 제품 행 2개를 쓰고, 각 행에 `preferred_product` 같은 값을 넣습니다.
 
 | 컬럼 | 필수 | 설명 |
 |---|---|---|
@@ -23,19 +28,27 @@
 | `purchase_intent` | 선택 | 구매의향 1~5점 |
 | `price_acceptance` | 선택 | 가격 적절성 1~5점 |
 | `preferred_product` | 선택 | 두 제품 비교 시 선호 제품: A / B / none |
-| `choice_reason` | 선택 | 선택 이유 또는 구매하고 싶은/망설여지는 이유 (자유 응답) |
+| `choice_reason` | 선택 | 선택 이유, 또는 구매하고 싶은/망설여지는 이유 (자유 응답) |
 
 ## 검증 규칙 (현재 구현)
-- 필수 컬럼 `respondent_id`, `product_id` 존재 및 값 필수
-- `purchase_intent`, `price_acceptance`는 1~5 정수, `preferred_product`는 A / B / none, `presentation_order`는 single / A_first / B_first
-- 각 행에 결과 값(구매의향·가격 적절성·선호 제품·이유) 중 하나 이상 필요
-- 같은 응답자·제품 조합 중복 금지
-- 템플릿에 없는 컬럼은 '추가 컬럼'으로 알려 줌 (오류 아님)
-
-## 활용 절차 (계획)
-1. 조사 전에 비교 지표, 성공 기준, 페르소나와 프롬프트 버전을 고정
-2. **보정(학습)용과 검증용 데이터를 분리**: 응답자 단위로 나누고, 가능하면 제품 단위로도 분리. 검증용 데이터는 보정이 끝날 때까지 보지 않음
-3. 보정 전 비교: 같은 제품·타깃 조건의 AI 결과와 실측(자유응답 주제 일치도, 구매의향 분포, 선호 방향)을 비교
-4. 보정은 보정용 데이터로만 수행(페르소나 설명·프롬프트 조정 등). 파인튜닝은 데이터 규모·라이선스·자원 검토 후 별도로 결정
-5. 검증용 데이터로 다시 비교해, 개선된 지표만 서비스에 표시
-6. 개인정보: 응답자 ID는 익명화, 이름·연락처 등은 넣지 않음
+
+- 필수 컬럼 `respondent_id`, `product_id`가 있어야 하고, 값도 비어 있으면 안 됩니다.
+- 값 범위
+  - `purchase_intent`, `price_acceptance`: 1~5 정수
+  - `preferred_product`: A / B / none
+  - `presentation_order`: single / A_first / B_first
+- 각 행에 결과 값(구매의향, 가격 적절성, 선호 제품, 이유) 중 하나 이상이 있어야 합니다.
+- 같은 응답자·제품 조합은 중복될 수 없습니다.
+- 템플릿에 없는 컬럼은 '추가 컬럼'으로 알려 주고, 오류로 처리하지 않습니다.
+- 검증 결과만 화면에 보여 주고, 업로드한 데이터는 저장하지 않습니다.
+
+## 활용 절차 (계획, 미구현)
+
+1. **사전 고정:** 조사 전에 비교 지표, 성공 기준, 페르소나와 프롬프트 버전을 고정합니다.
+2. **데이터 분리:** 보정(학습)용과 검증용 데이터를 나눕니다.
+   - 응답자 단위로 나누고, 가능하면 제품 단위로도 나눕니다.
+   - 검증용 데이터는 보정이 끝날 때까지 보지 않습니다.
+3. **보정 전 비교:** 같은 제품·타깃 조건의 AI 결과를 실측과 비교합니다. 자유응답 주제 일치도, 구매의향 분포, 선호 방향을 봅니다.
+4. **보정:** 보정용 데이터로만 합니다(페르소나 설명이나 프롬프트 조정 등). 파인튜닝은 데이터 규모, 라이선스, 자원을 검토한 뒤 따로 결정합니다.
+5. **검증:** 검증용 데이터로 다시 비교하고, 개선이 확인된 지표만 서비스에 표시합니다.
+6. **개인정보:** 응답자 ID는 익명화하고, 이름이나 연락처 등은 넣지 않습니다.
```
