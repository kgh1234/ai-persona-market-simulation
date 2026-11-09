# NOTICE — 출처·라이선스 고지

이 문서는 저장소의 코드, 데이터, 외부 모델·소프트웨어의 출처와 라이선스를 구분해 기록합니다.

## 1. 이 저장소의 코드

- 이 저장소의 소스코드(`app.py`, `src/`, `experiments/nemotron_personas/`)와 문서는 프로젝트 작성자가 작성했습니다.
- **라이선스: 미정.** 대회 출품물 관련 권리 관계를 확인한 뒤 지정할 예정입니다.
  라이선스가 지정되기 전에는 오픈소스 라이선스가 부여되지 않은 상태입니다.

## 2. 데이터

### 2.1 저장소에 포함된 데이터 (프로젝트에서 직접 작성)

| 파일 | 내용 | 비고 |
|---|---|---|
| `data/personas_v2.json` | 기본 가상 소비자 3유형 (가격 민감형·건강 지향형·맛 중심형) | 프로젝트에서 설계한 가상 조건입니다. 실제 조사로 검증된 세그먼트가 아닙니다. |
| `data/examples/product_concepts.json` | 입력 예시용 가상 제품 콘셉트 | 실제 출시 제품, 가격, 영양 정보가 아닙니다. |
| `examples/sample_outputs/` | 샘플 실행 기록 1건: Qwen2.5 7B가 생성한 가상 소비자 의견(프롬프트·응답 원문 포함)과 담당자의 문장별 승인·제외 기록 | 기본 3유형만 사용했고 NVIDIA 데이터는 들어 있지 않습니다. 개인정보, 인증 정보, 서버 경로가 없음을 확인했습니다. AI가 생성한 가상 의견이며 실제 소비자 의견이 아닙니다. |

### 2.2 실행 중에 원격으로 일부를 읽는 외부 데이터 (저장소에 포함하지 않음)

**NVIDIA Nemotron-Personas-Korea**

- 출처: NVIDIA, <https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea>
- 라이선스: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- 사용 리비전: `ada0f5b53a38bb5a30cce09358adde883c1ab63a` (코드에 고정)
- 사용 방식
  - 데이터셋 원본이나 캐시는 이 저장소에 포함하지 않습니다.
  - '페르소나 설정 → NVIDIA 합성 페르소나 불러오기(실험)'를 처음 사용할 때 인구통계 열만 HTTP 범위 요청으로 읽습니다. 9개 parquet 파일에서 합계 약 9MB입니다.
  - 읽은 데이터는 사용자 컴퓨터의 `outputs/cache/nemotron/`에 캐시합니다.
- 변경 사항 (CC BY 4.0의 변경 표시)
  - 조건에 맞는 일부 레코드만 선별합니다.
  - 연령·성별·직업·지역·가구 형태·주거 형태 필드를 이 프로젝트의 페르소나 형식(연령대, 성별, 직업/생활)으로 변환합니다.
  - 서사 텍스트 필드(`persona`, `culinary_persona` 등)는 기본 설정에서 사용하지 않습니다.
- 성격: 공개 통계 분포를 바탕으로 만든 합성 데이터입니다. 실존 인물이나 실제 소비자 응답이 아닙니다.
- 접근: 데이터셋에 접근 제한(gated)이 없어 Hugging Face 로그인 없이 읽을 수 있습니다(개발 환경에서 토큰 없이 확인).
- 이용 조건과 이 저장소의 대응

| CC BY 4.0 조건 | 대응 |
|---|---|
| 저작자 표시 | 앱의 'NVIDIA 합성 페르소나 불러오기' 화면, 등록된 합성 페르소나 상세, 합성 페르소나가 포함된 평가 결과 화면, README, 이 문서에 출처를 표시합니다. |
| 라이선스 고지 (링크) | 불러오기 화면, 등록된 합성 페르소나 상세, README, 이 문서에 CC BY 4.0 링크를 표시합니다. 평가 결과 화면에는 데이터셋 이름과 라이선스 이름을 표시합니다. |
| 변경 사항 표시 | 아래 '변경 사항'과 앱 화면의 출처 문구에 표시합니다. |
| 파생 결과를 공유할 때 | 합성 페르소나로 만든 페르소나 파일이나 평가 기록(`outputs/personas`, `outputs/runs` 등)을 공유할 때도 같은 출처 표시가 필요합니다. 이 파일들에는 데이터셋 레코드 uuid와 변환한 인구통계 값이 들어 있습니다. |

- 인용

```bibtex
@software{nvidia/Nemotron-Personas-Korea,
  author = {Kim, Hyunwoo and Ryu, Jihyeon and Lee, Jinho and Ryu, Hyungon and Praveen, Kiran and Prayaga, Shyamala and Thadaka, Kirit and Jennings, Will and Sadeghi, Bardiya and Sharabiani, Ashton and Choi, Yejin and Meyer, Yev},
  title = {Nemotron-Personas-Korea: Synthetic Personas Aligned to Real-World Distributions for Korea},
  month = {April},
  year = {2026},
  url = {https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea}
}
```

## 3. 외부 모델 (저장소에 포함하지 않음)

| 모델 | 제공 | 라이선스 | 사용 방식 |
|---|---|---|---|
| Qwen2.5 7B (Ollama `qwen2.5:7b`, Instruct 모델, 4-bit 양자화 Q4_K_M) | Alibaba Cloud Qwen Team | Apache License 2.0 (Ollama 모델 패키지에 포함된 라이선스로 확인) | 사용자가 `ollama pull qwen2.5:7b`로 직접 내려받습니다. 모델 가중치는 이 저장소에 포함하지 않으며 재배포하지 않습니다. |

- 이용 조건: 모델의 이용·재배포 조건은 모델과 함께 배포되는 Apache License 2.0을 따릅니다. 이 저장소는 모델 파일을 포함하지 않으므로 모델 라이선스 사본을 재배포하지 않습니다.
- 다른 버전(예: Qwen2.5 3B·72B)은 라이선스가 다를 수 있습니다. 모델을 바꿀 때는 해당 모델의 라이선스를 따로 확인해야 합니다.
- 모델이 생성한 문장은 AI가 만든 가상 의견입니다. 특정 개인이나 실제 소비자의 의견이 아닙니다.

## 4. 외부 소프트웨어 (저장소에 포함하지 않음)

| 소프트웨어 | 라이선스 | 용도 |
|---|---|---|
| [Ollama](https://github.com/ollama/ollama) | MIT | 로컬 LLM 추론 서버 |
| [Streamlit](https://github.com/streamlit/streamlit) 1.65.0 | Apache-2.0 | 웹 UI |
| [pandas](https://github.com/pandas-dev/pandas) 2.3.3 | BSD-3-Clause | 표 처리, CSV |
| [Pydantic](https://github.com/pydantic/pydantic) 2.14.2 | MIT | 스키마 검증 |
| [Requests](https://github.com/psf/requests) 2.34.2 | Apache-2.0 | HTTP 호출 |
| [Apache Arrow (pyarrow)](https://github.com/apache/arrow) 25.0.1 | Apache-2.0 | parquet 읽기 |
| [huggingface_hub](https://github.com/huggingface/huggingface_hub) 2.4.0 | Apache-2.0 | 데이터셋 원격 파일 접근 |
| fsspec, NumPy (위 패키지의 의존성) | BSD-3-Clause | – |

Python 패키지는 `requirements.txt`로 설치하며, 이 저장소에 포함하거나 수정하지 않았습니다.

## 5. 참고했지만 포함하지 않은 외부 코드

| 저장소 | 라이선스 | 관계 |
|---|---|---|
| [microsoft/TinyTroupe](https://github.com/microsoft/TinyTroupe) | MIT | 개발 초기에 별도로 실험한 다중 에이전트 시뮬레이션 프레임워크입니다. **최종 서비스의 실행 엔진이 아니며**, 코드를 포함하거나 import하지 않습니다. |
| [Power-xc/power-persona-sim](https://github.com/Power-xc/power-persona-sim) (확인 시점 커밋 `a24b7c8`) | 저장소에 LICENSE 파일 없음 (`pyproject.toml`에만 "MIT" 표기) | Nemotron 데이터를 원격으로 일부만 읽는 방식과, seed 해시 정렬로 재현 가능하게 선별하는 **아이디어만 참고**했습니다. 코드는 복사하지 않고 새로 작성했습니다(`experiments/nemotron_personas/nemotron_source.py`, `selection.py` 주석 참고). |

## 6. 상표

- NVIDIA와 Nemotron은 NVIDIA Corporation의 상표입니다. Qwen은 Alibaba Cloud의 상표입니다. 그 밖의 제품명과 회사명은 각 권리자의 상표입니다.
- 이 프로젝트는 LOTTE AX CHALLENGE 참가 과제로 개발했습니다. NVIDIA, Alibaba Cloud, Ollama, 롯데 및 계열사가 공식 제공하거나 보증하는 제품이 아닙니다.
