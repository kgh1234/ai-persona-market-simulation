"""NVIDIA Nemotron-Personas-Korea 원격 부분 읽기 (실험 모듈, 기존 MVP와 분리).

데이터: https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea  (CC BY 4.0, NVIDIA)
- 전체 1,000,000행 · parquet 9개(각 약 220MB, 합계 약 1.98GB) · 행 그룹 5개/파일
- 전체를 내려받지 않는다. HTTP 범위 요청으로 필요한 열·행 그룹만 읽는다.
  1) 인구통계 열(성별·나이·직업·지역 등, 파일당 약 0.8MB)만 읽어 선별
  2) 선별된 행이 속한 행 그룹에서만 uuid·서사 열을 읽음
- 접근 방식은 Power-xc/power-persona-sim 의 'remote / shard' 경로 아이디어를 참고해 새로 작성(코드 복사 아님).
  (참고 저장소는 DuckDB httpfs 사용, 여기서는 기존 환경에 있는 pyarrow + huggingface_hub 만 사용)
- 캐시: outputs/cache/nemotron/ (HF_HOME 도 outputs/cache/hf_home 으로 고정, 홈 디렉터리에 쓰지 않음)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# [공개본 변경] 캐시를 소스 폴더가 아닌 실행 결과 폴더(outputs/)에 저장. 원본: HERE/"cache", HERE/"hf_home"
if str(HERE.parents[1] / "src") not in sys.path:
    sys.path.insert(0, str(HERE.parents[1] / "src"))
import runtime_config as rc  # noqa: E402

CACHE = rc.NEMOTRON_CACHE_DIR
os.environ.setdefault("HF_HOME", str(rc.HF_HOME_DIR))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

DATASET_ID = "nvidia/Nemotron-Personas-Korea"
DATASET_REVISION = "ada0f5b53a38bb5a30cce09358adde883c1ab63a"  # 2026-11-04 확인한 커밋 (재현성 고정)
N_SHARDS = 9
DEMO_COLS = ["sex", "age", "marital_status", "family_type", "housing_type", "education_level",
             "bachelors_field", "occupation", "district", "province"]
TEXT_COLS = ["uuid", "persona", "culinary_persona", "hobbies_and_interests", "professional_persona"]
ATTRIBUTION = ("NVIDIA Nemotron-Personas-Korea (CC BY 4.0, https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea). "
               "합성 데이터이며, 본 실험에서 일부 레코드를 선별·형식 변환해 사용함")

_LOG: list[dict] = []  # 원격 읽기 기록 (비용 보고용)


def _path(shard: int) -> str:
    return f"datasets/{DATASET_ID}@{DATASET_REVISION}/data/train-{shard:05d}-of-{N_SHARDS:05d}.parquet"


def _open(shard: int):
    from huggingface_hub import HfFileSystem
    return HfFileSystem().open(_path(shard), "rb", block_size=1 << 20)


def read_log() -> list[dict]:
    return list(_LOG)


def load_demographics(shards=range(N_SHARDS)) -> pa.Table:
    """인구통계 열 + 위치(shard, row_group, row_in_group). 샤드별로 캐시."""
    CACHE.mkdir(parents=True, exist_ok=True)  # [공개본 변경] parents=True
    tables = []
    for s in shards:
        cp = CACHE / f"demo_shard{s}.parquet"
        if cp.exists():
            tables.append(pq.read_table(cp))
            continue
        t0 = time.time()
        with _open(s) as f:
            pf = pq.ParquetFile(f)
            parts = []
            for rg in range(pf.metadata.num_row_groups):
                t = pf.read_row_group(rg, columns=DEMO_COLS)
                n = t.num_rows
                t = t.append_column("shard", pa.array([s] * n, pa.int16()))
                t = t.append_column("row_group", pa.array([rg] * n, pa.int16()))
                t = t.append_column("row_in_group", pa.array(range(n), pa.int32()))
                parts.append(t)
        tb = pa.concat_tables(parts)
        pq.write_table(tb, cp)
        _LOG.append({"what": f"demographics shard {s}", "rows": tb.num_rows, "sec": round(time.time() - t0, 1),
                     "cached_bytes": cp.stat().st_size})
        tables.append(tb)
    return pa.concat_tables(tables)


def fetch_text(locs: list[tuple[int, int, int]], columns=TEXT_COLS) -> dict[tuple[int, int, int], dict]:
    """선별된 위치의 서사 열만 읽음. 같은 행 그룹은 한 번만 읽고, 필요한 행만 캐시에 남김."""
    CACHE.mkdir(parents=True, exist_ok=True)  # [공개본 변경] parents=True
    out: dict = {}
    by_group: dict = {}
    for s, rg, i in locs:
        by_group.setdefault((s, rg), []).append(i)
    for (s, rg), rows in sorted(by_group.items()):
        tag = "" if list(columns) == TEXT_COLS else "_" + "-".join(columns)
        cp = CACHE / f"text_s{s}_rg{rg}{tag}.parquet"
        cached = pq.read_table(cp).to_pylist() if cp.exists() else []
        have = {r["row_in_group"]: r for r in cached}
        need = [i for i in rows if i not in have]
        if need:
            t0 = time.time()
            with _open(s) as f:
                t = pq.ParquetFile(f).read_row_group(rg, columns=columns)
            for i in need:
                rec = {c: t.column(c)[i].as_py() for c in columns}
                rec["row_in_group"] = i
                have[i] = rec
            pq.write_table(pa.Table.from_pylist(list(have.values())), cp)
            _LOG.append({"what": f"text shard {s} rg {rg} ({len(need)} rows kept)", "sec": round(time.time() - t0, 1)})
        for i in rows:
            out[(s, rg, i)] = have[i]
    return out
