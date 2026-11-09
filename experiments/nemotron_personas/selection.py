"""타깃 조건 기반 소규모 페르소나 선별 (실험 모듈).

- 조건: 연령 범위, 성별, 시도, 직업(포함/제외 단어), 가구 형태(부분 문자열)
- 층화: 연령대(10세) × 성별 칸으로 나누고, 표본 수를 '조건을 만족하는 데이터셋 내 분포'에 비례 배분(최대 나머지 방식)
- 칸 안의 순서: seed + 위치 해시로 정렬 (난수 생성기 버전과 무관하게 재현).
  power-persona-sim sampling/__init__.py 의 '정렬 키 = 표본, 동점은 seed 해시' 아이디어를 참고해 새로 작성.
- 가중치: 외부 기준 분포표(출처·연도 필수)가 주어진 '인구통계 칸'에만 계산. 없으면 전부 1.0 + '미보정' 표시.
  음료 취향·소비 성향에는 외부 통계가 없으므로 가중치를 두지 않는다 (데이터에도 해당 필드 없음).
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Optional

import pandas as pd

METRO = ["서울", "경기", "인천"]


@dataclass
class Target:
    age_min: int = 19
    age_max: int = 99
    sex: Optional[str] = None                 # '여자' / '남자' / None
    provinces: list = field(default_factory=list)   # 비우면 전국
    occupation_include: list = field(default_factory=list)  # 부분 문자열, 하나라도 포함
    occupation_exclude: list = field(default_factory=list)  # 예: ['무직']
    family_include: list = field(default_factory=list)       # 예: ['혼자 거주']

    def describe(self) -> str:
        p = [f"{self.age_min}~{self.age_max}세"]
        p.append(self.sex or "성별 무관")
        p.append(", ".join(self.provinces) if self.provinces else "전국")
        if self.occupation_include:
            p.append("직업 포함: " + ", ".join(self.occupation_include))
        if self.occupation_exclude:
            p.append("직업 제외: " + ", ".join(self.occupation_exclude))
        if self.family_include:
            p.append("가구: " + ", ".join(self.family_include))
        return " · ".join(p)


def filter_target(df: pd.DataFrame, t: Target) -> pd.DataFrame:
    m = df.age.between(t.age_min, t.age_max)
    if t.sex:
        m &= df.sex == t.sex
    if t.provinces:
        m &= df.province.isin(t.provinces)
    if t.occupation_include:
        m &= df.occupation.apply(lambda o: any(k in o for k in t.occupation_include))
    if t.occupation_exclude:
        m &= ~df.occupation.apply(lambda o: any(k in o for k in t.occupation_exclude))
    if t.family_include:
        m &= df.family_type.apply(lambda o: any(k in o for k in t.family_include))
    return df[m]


def cell_of(row) -> str:
    return f"{min(row['age'] // 10 * 10, 70)}대{'+' if row['age'] >= 70 else ''}|{row['sex']}"


def _hash(seed: int, s, rg, i) -> str:
    return hashlib.blake2b(f"{seed}:{s}:{rg}:{i}".encode(), digest_size=12).hexdigest()


def allocate(counts: dict, n: int) -> dict:
    """칸별 모집단(=조건 만족 레코드) 수에 비례해 n 배분 (최대 나머지 방식)."""
    tot = sum(counts.values())
    raw = {k: n * v / tot for k, v in counts.items()}
    base = {k: int(x) for k, x in raw.items()}
    rest = n - sum(base.values())
    for k in sorted(raw, key=lambda k: (-(raw[k] - base[k]), k))[:rest]:
        base[k] += 1
    return {k: v for k, v in base.items() if v}


def select(df: pd.DataFrame, t: Target, n: int, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    pool = filter_target(df, t).copy()
    if pool.empty:
        return pool, {"target": t.describe(), "pool": 0, "selected": 0}
    pool["cell"] = pool.apply(cell_of, axis=1)
    pool["rank"] = [_hash(seed, s, rg, i) for s, rg, i in zip(pool.shard, pool.row_group, pool.row_in_group)]
    counts = pool.cell.value_counts().to_dict()
    alloc = allocate(counts, min(n, len(pool)))
    picked = pd.concat([pool[pool.cell == c].sort_values("rank").head(k) for c, k in sorted(alloc.items())])
    diag = {"target": t.describe(), "target_spec": asdict(t), "seed": seed, "pool": int(len(pool)),
            "pool_share_of_dataset": round(len(pool) / len(df), 4), "cells_in_pool": {k: int(v) for k, v in sorted(counts.items())},
            "allocation": alloc, "selected": int(len(picked)),
            "note": "표본 수는 '조건을 만족하는 Nemotron 레코드'의 연령대×성별 분포에 비례 배분. 외부 통계로 보정한 것이 아님."}
    return picked.drop(columns=["rank"]), diag


# ---------------------------------------------------------------------------
# 가중치: 외부 기준표가 있을 때만 (출처·연도 필수). 없으면 미보정.
# ---------------------------------------------------------------------------
def compute_weights(sample: pd.DataFrame, reference: Optional[dict] = None) -> tuple[list, str]:
    """reference = {"source": "...", "year": 2025, "shares": {cell: share}}  (cell 표기는 cell_of 와 동일)
    weight = 기준 비율 / 표본 비율. 기준표가 없거나 출처가 없으면 전부 1.0 (미보정)."""
    if not reference or not reference.get("source") or not reference.get("shares"):
        return [1.0] * len(sample), "미보정 — 출처가 확인된 외부 기준 분포표 없음 (전원 weight=1.0)"
    shares = reference["shares"]
    tot = sum(shares.values())
    cells = sample.apply(cell_of, axis=1)
    frac = cells.value_counts(normalize=True).to_dict()
    w = [round((shares.get(c, 0.0) / tot) / frac[c], 4) if frac.get(c) else 0.0 for c in cells]
    return w, f"사후 보정 — {reference['source']} ({reference.get('year', '연도 미상')})"
