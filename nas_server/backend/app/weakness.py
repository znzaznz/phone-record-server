"""S7 薄弱点分析：按知识点聚合与优先级排序（纯函数）。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

MASTERY_WEIGHT = {"掌握": 3.0, "熟悉": 2.0, "了解": 1.0}
ACCURACY_WEAK_THRESHOLD = 0.7
STALE_DAYS = 7


@dataclass(frozen=True)
class KpStats:
    knowledge_point_id: int
    name: str
    chapter: str
    mastery_requirement: str | None
    attempt_count: int
    correct_count: int
    wrong_count: int
    mistake_count: int
    last_attempt_at: str | None


@dataclass(frozen=True)
class WeaknessItem:
    knowledge_point_id: int
    name: str
    chapter: str
    mastery_requirement: str | None
    attempt_count: int
    correct_count: int
    wrong_count: int
    mistake_count: int
    accuracy: float | None
    last_attempt_at: str | None
    priority: float
    tags: tuple[str, ...]


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", ""))
    except ValueError:
        return None


def _days_since(ts: str | None, now: datetime) -> int | None:
    dt = _parse_ts(ts)
    if dt is None:
        return None
    return max(0, (now - dt).days)


def weakness_tags(stats: KpStats, now: datetime) -> list[str]:
    tags: list[str] = []
    acc = (
        stats.correct_count / stats.attempt_count
        if stats.attempt_count > 0
        else None
    )
    if acc is not None and acc < ACCURACY_WEAK_THRESHOLD:
        tags.append("正确率低")
    if stats.mistake_count >= 2:
        tags.append("高频错题")
    if stats.mistake_count >= 1 and stats.wrong_count >= 2:
        tags.append("重复做错")
    days = _days_since(stats.last_attempt_at, now)
    if days is not None and days >= STALE_DAYS:
        tags.append("久未复习")
    weight = MASTERY_WEIGHT.get(stats.mastery_requirement or "", 0)
    if weight >= 3 and acc is not None and acc < 0.6:
        tags.append("重点待加强")
    return tags


def weakness_priority(stats: KpStats, now: datetime) -> float:
    """分数越高越薄弱，应优先复习。"""
    if stats.attempt_count == 0 and stats.mistake_count == 0:
        return -1.0

    w = MASTERY_WEIGHT.get(stats.mastery_requirement or "熟悉", 2.0)
    acc = stats.correct_count / stats.attempt_count if stats.attempt_count else 0.0

    score = (1.0 - acc) * w * 10.0
    score += stats.mistake_count * 2.0
    score += stats.wrong_count * 0.5

    days = _days_since(stats.last_attempt_at, now)
    if days is not None and days >= STALE_DAYS:
        score += min(days / STALE_DAYS, 3.0)
    elif stats.attempt_count == 0 and stats.mistake_count > 0:
        score += 2.0

    return score


def rank_weaknesses(stats_list: list[KpStats], now: datetime | None = None) -> list[WeaknessItem]:
    now = now or datetime.now()
    items: list[WeaknessItem] = []
    for s in stats_list:
        if s.attempt_count == 0 and s.mistake_count == 0:
            continue
        acc = s.correct_count / s.attempt_count if s.attempt_count else None
        items.append(
            WeaknessItem(
                knowledge_point_id=s.knowledge_point_id,
                name=s.name,
                chapter=s.chapter,
                mastery_requirement=s.mastery_requirement,
                attempt_count=s.attempt_count,
                correct_count=s.correct_count,
                wrong_count=s.wrong_count,
                mistake_count=s.mistake_count,
                accuracy=round(acc, 3) if acc is not None else None,
                last_attempt_at=s.last_attempt_at,
                priority=round(weakness_priority(s, now), 2),
                tags=tuple(weakness_tags(s, now)),
            )
        )
    items.sort(key=lambda x: (-x.priority, x.name))
    return items
