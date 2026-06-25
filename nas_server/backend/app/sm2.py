"""S9 SM-2 调度纯函数。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta

MIN_EASE = 1.3
DEFAULT_EASE = 2.5
LAPSE_INTERVAL_DAYS = 3  # 答错后的基础复现间隔（天）：不再天天追着出

# 答错复现间隔分布：多数 3~5 天后再出，偶尔（约 1/12）隔天再来一次
_LAPSE_CHOICES = (1, 2, 3, 4, 5)
_LAPSE_WEIGHTS = (1, 2, 4, 3, 2)


def random_lapse_interval() -> int:
    """答错后下次出现的间隔（带抖动）。"""
    return random.choices(_LAPSE_CHOICES, weights=_LAPSE_WEIGHTS, k=1)[0]


@dataclass(frozen=True)
class Sm2State:
    ease: float = DEFAULT_EASE
    interval_days: int = 0
    repetition: int = 0
    due_date: str | None = None  # YYYY-MM-DD


def quality_from_correct(is_correct: bool) -> int:
    """对错 → SM-2 质量分：答对=良好(4)，答错=重来(0)。"""
    return 4 if is_correct else 0


def sm2_schedule(
    state: Sm2State,
    quality: int,
    today: date,
    *,
    lapse_interval: int = LAPSE_INTERVAL_DAYS,
) -> Sm2State:
    """SM-2 一步调度，返回新状态。

    答错时间隔置为 lapse_interval（默认 3 天，非传统 SM-2 的 1 天），
    避免错题天天复现；调用方可传 random_lapse_interval() 加抖动。
    """
    ease = state.ease
    interval = state.interval_days
    repetition = state.repetition

    if quality < 3:
        repetition = 0
        interval = max(1, lapse_interval)
    else:
        if repetition == 0:
            interval = 1
        elif repetition == 1:
            interval = 6
        else:
            interval = max(1, round(interval * ease))
        repetition += 1

    ease = ease + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    if ease < MIN_EASE:
        ease = MIN_EASE

    due = today + timedelta(days=interval)
    return Sm2State(
        ease=round(ease, 2),
        interval_days=interval,
        repetition=repetition,
        due_date=due.isoformat(),
    )


def is_due(due_date: str | None, today: date) -> bool:
    if not due_date:
        return False
    return date.fromisoformat(due_date) <= today
