"""S9 每日任务生成纯函数。"""

from __future__ import annotations

MASTERY_WEIGHT = {"掌握": 3.0, "熟悉": 2.0, "了解": 1.0}


def generate_task_question_ids(
    *,
    due_ids: list[int],
    new_candidates: list[tuple[int, float]],
    target: int,
    exclude: set[int] | None = None,
) -> list[int]:
    """到期复习优先 → 按权重补足新题，去重，截断到 target。"""
    exclude = exclude or set()
    picked: list[int] = []
    seen: set[int] = set()

    for qid in due_ids:
        if qid in exclude or qid in seen:
            continue
        picked.append(qid)
        seen.add(qid)
        if len(picked) >= target:
            return picked

    ranked = sorted(new_candidates, key=lambda x: (-x[1], x[0]))
    for qid, _w in ranked:
        if qid in exclude or qid in seen:
            continue
        picked.append(qid)
        seen.add(qid)
        if len(picked) >= target:
            break
    return picked


def mastery_weight(mastery_requirement: str | None) -> float:
    return MASTERY_WEIGHT.get(mastery_requirement or "熟悉", 2.0)
