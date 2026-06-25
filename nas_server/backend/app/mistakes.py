"""错题本状态机（纯函数）：一次作答如何改变某题的错题记录。

S3 范围：答错自动收录、重复出错累加；答对仅对已在本中的题累加答对次数。
掌握状态升级/降级留作后续批次，这里掌握状态保持「未掌握」。
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class MistakeRecord:
    wrong_answer: list[str]
    correct_answer: list[str]
    wrong_count: int
    correct_count: int
    first_wrong_at: str
    last_attempt_at: str
    mastery: str = "未掌握"


def apply_attempt(
    current: MistakeRecord | None,
    *,
    is_correct: bool,
    user_answer: list[str],
    correct_answer: list[str],
    now: str,
) -> MistakeRecord | None:
    """返回作答后的错题记录；返回 None 表示该题不应出现在错题本里。

    - 未在本中 + 答对 → None（不收录）
    - 未在本中 + 答错 → 新建记录（错误次数 1，掌握状态 未掌握）
    - 已在本中 + 答错 → 错误次数 +1，更新最近作答与错误答案
    - 已在本中 + 答对 → 答对次数 +1，更新最近作答
    """
    if current is None:
        if is_correct:
            return None
        return MistakeRecord(
            wrong_answer=user_answer,
            correct_answer=correct_answer,
            wrong_count=1,
            correct_count=0,
            first_wrong_at=now,
            last_attempt_at=now,
            mastery="未掌握",
        )

    if is_correct:
        return replace(
            current,
            correct_count=current.correct_count + 1,
            last_attempt_at=now,
            correct_answer=correct_answer,
        )
    return replace(
        current,
        wrong_count=current.wrong_count + 1,
        wrong_answer=user_answer,
        last_attempt_at=now,
        correct_answer=correct_answer,
    )
