"""判题纯函数。单选/多选/判断统一按「答案集合是否相等」判定。"""

from __future__ import annotations


def _normalize(answer: list[str]) -> set[str]:
    return {str(a).strip().upper() for a in answer if str(a).strip()}


def judge(correct_answer: list[str], user_answer: list[str]) -> bool:
    """用户答案与正确答案完全一致才算对。

    - 单选：单元素集合相等
    - 多选：必须选全且不多选（集合相等）
    - 判断：{'对'} / {'错'} 集合相等
    顺序与大小写不敏感。
    """
    return _normalize(correct_answer) == _normalize(user_answer)
