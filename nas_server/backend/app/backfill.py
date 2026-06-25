"""S5 题库数据补全：把"解析页"记录的答案+解析配对回对应真题。

数据事实：解析记录的 correct_answer / explanation 已由 S1 提取好，
本模块只负责"哪条解析属于哪道真题"的配对，并产出补全动作。

配对策略（纯函数、可测、无需模型）：
- 同一 PDF 内题号唯一 → **按题号匹配**（PDF2: 真题1-35 ↔ 解析1-35，铁稳）。
- 题号有重复（PDF1: 奇兵+章后两套编号）→ **按文档顺序位置配对**。
- 一律用"解析答案 vs 真题已有答案"做**交叉校验**定置信度；不一致/缺失 → 进人工确认队列。
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class QRow:
    id: int
    number: int | None      # 题干前导题号
    answer: list[str]       # 已提取的答案（真题可能为空）
    explanation: str | None


@dataclass
class BackfillAction:
    zhenti_id: int
    jiexi_id: int | None        # 配到的解析记录 id（None=没配到）
    set_answer: list[str] | None    # 需要写入真题的答案（None=不改）
    set_explanation: str | None     # 需要写入真题的解析
    needs_review: bool
    note: str


_NUM_RE = re.compile(r"^\s*(\d+)")


def parse_number(stem: str) -> int | None:
    m = _NUM_RE.match(stem or "")
    return int(m.group(1)) if m else None


def _normalize(ans: list[str]) -> set[str]:
    return {str(a).strip().upper() for a in (ans or []) if str(a).strip()}


def _unique_numbers(rows: list[QRow]) -> bool:
    nums = [r.number for r in rows if r.number is not None]
    return len(nums) == len(set(nums)) and len(nums) == len(rows)


def _split_sections(rows: list[QRow]) -> list[list[QRow]]:
    """按"题号重置（下一题号 <= 上一题号）"把文档顺序切成小节。

    例：奇兵题号 1-5 后章后题号又从 1 开始 → 切成两节，避免跨节误配。
    """
    sections: list[list[QRow]] = []
    cur: list[QRow] = []
    prev = 0
    for r in rows:
        if r.number is not None and r.number <= prev and cur:
            sections.append(cur)
            cur = []
        cur.append(r)
        if r.number is not None:
            prev = r.number
    if cur:
        sections.append(cur)
    return sections


def _pair_one_section(zt: list[QRow], jx: list[QRow]) -> list[tuple[QRow, QRow | None]]:
    if _unique_numbers(zt) and _unique_numbers(jx):
        jx_by_num = {j.number: j for j in jx}
        return [(z, jx_by_num.get(z.number)) for z in zt]
    return [(zt[i], jx[i] if i < len(jx) else None) for i in range(len(zt))]


def pair_rows(zhenti: list[QRow], jiexi: list[QRow]) -> list[tuple[QRow, QRow | None]]:
    """配对：先按编号重置切小节，节节对应，节内按题号匹配（题号唯一时）。"""
    zt_secs = _split_sections(zhenti)
    jx_secs = _split_sections(jiexi)
    pairs: list[tuple[QRow, QRow | None]] = []
    for i, zt_sec in enumerate(zt_secs):
        jx_sec = jx_secs[i] if i < len(jx_secs) else []
        pairs.extend(_pair_one_section(zt_sec, jx_sec))
    return pairs


def plan_backfill(zhenti: list[QRow], jiexi: list[QRow]) -> list[BackfillAction]:
    """对一个 PDF 的真题/解析两组，产出补全动作列表。"""
    actions: list[BackfillAction] = []
    for zt, jx in pair_rows(zhenti, jiexi):
        if jx is None:
            actions.append(BackfillAction(zt.id, None, None, None, True, "无对应解析页"))
            continue
        zt_ans, jx_ans = _normalize(zt.answer), _normalize(jx.answer)
        expl = (jx.explanation or "").strip() or None
        if zt_ans and jx_ans and zt_ans == jx_ans:
            # 真题已有答案且与解析一致 → 高置信，仅补解析
            actions.append(BackfillAction(zt.id, jx.id, None, expl, False, "答案一致，补解析"))
        elif not zt_ans and jx_ans:
            # 真题缺答案，解析有明确答案（来自"【答案】X"）→ 信解析，高置信补全
            actions.append(
                BackfillAction(zt.id, jx.id, list(jx.answer), expl, False, "答案取自解析")
            )
        elif zt_ans and jx_ans and zt_ans != jx_ans:
            # 答案冲突 → 可能配对错位，进队列，不覆盖原答案
            actions.append(
                BackfillAction(zt.id, jx.id, None, expl, True, f"答案不一致(题{zt_ans}/解析{jx_ans})")
            )
        else:
            actions.append(BackfillAction(zt.id, jx.id, None, expl, True, "解析无答案，待核"))
    return actions
