"""客观题集训 PDF 专用解析器（计划 A）。

格式：正文按题号出题，文末按题号附【答案】【解析】。题号 1:1 唯一键配对。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz

from .schema import Option, QuestionDraft

CHAPTER_NAMES = (
    "总论",
    "公司法律制度",
    "合伙企业法律制度",
    "物权法律制度",
    "合同法律制度",
    "金融法律制度",
    "财政法律制度",
)

# 页眉页脚噪音行（整行匹配或包含即丢弃）
_NOISE_PATTERNS = (
    re.compile(r"^学会计就到之了课堂\s*$"),
    re.compile(r"^\|\s*强化班[-—]周周\s*$"),
    re.compile(r"^第二部分\s*客观题集训\s*$"),
    re.compile(r"^\d+\s*$"),  # 孤立页码
)

_QTYPE_MAP = {"单选题": "单选", "多选题": "多选", "判断题": "判断"}


def _map_qtype(raw: str) -> str:
    return _QTYPE_MAP.get(raw, raw)


_Q_START = re.compile(r"^(\d+)[.、]【(单选题|多选题|判断题)】(.*)$", re.M)
_A_START = re.compile(r"^(\d+)[.、]【答案】(.*)$", re.M)
_KAODIAN = re.compile(r"考点(\d+)[:：](.+)")
_OPTION = re.compile(r"^([A-D])[.、．]\s*(.*)$")


@dataclass
class ParseReport:
    pdf_name: str
    chapter: str | None
    question_count: int = 0
    answer_count: int = 0
    question_numbers: list[int] = field(default_factory=list)
    answer_numbers: list[int] = field(default_factory=list)
    missing_in_answers: list[int] = field(default_factory=list)
    missing_in_questions: list[int] = field(default_factory=list)
    continuous: bool = False
    drafts: list[QuestionDraft] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            self.question_count > 0
            and self.question_count == self.answer_count
            and not self.missing_in_answers
            and not self.missing_in_questions
            and self.continuous
            and not self.errors
        )


def _is_noise(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    for pat in _NOISE_PATTERNS:
        if pat.search(s):
            return True
    # 重复章名行（无其它内容）
    if s in CHAPTER_NAMES:
        return True
    return False


def _clean_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if _is_noise(line):
            continue
        lines.append(line)
    return "\n".join(lines)


def infer_chapter(text: str) -> str | None:
    for name in CHAPTER_NAMES:
        if name in text:
            return name
    m = re.search(r"第[一二三四五六七]章\s*(\S+)", text)
    if m:
        frag = m.group(1)
        for name in CHAPTER_NAMES:
            if frag in name or name.startswith(frag):
                return name
    return None


def _normalize_answer(raw: str, question_type: str) -> list[str]:
    s = raw.strip()
    # 去掉同行残留的【解析】前缀
    if "【解析】" in s:
        s = s.split("【解析】", 1)[0].strip()
    if question_type == "判断":
        if s in ("√", "对", "正确", "Y", "T"):
            return ["对"]
        if s in ("×", "错", "错误", "N", "F"):
            return ["错"]
    if question_type == "多选" and s and all(c.isalpha() for c in s):
        return sorted({c.upper() for c in s})
    if s and all(c.isalpha() and c.isascii() for c in s) and len(s) == 1:
        return [s.upper()]
    if s:
        return [s]
    return []


def _parse_answers(answer_text: str) -> dict[int, tuple[list[str], str]]:
    """题号 -> (correct_answer, explanation)。"""
    out: dict[int, tuple[list[str], str]] = {}
    matches = list(_A_START.finditer(answer_text))
    for i, m in enumerate(matches):
        num = int(m.group(1))
        rest = m.group(2).strip()
        ans_part = rest
        expl = ""
        if "【解析】" in rest:
            ans_part, expl = rest.split("【解析】", 1)
            ans_part = ans_part.strip()
            expl = expl.strip()
        else:
            # 答案在首行，解析从下一行【解析】开始
            tail_start = m.end()
            tail_end = matches[i + 1].start() if i + 1 < len(matches) else len(answer_text)
            tail = answer_text[tail_start:tail_end]
            em = re.search(r"【解析】(.*)", tail, re.S)
            if em:
                expl = em.group(1).strip()
        out[num] = (ans_part, expl)
    return out


def _parse_questions(question_text: str) -> list[tuple[int, str, str, str, list[Option], str]]:
    """返回 [(num, qtype, stem, exam_point, options, raw_block), ...]。"""
    items: list[tuple[int, str, str, str, list[Option], str]] = []
    current_kaodian = ""
    matches = list(_Q_START.finditer(question_text))
    for i, m in enumerate(matches):
        num = int(m.group(1))
        qtype = _map_qtype(m.group(2))
        block_start = m.start()
        block_end = matches[i + 1].start() if i + 1 < len(matches) else len(question_text)
        block = question_text[block_start:block_end]

        # 更新考点上下文（块内 + 块前最近考点）
        prefix = question_text[:block_start]
        for km in _KAODIAN.finditer(prefix):
            current_kaodian = km.group(2).strip()
        for km in _KAODIAN.finditer(block):
            current_kaodian = km.group(2).strip()

        lines = block.splitlines()
        stem_lines = [m.group(3).strip()]
        options: list[Option] = []
        for line in lines[1:]:
            line = line.strip()
            if not line or _KAODIAN.search(line):
                continue
            om = _OPTION.match(line)
            if om:
                options.append(Option(key=om.group(1), text=om.group(2).strip()))
            elif not om and not line.startswith("考点"):
                # 题干续行（非选项）
                if not _OPTION.match(line) and line not in ("", "）", "（"):
                    stem_lines.append(line)
        stem = "".join(stem_lines).strip()
        items.append((num, qtype, stem, current_kaodian, options, block))
    return items


def parse_jixun_pdf(pdf_path: str | Path) -> ParseReport:
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    try:
        raw = "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()

    text = _clean_text(raw)
    report = ParseReport(pdf_name=pdf_path.name, chapter=infer_chapter(text))

    split = _A_START.search(text)
    if not split:
        report.errors.append("未找到答案区（无 N.【答案】 标记）")
        return report

    q_text, a_text = text[: split.start()], text[split.start() :]
    q_items = _parse_questions(q_text)
    a_map_raw = _parse_answers(a_text)

    report.question_count = len(q_items)
    report.answer_count = len(a_map_raw)
    report.question_numbers = [x[0] for x in q_items]
    report.answer_numbers = sorted(a_map_raw.keys())

    qset, aset = set(report.question_numbers), set(report.answer_numbers)
    report.missing_in_answers = sorted(qset - aset)
    report.missing_in_questions = sorted(aset - qset)
    if report.question_numbers:
        n = len(report.question_numbers)
        report.continuous = (
            report.question_numbers == list(range(1, n + 1))
            and report.answer_numbers == list(range(1, n + 1))
        )

    for num, qtype, stem, exam_point, options, _ in q_items:
        if num not in a_map_raw:
            report.errors.append(f"题号 {num} 无对应答案")
            continue
        ans_raw, expl = a_map_raw[num]
        answer = _normalize_answer(ans_raw, qtype)
        if not answer:
            report.errors.append(f"题号 {num} 答案无法解析: {ans_raw!r}")
            continue
        report.drafts.append(
            QuestionDraft(
                stem=stem,
                question_type=qtype,
                options=options,
                correct_answer=answer,
                explanation=expl or None,
                chapter=report.chapter,
                exam_point=exam_point or None,
                confidence=1.0,
            )
        )

    return report


def sample_lines(report: ParseReport, limit: int = 5) -> list[str]:
    """抽查样本：题号 ↔ 题干 ↔ 答案 ↔ 解析首句。"""
    lines: list[str] = []
    for i, d in enumerate(report.drafts[:limit]):
        expl = (d.explanation or "")[:60].replace("\n", " ")
        lines.append(
            f"#{i + 1} [{d.question_type}] {d.stem[:50]}… "
            f"→ 答案={d.correct_answer} 解析={expl}…"
        )
    return lines
