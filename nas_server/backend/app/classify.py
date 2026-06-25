"""S6 题目归类：纯函数层（阈值分流、名称解析）。"""

from __future__ import annotations

import json
from dataclasses import dataclass

CLASSIFY_CONFIDENCE_THRESHOLD = 0.75


@dataclass
class ClassifyOutcome:
    knowledge_point_id: int | None
    confidence: float
    needs_review: bool
    note: str


def format_question_text(stem: str, options_json: str) -> str:
    """题干 + 选项 → 给模型看的纯文本。"""
    lines = [f"题干：{stem}"]
    try:
        opts = json.loads(options_json or "[]")
    except json.JSONDecodeError:
        opts = []
    if opts:
        lines.append("选项：")
        for o in opts:
            lines.append(f"  {o.get('key', '?')}. {o.get('text', '')}")
    return "\n".join(lines)


def resolve_knowledge_point_id(name: str, name_to_id: dict[str, int]) -> int | None:
    name = (name or "").strip()
    if not name:
        return None
    if name in name_to_id:
        return name_to_id[name]
    # 容错：模型多带了能力要求后缀
    for full, kid in name_to_id.items():
        if name.startswith(full) or full in name:
            return kid
    return None


# 确定性关键词归类（优先于 LLM，消歧高频考点）
_KEYWORD_RULES: list[tuple[str, str]] = [
    ("民事诉讼法律制度的规定", "诉讼时效"),
    ("民事诉讼法律制度的规定", "时效期间"),
    ("民事诉讼法律制度的规定", "上诉期限"),
    ("民事诉讼法律制度的规定", "两审终审"),
    ("民事诉讼法律制度的规定", "专属管辖"),
    ("民事诉讼法律制度的规定", "协议管辖"),
    ("代理制度", "表见代理"),
    ("代理制度", "无权代理"),
    ("仲裁法律制度的规定", "仲裁协议"),
    ("仲裁法律制度的规定", "一裁终局"),
    ("行政复议法律制度的规定", "行政复议"),
    ("行政诉讼法律制度的规定", "行政诉讼"),
]


def keyword_classify(stem: str, name_to_id: dict[str, int]) -> ClassifyOutcome | None:
    """题干含强特征词时直接归类，高置信。"""
    for kp_name, keyword in _KEYWORD_RULES:
        if keyword in stem:
            # 「行政复议」不要误伤「行政诉讼」题
            if keyword == "行政复议" and "行政诉讼" in stem:
                continue
            kp_id = resolve_knowledge_point_id(kp_name, name_to_id)
            if kp_id is not None:
                return ClassifyOutcome(kp_id, 0.95, False, f"关键词「{keyword}」")
    return None


def plan_classification(
    *,
    knowledge_point_name: str,
    confidence: float,
    name_to_id: dict[str, int],
    threshold: float = CLASSIFY_CONFIDENCE_THRESHOLD,
) -> ClassifyOutcome:
    """模型输出 → 写库计划。"""
    kp_id = resolve_knowledge_point_id(knowledge_point_name, name_to_id)
    if kp_id is None:
        return ClassifyOutcome(None, confidence, True, "知识点名无法匹配")
    if confidence < threshold:
        return ClassifyOutcome(kp_id, confidence, True, f"置信度低({confidence:.2f})")
    return ClassifyOutcome(kp_id, confidence, False, "归类成功")


# PDF 文件名 → 考纲章名（按特异性从高到低匹配）
_PDF_CHAPTER_RULES: list[tuple[str, str]] = [
    ("合伙企业法律制度", "合伙企业法律制度"),
    ("物权法律制度", "物权法律制度"),
    ("合同法律制度", "合同法律制度"),
    ("第三章", "合伙企业法律制度"),
    ("第四章", "物权法律制度"),
    ("第五章", "合同法律制度"),
    ("第二章", "公司法律制度"),
    ("第二部分", "公司法律制度"),
    ("第一章", "总论"),
    ("第一部分", "总论"),
    ("总论", "总论"),
]


def infer_chapter_from_source(source_ref: str | None) -> str | None:
    """从 source_ref（pdf名#page=N）推断所属章。"""
    pdf = (source_ref or "").split("#")[0]
    for keyword, chapter in _PDF_CHAPTER_RULES:
        if keyword in pdf:
            return chapter
    return None
