"""上传页错题截图专用管线：VLM OCR 转写 → 文本模型归档/补全。

仅由 POST /uploads 调用；PDF 批量导入仍走 vlm.extract_questions。
确认入库仍走 uploads.confirm_upload（错题本、知识点挂接等逻辑不变）。
"""

from __future__ import annotations

import json
import sqlite3

from .classify import plan_classification
from .extraction import vlm
from .llm import classify_question, enrich_missing_fields, infer_subject, polish_ocr_question


def format_ocr_question_text(raw: dict) -> str:
    opts = raw.get("options") or []
    if isinstance(opts, str):
        opts = json.loads(opts)
    lines = [f"题干：{raw.get('stem', '')}"]
    if opts:
        lines.append("选项：")
        for o in opts:
            if isinstance(o, dict):
                lines.append(f"  {o.get('key', '?')}. {o.get('text', '')}")
    return "\n".join(lines)


def list_knowledge_points_by_subject(conn: sqlite3.Connection, subject: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT k.id, k.name, k.essence, e.chapter, e.name AS exam_point
        FROM knowledge_points k
        JOIN exam_points e ON e.id = k.exam_point_id
        WHERE e.subject = ?
        ORDER BY k.seq
        """,
        (subject,),
    ).fetchall()
    return [dict(r) for r in rows]


def _narrow_by_chapter_hint(kps: list[dict], chapter_hint: str) -> list[dict]:
    hint = (chapter_hint or "").strip()
    if not hint:
        return kps
    matched = [
        k
        for k in kps
        if hint in (k.get("chapter") or "") or (k.get("chapter") or "") in hint
    ]
    return matched if matched else kps


_JUDGE_MAP = {
    "对": "对", "正确": "对", "√": "对", "T": "对", "t": "对", "Y": "对",
    "错": "错", "错误": "错", "×": "错", "F": "错", "f": "错", "N": "错",
}


def _normalize_judge_answers(answers: list[str]) -> list[str]:
    """判断题答案统一成 对/错（前端只认这两个值），无法映射的丢弃。"""
    out: list[str] = []
    for a in answers:
        v = _JUDGE_MAP.get(str(a).strip())
        if v and v not in out:
            out.append(v)
    return out


def _default_question_type(raw: dict) -> str:
    qt = raw.get("question_type")
    if qt in ("单选", "多选", "判断"):
        return qt
    ans = raw.get("correct_answer") or []
    if isinstance(ans, str):
        ans = [ans]
    if len(ans) > 1:
        return "多选"
    opts = raw.get("options") or []
    if (
        len(opts) == 2
        and all(
            isinstance(o, dict) and o.get("key") in ("对", "错") for o in opts
        )
    ):
        return "判断"
    return "单选"


def process_screenshot_upload(conn: sqlite3.Connection, image_bytes: bytes) -> dict:
    """上传页：OCR → 润色 → 分科 → 知识点归类 → 补缺答案/解析。"""
    ocr_list = vlm.extract_screenshot(image_bytes)
    if not ocr_list:
        raise ValueError("未识别到题目")
    ocr = polish_ocr_question(ocr_list[0])

    question_text = format_ocr_question_text(ocr)
    chapter_hint = str(ocr.get("chapter") or "").strip()

    subj = infer_subject(question_text, chapter_hint)
    subject = subj["subject"]

    kps = list_knowledge_points_by_subject(conn, subject)
    candidates = _narrow_by_chapter_hint(kps, chapter_hint)

    chapter_for_disambig = chapter_hint or (
        (candidates[0].get("chapter") or "") if candidates else ""
    )
    cls = classify_question(
        question_text,
        candidates,
        subject=subject,
        chapter=chapter_for_disambig or None,
    )
    name_to_id = {k["name"]: k["id"] for k in kps}
    outcome = plan_classification(
        knowledge_point_name=cls["knowledge_point_name"],
        confidence=cls["confidence"],
        name_to_id=name_to_id,
    )

    kp_id = outcome.knowledge_point_id
    kp_meta = next((k for k in kps if k["id"] == kp_id), None) if kp_id else None
    kp_name = (kp_meta or {}).get("name") or cls["knowledge_point_name"]
    essence = (kp_meta or {}).get("essence") or ""

    qtype = _default_question_type(ocr)
    ocr_answers = ocr.get("correct_answer") or []
    if isinstance(ocr_answers, str):
        ocr_answers = [ocr_answers]
    ocr_expl = str(ocr.get("explanation") or "")
    answer_visible = bool(ocr.get("answer_visible")) and bool(ocr_answers)
    explanation_visible = bool(ocr.get("explanation_visible")) and bool(ocr_expl.strip())

    enriched = enrich_missing_fields(
        question_text,
        qtype,
        subject=subject,
        knowledge_point_name=kp_name,
        essence=essence,
        correct_answer=list(ocr_answers) if answer_visible else [],
        explanation=ocr_expl,
        answer_visible=answer_visible,
        explanation_visible=explanation_visible,
    )

    options = ocr.get("options") or []
    correct_answer = enriched["correct_answer"]
    if qtype == "判断":
        # 判断题强制规整：选项 对/错、答案归一，否则前端按钮判不对（死题）
        options = [{"key": "对", "text": "对"}, {"key": "错", "text": "错"}]
        correct_answer = _normalize_judge_answers(correct_answer)

    return {
        "stem": str(ocr.get("stem") or "").strip(),
        "question_type": qtype,
        "options": options,
        "correct_answer": correct_answer,
        "explanation": enriched["explanation"],
        "chapter": (kp_meta or {}).get("chapter") or chapter_hint or None,
        "exam_point": (kp_meta or {}).get("exam_point") or ocr.get("exam_point") or None,
        "year": ocr.get("year"),
        "confidence": float(ocr.get("confidence", 0.5)),
        "subject": subject,
        "knowledge_point_id": kp_id,
        "classify_confidence": cls["confidence"],
        "needs_review": outcome.needs_review or kp_id is None,
        "classify_note": outcome.note,
        "answer_inferred": enriched["answer_inferred"],
        "explanation_generated": enriched["explanation_generated"],
        "answer_visible": answer_visible,
        "explanation_visible": explanation_visible,
        "text_polished": bool(ocr.get("text_polished")),
    }
