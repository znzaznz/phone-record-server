"""S8 相似题：入库与删除（纯逻辑）。"""

from __future__ import annotations

import json
import sqlite3

from .extraction.schema import QuestionDraft, validate_draft

SIMILAR_SOURCE = "相似题生成"


def similar_source_ref(origin_question_id: int) -> str:
    return f"similar_of:{origin_question_id}"


def get_saved_similar_question_id(
    conn: sqlite3.Connection, origin_question_id: int
) -> int | None:
    """该错题已保存的相似题 id（取最新一条）。"""
    row = conn.execute(
        """
        SELECT id FROM questions
        WHERE source = ? AND source_ref = ?
        ORDER BY id DESC LIMIT 1
        """,
        (SIMILAR_SOURCE, similar_source_ref(origin_question_id)),
    ).fetchone()
    return int(row["id"]) if row else None


def delete_saved_similar_for_origin(
    conn: sqlite3.Connection, origin_question_id: int
) -> int:
    """删除某道原题下的全部已存相似题，返回删除条数。"""
    rows = conn.execute(
        "SELECT id FROM questions WHERE source = ? AND source_ref = ?",
        (SIMILAR_SOURCE, similar_source_ref(origin_question_id)),
    ).fetchall()
    for r in rows:
        delete_similar_question(conn, int(r["id"]))
    return len(rows)


def insert_similar_question(
    conn: sqlite3.Connection,
    draft: QuestionDraft,
    *,
    knowledge_point_id: int,
    origin_question_id: int,
    chapter: str | None = None,
    exam_point: str | None = None,
) -> int:
    """把模型生成的相似题写入题库，返回新题 id。"""
    cur = conn.execute(
        """
        INSERT INTO questions
            (chapter, exam_point, question_type, difficulty, year, stem,
             options, correct_answer, explanation, images, source, source_ref,
             confidence, needs_review, knowledge_point_id)
        VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?, '[]', ?, ?, 1.0, 0, ?)
        """,
        (
            chapter,
            exam_point,
            draft.question_type,
            draft.difficulty,
            draft.stem,
            json.dumps([o.model_dump() for o in draft.options], ensure_ascii=False),
            json.dumps(draft.correct_answer, ensure_ascii=False),
            draft.explanation or "",
            SIMILAR_SOURCE,
            similar_source_ref(origin_question_id),
            knowledge_point_id,
        ),
    )
    conn.commit()
    return cur.lastrowid


def draft_from_llm(raw: dict) -> QuestionDraft:
    draft, err = validate_draft(raw)
    if draft is None:
        raise ValueError(f"相似题 schema 校验失败: {err}")
    return draft


def normalize_similar_draft(draft: QuestionDraft, *, question_type: str) -> QuestionDraft:
    """强制与原错题题型一致；判断题补全对/错选项。"""
    from .extraction.schema import Option

    opts = list(draft.options)
    if question_type == "判断" and not opts:
        opts = [Option(key="对", text="对"), Option(key="错", text="错")]
    return draft.model_copy(update={"question_type": question_type, "options": opts})


def delete_similar_question(conn: sqlite3.Connection, question_id: int) -> bool:
    """仅允许删除 AI 相似题。返回是否删除成功。"""
    row = conn.execute(
        "SELECT source FROM questions WHERE id = ?", (question_id,)
    ).fetchone()
    if row is None or row["source"] != SIMILAR_SOURCE:
        return False
    conn.execute("DELETE FROM attempts WHERE question_id = ?", (question_id,))
    conn.execute("DELETE FROM mistakes WHERE question_id = ?", (question_id,))
    conn.execute("DELETE FROM questions WHERE id = ?", (question_id,))
    conn.commit()
    return True
