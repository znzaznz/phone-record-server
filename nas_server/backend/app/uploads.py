"""S10 截图上传识题：草稿确认入库。"""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from pathlib import Path

from .extraction.schema import QuestionDraft, validate_draft

UPLOAD_SOURCE = "截图上传"

# 去空格/数字/标点，只留文字（含中文），用于题干判重
_NORM_RE = re.compile(r"[\s\d\W]+", re.UNICODE)


def _normalize_stem(s: str) -> str:
    return _NORM_RE.sub("", s or "")


def find_duplicate_question(conn: sqlite3.Connection, stem: str) -> int | None:
    """题库里已有同题干的题则返回其 id（归一化后全等比较）。"""
    target = _normalize_stem(stem)
    if not target:
        return None
    # ponytail: 全表 O(n) 扫描，几千题够用；量级到十万级再加 normalized_stem 索引列
    for r in conn.execute("SELECT id, stem FROM questions"):
        if _normalize_stem(r["stem"]) == target:
            return int(r["id"])
    return None


def _upsert_mistake_on_upload(
    conn: sqlite3.Connection, qid: int, correct_answer: list[str], now: str
) -> None:
    """上传错题入错题本：已有则累加一次做错，没有则新建。"""
    row = conn.execute(
        "SELECT question_id FROM mistakes WHERE question_id = ?", (qid,)
    ).fetchone()
    if row is not None:
        conn.execute(
            "UPDATE mistakes SET wrong_count = wrong_count + 1, last_attempt_at = ? "
            "WHERE question_id = ?",
            (now, qid),
        )
        return
    conn.execute(
        """
        INSERT INTO mistakes
            (question_id, wrong_answer, correct_answer, wrong_count,
             correct_count, first_wrong_at, last_attempt_at, mastery, favorite)
        VALUES (?, '[]', ?, 1, 0, ?, ?, '未掌握', 0)
        """,
        (qid, json.dumps(correct_answer, ensure_ascii=False), now, now),
    )


def save_upload_image(data: bytes, media_dir: Path, suffix: str = ".png") -> str:
    """存上传原图，返回相对 media_dir 的路径。"""
    sub = Path("uploads")
    (media_dir / sub).mkdir(parents=True, exist_ok=True)
    rel = sub / f"{uuid.uuid4().hex}{suffix}"
    (media_dir / rel).write_bytes(data)
    return rel.as_posix()


def insert_draft(conn: sqlite3.Connection, image_path: str, raw: dict) -> int:
    cur = conn.execute(
        """
        INSERT INTO upload_drafts (image_path, draft_json, confidence)
        VALUES (?, ?, ?)
        """,
        (
            image_path,
            json.dumps(raw, ensure_ascii=False),
            float(raw.get("confidence", 0.5)),
        ),
    )
    conn.commit()
    return cur.lastrowid


def get_draft(conn: sqlite3.Connection, draft_id: int) -> dict | None:
    row = conn.execute(
        "SELECT id, image_path, draft_json, confidence FROM upload_drafts WHERE id = ?",
        (draft_id,),
    ).fetchone()
    if row is None:
        return None
    draft = json.loads(row["draft_json"])
    return {
        "id": row["id"],
        "image_path": row["image_path"],
        "confidence": row["confidence"],
        **draft,
    }


def confirm_upload(
    conn: sqlite3.Connection,
    draft_id: int,
    draft: QuestionDraft,
    *,
    knowledge_point_id: int | None,
    image_path: str,
    now: str,
) -> dict:
    """确认草稿入库并写入错题本。返回 {question_id, duplicate}。

    - 题库已有同题干：不重复入库，只给已有题累加一次错题计数。
    - 没选知识点：needs_review=1，进待核对队列（否则这道没掌握的题会隐身）。
    """
    dup_id = find_duplicate_question(conn, draft.stem)
    if dup_id is not None:
        _upsert_mistake_on_upload(conn, dup_id, draft.correct_answer, now)
        conn.execute("DELETE FROM upload_drafts WHERE id = ?", (draft_id,))
        conn.commit()
        return {"question_id": dup_id, "duplicate": True}

    needs_review = 1 if knowledge_point_id is None else 0
    cur = conn.execute(
        """
        INSERT INTO questions
            (chapter, exam_point, question_type, difficulty, year, stem,
             options, correct_answer, explanation, images, source, source_ref,
             confidence, needs_review, knowledge_point_id)
        VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            draft.chapter,
            draft.exam_point,
            draft.question_type,
            draft.difficulty,
            draft.stem,
            json.dumps([o.model_dump() for o in draft.options], ensure_ascii=False),
            json.dumps(draft.correct_answer, ensure_ascii=False),
            draft.explanation or "",
            json.dumps([image_path], ensure_ascii=False),
            UPLOAD_SOURCE,
            f"upload_draft:{draft_id}",
            draft.confidence,
            needs_review,
            knowledge_point_id,
        ),
    )
    qid = cur.lastrowid
    _upsert_mistake_on_upload(conn, qid, draft.correct_answer, now)
    conn.execute("DELETE FROM upload_drafts WHERE id = ?", (draft_id,))
    conn.commit()
    return {"question_id": qid, "duplicate": False}


def draft_from_confirm_body(body: dict) -> QuestionDraft:
    d, err = validate_draft(body)
    if d is None:
        raise ValueError(err)
    return d
