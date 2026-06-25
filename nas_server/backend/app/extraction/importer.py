"""提取编排：把一份 PDF 跑成题库写入 SQLite。

VLM 调用通过 extract_fn 注入，单测里传 mock，不依赖真实 API。
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import fitz

from ..config import settings
from . import pdf
from .schema import QuestionDraft, validate_draft

# 置信度低于此值的题进人工确认队列
CONFIDENCE_THRESHOLD = 0.75

# extract_fn: 整页 PNG 字节 -> 该页题目原始 dict 列表
ExtractFn = Callable[[bytes], list[dict]]


@dataclass
class ImportSummary:
    pdf_name: str
    pages: int = 0
    imported: int = 0          # 成功入库的题数
    needs_review: int = 0      # 其中标记需人工确认的题数
    invalid: int = 0           # schema 校验失败、无法入库的条目数
    errors: list[str] = field(default_factory=list)

    def describe(self) -> str:
        return (
            f"{self.pdf_name}: 共 {self.pages} 页，"
            f"成功识别 {self.imported} 道题，其中 {self.needs_review} 道需人工确认"
            f"（另有 {self.invalid} 条无法解析）。"
        )


def _save_image(png_bytes: bytes, media_dir: Path) -> str:
    """存配图到 media 目录，返回相对 media_dir 的路径。"""
    sub = Path("questions")
    (media_dir / sub).mkdir(parents=True, exist_ok=True)
    rel = sub / f"{uuid.uuid4().hex}.png"
    (media_dir / rel).write_bytes(png_bytes)
    return rel.as_posix()


def _attach_images(
    page: fitz.Page,
    drafts: list[QuestionDraft],
    media_dir: Path,
) -> list[list[str]]:
    """为本页中 has_image 的题裁剪渲染配图，按阅读顺序与内容图框配对。

    返回与 drafts 等长的「每题配图相对路径列表」。
    """
    images_per_draft: list[list[str]] = [[] for _ in drafts]
    img_indices = [i for i, d in enumerate(drafts) if d.has_image]
    if not img_indices:
        return images_per_draft

    rects = pdf.content_image_rects(page)
    # 顺序配对：第 k 道带图题 ← 第 k 个内容图框
    for k, draft_idx in enumerate(img_indices):
        if k >= len(rects):
            break
        png = pdf.crop_render_png(page, rects[k])
        if pdf.is_blank_png(png):
            continue  # 全黑/全空的不存
        images_per_draft[draft_idx] = [_save_image(png, media_dir)]
    return images_per_draft


def _insert_question(
    conn: sqlite3.Connection,
    draft: QuestionDraft,
    images: list[str],
    source: str,
    source_ref: str,
    needs_review: bool,
) -> None:
    conn.execute(
        """
        INSERT INTO questions
            (chapter, exam_point, question_type, difficulty, year, stem,
             options, correct_answer, explanation, images, source, source_ref,
             confidence, needs_review)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            draft.chapter,
            draft.exam_point,
            draft.question_type,
            draft.difficulty,
            draft.year,
            draft.stem,
            json.dumps([o.model_dump() for o in draft.options], ensure_ascii=False),
            json.dumps(draft.correct_answer, ensure_ascii=False),
            draft.explanation,
            json.dumps(images, ensure_ascii=False),
            source,
            source_ref,
            draft.confidence,
            1 if needs_review else 0,
        ),
    )


def import_pdf(
    pdf_path: str | Path,
    extract_fn: ExtractFn,
    conn: sqlite3.Connection,
    media_dir: Path | None = None,
) -> ImportSummary:
    """把一份 PDF 提取入库。extract_fn 注入 VLM 调用，便于 mock。"""
    pdf_path = Path(pdf_path)
    media_dir = media_dir or settings.media_dir
    summary = ImportSummary(pdf_name=pdf_path.name)

    doc = fitz.open(pdf_path)
    summary.pages = len(doc)
    try:
        for page_no in range(len(doc)):
            page = doc[page_no]
            source_ref = f"{pdf_path.name}#page={page_no + 1}"
            try:
                page_png = pdf.render_page_png(page)
                raw_questions = extract_fn(page_png)
            except Exception as e:  # 单页失败不影响其它页
                summary.errors.append(f"page {page_no + 1}: {e}")
                continue

            # 先校验，分出有效草稿与无效条目
            drafts: list[QuestionDraft] = []
            for raw in raw_questions:
                draft, err = validate_draft(raw)
                if draft is None:
                    summary.invalid += 1
                    summary.errors.append(f"page {page_no + 1} 校验失败: {err}")
                else:
                    drafts.append(draft)

            images_per_draft = _attach_images(page, drafts, media_dir)

            for draft, images in zip(drafts, images_per_draft):
                needs_review = draft.confidence < CONFIDENCE_THRESHOLD
                _insert_question(
                    conn,
                    draft,
                    images,
                    source="PDF导入",
                    source_ref=source_ref,
                    needs_review=needs_review,
                )
                summary.imported += 1
                if needs_review:
                    summary.needs_review += 1
        conn.commit()
    finally:
        doc.close()
    return summary
