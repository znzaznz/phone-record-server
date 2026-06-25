"""Markdown 转 docx（轻量，不依赖 LibreOffice）。"""

from __future__ import annotations

import io
import re
from pathlib import Path

from docx import Document

_SAFE_STEM = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}_[^./\\]+$")


def validate_stem(stem: str) -> str:
    stem = (stem or "").strip()
    if not stem or ".." in stem or "/" in stem or "\\" in stem:
        raise ValueError("invalid stem")
    if not _SAFE_STEM.match(stem):
        raise ValueError("invalid stem format")
    return stem


def md_to_docx_bytes(md_text: str) -> bytes:
    doc = Document()
    for raw in md_text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.startswith("# "):
            doc.add_heading(line[2:].strip(), level=1)
        elif line.startswith("## "):
            doc.add_heading(line[3:].strip(), level=2)
        elif line.startswith("### "):
            doc.add_heading(line[4:].strip(), level=3)
        elif line.startswith("- "):
            doc.add_paragraph(line[2:].strip(), style="List Bullet")
        elif line.startswith("---"):
            continue
        else:
            doc.add_paragraph(line)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def md_file_to_docx_bytes(md_path: Path) -> bytes:
    return md_to_docx_bytes(md_path.read_text(encoding="utf-8"))