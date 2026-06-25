"""转写稿 → 一个中文短标题（用作文件名）。一次小的 LLM 调用（百炼 qwen 文本模型，
OpenAI 兼容接口）。失败时降级用逐字稿首句，绝不因为取名失败而丢掉转写结果。

注：总结报告功能待定，这里只负责「取名」。
"""

from __future__ import annotations

import logging
import re

from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

_ILLEGAL = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_PUNCT = re.compile(r"[\s。，、；：！？「」『』（）()\[\]【】《》\"'`~!@#$%^&*+=,.;:?/]+")


def sanitize_filename(name: str, max_len: int = 40) -> str:
    """清成能当文件名的串：去非法字符、空白→下划线、限长。保留中文。"""
    name = _ILLEGAL.sub("", name).strip().strip(".")
    name = re.sub(r"\s+", "_", name)
    name = name[:max_len].strip("_")
    return name or "未命名"


def _fallback_title(transcript: str) -> str:
    """取名失败时的兜底：逐字稿里第一段有意义的文字，截前 16 字。"""
    text = _PUNCT.sub(" ", transcript).strip()
    first = text.split()[0] if text.split() else ""
    return (first[:16] or "未命名").strip()


def generate_title(transcript: str) -> str:
    """让模型起一个 6-16 字中文短标题；任何异常都降级到 _fallback_title。"""
    snippet = (transcript or "").strip()
    if not snippet:
        return "未命名"
    sys_prompt = (
        "你给一段录音转写稿起标题。只输出一个 6 到 16 字的中文短标题，概括这段录音的"
        "核心内容，能直接当文件名。不要日期、不要标点符号、不要书名号引号、不要任何解释，"
        "只输出标题本身。"
    )
    try:
        client = OpenAI(api_key=settings.dashscope_api_key, base_url=settings.title_base_url)
        resp = client.chat.completions.create(
            model=settings.title_model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": snippet[: settings.title_max_chars]},
            ],
            temperature=0.3,
            max_tokens=40,
        )
        title = (resp.choices[0].message.content or "").strip()
        title = title.splitlines()[0].strip() if title else ""
        if not title:
            raise ValueError("empty title from model")
        return title
    except Exception as e:  # noqa: BLE001 — 取名失败不该影响主流程
        logger.warning("生成标题失败，降级用首句: %s", e)
        return _fallback_title(snippet)
