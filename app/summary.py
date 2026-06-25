"""转写稿 → 类型识别 + 结构化总结（Gemini，仅此项走代理）。"""

from __future__ import annotations

import json
import logging
import re

import httpx
from openai import OpenAI

from app.config import settings
from app.summary_templates import FALLBACK_OUTLINE, FALLBACK_TYPE, TEMPLATES

logger = logging.getLogger(__name__)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _build_system_prompt() -> str:
    lines = [
        "你是录音转写稿分析助手。用户会给你一段可能带说话人标记的逐字稿。",
        "",
        "任务：",
        "1. 从下列类型中选最匹配的一个（只选一个）："
        + "、".join(list(TEMPLATES) + [FALLBACK_TYPE]),
        "2. 严格按该类型的章节结构，写一份中文 markdown 总结",
        "",
        "类型判定线索：",
    ]
    for name, spec in TEMPLATES.items():
        lines.append(f"- {name}：{spec['hint']}")
    lines.append(f"- {FALLBACK_TYPE}：以上都不像时使用")
    lines.append("")
    lines.append("各类型章节结构（summary 必须按此组织，章节标题保留 ##）：")
    for name, spec in TEMPLATES.items():
        lines.append(f"- {name}：{spec['outline'].replace(chr(10), ' / ')}")
    lines.append(f"- {FALLBACK_TYPE}：{FALLBACK_OUTLINE.replace(chr(10), ' / ')}")
    lines.extend(
        [
            "",
            "硬性要求：",
            "- 只依据逐字稿事实，不编造人名、数字、结论",
            "- 信息不足写「未明确」，不要猜",
            "- 待办尽量写清 谁/做什么/何时；稿中没有就写「未明确」",
            "- 简洁、可读，中文",
            "",
            "只输出一行 JSON，不要 markdown 代码块，不要其它文字：",
            '{"type":"会议","summary":"## 议题\\n..."}',
            "type 必须是上面列出的类型名之一；summary 是完整 markdown 正文。",
        ]
    )
    return "\n".join(lines)


def _parse_json(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_RE.search(text)
        if not m:
            raise
        return json.loads(m.group(0))


def _gemini_client() -> OpenAI:
    kwargs: dict = {
        "api_key": settings.google_api_key,
        "base_url": settings.summary_base_url,
    }
    proxy = (settings.summary_proxy_url or "").strip()
    if proxy:
        kwargs["http_client"] = httpx.Client(proxy=proxy, timeout=120.0)
    return OpenAI(**kwargs)


def generate_summary(transcript: str, title: str = "") -> tuple[str, str] | None:
    """返回 (type, summary_md)；失败返回 None，不抛异常。"""
    if not settings.summary_enabled:
        return None
    if not settings.google_api_key:
        logger.warning("未配置 GOOGLE_API_KEY，跳过总结")
        return None
    snippet = (transcript or "").strip()
    if not snippet:
        return None

    user = f"录音标题（参考）：{title or '未命名'}\n\n---\n\n{snippet[: settings.summary_max_chars]}"
    try:
        client = _gemini_client()
        resp = client.chat.completions.create(
            model=settings.summary_model,
            messages=[
                {"role": "system", "content": _build_system_prompt()},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=2048,
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = _parse_json(raw)
        stype = str(data.get("type") or FALLBACK_TYPE).strip()
        summary = str(data.get("summary") or "").strip()
        if stype not in TEMPLATES and stype != FALLBACK_TYPE:
            stype = FALLBACK_TYPE
        if not summary:
            raise ValueError("empty summary")
        return stype, summary
    except Exception as e:  # noqa: BLE001
        logger.warning("生成总结失败，跳过: %s", e)
        return None