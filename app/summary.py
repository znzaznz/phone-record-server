"""转写稿 → 类型识别 + 结构化总结（Gemini 3.5 Flash，仅此项走代理）。

Prompt 结构参考：
- BrassTranscripts Meeting Summary（可执行信息 > 闲聊）
- 飞书/讯飞会议纪要（议题/决议/待办三分法）
- UX 访谈分析（洞察 + 原话引用）
"""

from __future__ import annotations

import json
import logging
import re

import httpx
from openai import OpenAI

from app.config import settings
from app.summary_templates import (
    FALLBACK_EXTRA,
    FALLBACK_OUTLINE,
    FALLBACK_TYPE,
    TEMPLATES,
)

logger = logging.getLogger(__name__)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _build_system_prompt() -> str:
    type_names = "、".join(list(TEMPLATES) + [FALLBACK_TYPE])
    return "\n".join([
        "你是专业的中文录音转写分析助手。输入是语音转文字逐字稿，可能含 Speaker1/说话人 等标记。",
        "",
        "## 你的任务",
        f"1. 判断录音类型，从 [{type_names}] 中选最匹配的一个",
        "2. 按该类型的章节结构输出 markdown 总结",
        "",
        "## 类型判定",
        *[f"- **{n}**：{s['hint']}" for n, s in TEMPLATES.items()],
        f"- **{FALLBACK_TYPE}**：{FALLBACK_EXTRA}",
        "",
        "## 各类型输出结构（summary 必须包含这些 ## 章节）",
        *(
            line
            for n, s in TEMPLATES.items()
            for line in (
                f"### {n}",
                s["outline"],
                f"写作要求：{s['extra']}" if s.get("extra") else "",
                "",
            )
        ),
        f"### {FALLBACK_TYPE}",
        FALLBACK_OUTLINE,
        "",
        "## 全局规则",
        "- 只依据逐字稿，不编造人名、数字、结论、未出现的决策",
        "- 忽略寒暄、重复、口误和无信息量的填充语",
        "- 有说话人标记时，待办和决议尽量标注负责人",
        "- 信息缺失写「未明确」，不要猜测",
        "- 中文，简洁可读，偏纪要而非散文",
        "- 待办格式：`- 负责人 / 事项 / 截止时间`",
        "",
        "## 输出格式（严格遵守）",
        "只输出一行 JSON，不要 markdown 代码块，不要其它文字：",
        '{"type":"会议","summary":"## 会议概览\\n..."}',
        f"type 必须是 [{type_names}] 之一；summary 是完整 markdown 正文。",
    ])


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
    """返回 (type, summary_md)；失败返回 None。"""
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
            max_tokens=2500,
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