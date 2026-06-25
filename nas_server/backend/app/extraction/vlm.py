"""视觉大模型客户端：整页图 → 该页题目的结构化 JSON。

这一层把真实 API 调用隔离起来，单测里整体 mock 掉（不打真实 API）。
"""

from __future__ import annotations

import base64
import json

import httpx

from ..config import settings

# 给 VLM 的指令：逐页识题，返回严格 JSON（PDF 批量导入用）。
_SYSTEM_PROMPT = """你是一个会计考试题目识别助手。我会给你一页讲义/真题的整页截图。
请识别这一页上的所有题目，按从上到下顺序，输出**严格 JSON**，形如：
{"questions": [
  {
    "stem": "题干文本",
    "question_type": "单选" | "多选" | "判断",
    "options": [{"key": "A", "text": "选项内容"}, ...],
    "correct_answer": ["A"],
    "explanation": "解析文本，没有就留空字符串",
    "chapter": "所属章节，如 第一章 总论",
    "exam_point": "所属考点",
    "year": "年份，如 2023，识别不出就 null",
    "has_image": true/false,
    "confidence": 0.0~1.0
  }
]}
要求：
- 只输出**一个** JSON 对象，不要任何额外说明、不要 markdown 代码块、JSON 之后不要再输出任何文字。
- **stem 只放题干本身，绝对不要把 A/B/C/D 选项文字写进 stem**；选项一律只放在 options 数组里。
- 题干里若带题号（如「5.」「17.【单选·2024】」）可保留，但选项必须从题干剥离。
- year 用字符串，如 "2024"；识别不出用 null。
- 若该页没有完整题目（如纯讲义/目录页），返回 {"questions": []}。
- 解析（explanation）只在本页确实出现时填写，没有就留空字符串，不要编造。
- has_image 表示该题是否配有图示/表格。
- confidence 反映你对识别准确度的把握，模糊不清的题给低分。"""

# 错题截图：只做 OCR 转写，禁止推断答案/解析。
_SCREENSHOT_OCR_PROMPT = """你是 OCR 转写助手。给你一张考试题目截图，请**只做文字转写**，不要做题、不要推断答案、不要补充解析。

只转写图片上**肉眼可见**的文字，输出严格 JSON：
{"questions": [
  {
    "stem": "题干可见文字",
    "question_type": "单选" | "多选" | "判断" | null,
    "options": [{"key": "A", "text": "选项可见文字"}, ...],
    "correct_answer": [],
    "explanation": "",
    "chapter": "",
    "exam_point": "",
    "year": "2022" | null,
    "has_image": true | false,
    "confidence": 0.0~1.0,
    "answer_visible": true | false,
    "explanation_visible": true | false
  }
]}

硬性要求：
- **绝对禁止**根据会计知识推断正确答案或编写解析
- correct_answer **仅当**图上有明确答案标注（如「答案：B」「正确答案 BD」、选项旁明文标出）才可填写；否则必须 []
- explanation **仅当**图上有「解析」「答案解析」等段落才可填写；否则 ""
- answer_visible / explanation_visible 如实标注图上是否看得见答案/解析
- 忽略视频播放器按钮、字幕、水印、UI 浮层等无关文字
- stem 只放题干；选项放进 options，不要合并进 stem
- 黄色圈划/手写标记**不算**答案，除非旁边有明文「答案」
- 看不清的字用「□」占位，不要猜
- 只输出一个 JSON 对象，不要 markdown、不要额外说明"""


class VLMError(RuntimeError):
    pass


def _build_client():
    # 延迟导入，避免无 key 环境下 import 即失败
    from openai import OpenAI

    if not settings.dashscope_api_key.strip():
        raise VLMError("未配置 DASHSCOPE_API_KEY，无法调用视觉模型")
    return OpenAI(
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
    )


def _extract_questions_ollama(
    page_png: bytes,
    system_prompt: str = _SYSTEM_PROMPT,
    user_text: str = "识别这一页上的所有题目。只输出 JSON。",
) -> list[dict]:
    b64 = base64.b64encode(page_png).decode("ascii")
    prompt = system_prompt + "\n\n" + user_text
    url = settings.ollama_base_url.rstrip("/") + "/api/generate"
    try:
        with httpx.Client(trust_env=False, timeout=180) as http:
            resp = http.post(
                url,
                json={
                    "model": settings.ollama_vl_model,
                    "prompt": prompt,
                    "images": [b64],
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.1},
                },
            )
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise VLMError(f"Ollama 调用失败: {e}") from e

    data = resp.json()
    content = data.get("response") or data.get("thinking") or ""
    if not content:
        raise VLMError("Ollama 返回为空")
    return _parse_questions_json(content)


def _parse_questions_json(content: str) -> list[dict]:
    text = content.strip()
    # 容忍模型偶尔包了 ```json ``` 代码块
    if text.startswith("```"):
        text = text.strip("`")
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    text = text.strip()
    # 容忍 JSON 后面跟了多余文字（"Extra data"）：只取第一个完整 JSON 值
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data, _ = json.JSONDecoder().raw_decode(text)
    questions = data.get("questions", data if isinstance(data, list) else [])
    if not isinstance(questions, list):
        raise VLMError(f"VLM 返回的 questions 不是列表: {type(questions)}")
    return questions


def extract_questions(page_png: bytes, client=None) -> list[dict]:
    """对一页整页图调用 VLM，返回该页题目的原始 dict 列表（未校验）。PDF 导入用。"""
    return _call_vlm(page_png, _SYSTEM_PROMPT, "识别这一页上的所有题目。", client)


def extract_screenshot(page_png: bytes, client=None) -> list[dict]:
    """错题截图 OCR 转写：只提取可见文字，不推断答案/解析。"""
    return _call_vlm(
        page_png,
        _SCREENSHOT_OCR_PROMPT,
        "请 OCR 转写截图上的题目文字，严格遵守不做推断。",
        client,
    )


def _call_vlm(page_png: bytes, system_prompt: str, user_text: str, client=None) -> list[dict]:
    if settings.vlm_provider.lower() == "ollama":
        return _extract_questions_ollama(page_png, system_prompt, user_text)

    client = client or _build_client()
    b64 = base64.b64encode(page_png).decode("ascii")
    resp = client.chat.completions.create(
        model=settings.qwen_vl_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            },
        ],
        temperature=0.1,
    )
    content = resp.choices[0].message.content or ""
    return _parse_questions_json(content)
