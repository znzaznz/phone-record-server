"""文本大模型客户端：题目归类、相似题生成等。

真实 API 调用隔离在此层，单测里整体 mock。
"""

from __future__ import annotations

import json

from .config import settings

SUBJECTS = ("中级会计实务", "财务管理", "经济法")

_CLASSIFY_PROMPT = """你是中级会计《{subject}》题目分类助手。
把下面这道题归类到本科的**唯一一个**知识点。

可选知识点（knowledge_point_name 必须与下列名称完全一致）：
{candidates}
{disambiguation}

题目：
{question}

只输出一个 JSON 对象，不要 markdown、不要额外说明：
{{"knowledge_point_name": "...", "confidence": 0.0~1.0}}
confidence 反映你对归类准确度的把握。"""

_INFER_SUBJECT_PROMPT = """你是中级会计考试分科助手。根据题目内容和可见章节标题，判断属于哪一科。

科目只能是：中级会计实务、财务管理、经济法

可见章节/讲义标题（可能为空）：{chapter_hint}

题目：
{question}

只输出 JSON：{{"subject": "中级会计实务|财务管理|经济法", "confidence": 0.0~1.0}}"""

_POLISH_PROMPT = """你是中级会计考试题目的 OCR 校对编辑。下面是视觉模型从截图转写的题目 JSON，可能有缺字、错字、选项错位或噪声。

OCR 原始 JSON：
{ocr_json}

任务：**只做文字润色与结构整理**，输出规范、可入库的题目 JSON。

要求：
1. 修正 OCR 缺字/错字/乱码（如「所得税付」→「所得税收付」），「□」占位可结合上下文合理补全
2. stem 只保留题干，不得把 A/B/C/D 选项文字塞进 stem
3. 单选/多选：options 规整为 4 项，key 依次为 A/B/C/D，text 非空；判断题：对/错 两项
4. question_type：OCR 为 null 时，根据题干「单选/多选/判断」或选项结构推断
5. correct_answer：仅当 OCR 里 answer_visible 为 true 且有内容时润色保留；否则必须 []
6. explanation：仅当 OCR 里 explanation_visible 为 true 且有内容时润色保留；否则必须 ""
7. chapter / exam_point / year 只润色 OCR 已有字段，不要凭空编造
8. **禁止**根据会计知识新写答案或解析

只输出 JSON，不要 markdown：
{{
  "stem": "...",
  "question_type": "单选" | "多选" | "判断",
  "options": [{{"key": "A", "text": "..."}}],
  "correct_answer": [],
  "explanation": "",
  "chapter": "",
  "exam_point": "",
  "year": "2022" | null
}}"""

_ENRICH_PROMPT = """你是中级会计《{subject}》教研助手。下面是从截图 OCR 转写的一道题（答案/解析可能缺失）。

知识点：{knowledge_point_name}
知识点要义：{essence}

题目：
{question}

题型：{question_type}
图上已有答案：{has_answer}
图上已有解析：{has_explanation}

任务：
1. 若图上**没有**答案，给出正确答案（correct_answer）
2. 若图上**没有**解析，写一段简洁解析（80~200字）
3. 若图上已有答案/解析，原样保留在输出里，不要改
4. 若题型为「判断」，correct_answer 必须是 ["对"] 或 ["错"]，不得用 正确/错误/√/× 或 A/B

只输出 JSON，不要 markdown：
{{
  "correct_answer": ["A"],
  "explanation": "...",
  "answer_inferred": true,
  "explanation_generated": true
}}
answer_inferred / explanation_generated 表示该字段是否由你补充（图上已有则为 false）。"""

_CHAPTER_DISAMBIGUATION: dict[str, str] = {
    "总论": """
⚠️ 重要消歧规则（按考纲归类，别被"民事/民法"字样带偏）：
- **诉讼时效**（时效期间、中止、中断、起算、届满）→ 归「民事诉讼法律制度的规定」，**不是**「法律行为制度」。
- 管辖、上诉、再审、审判监督、两审终审、调解 → 「民事诉讼法律制度的规定」。
- 法律行为的成立/生效/无效/可撤销/效力待定、附条件附期限 → 「法律行为制度」。
- 代理（委托/法定/无权/表见/代理终止）→ 「代理制度」。
- 仲裁协议/范围/一裁终局/裁决 → 「仲裁法律制度的规定」。""",
}

_SIMILAR_PROMPT = """你是中级会计《经济法》命题助手。

任务：针对**下面这道原错题**，生成**一道新的**练习题。

硬性约束（违反任何一条都视为失败）：
1. **题型必须是「{question_type}」**，JSON 里 question_type 填「{question_type}」，不得改成其他题型。
   - 单选：4 个选项，correct_answer 仅 1 个字母
   - 多选：4 个选项，correct_answer 至少 2 个字母
   - 判断：options 为 [{{"key":"对","text":"对"}}, {{"key":"错","text":"错"}}]，correct_answer 为 ["对"] 或 ["错"]
2. **考点锁定为「{knowledge_point_name}」**，只考这个知识点，禁止换成同章其他知识点。
3. **必须紧扣本道原错题的考查意图**（看原题问的是什么、考的是哪条规则/哪种情形），在此基础上换案情、换选项、换问法。
   - 禁止照抄原题
   - 禁止出一道「只是同知识点、但与原题考查角度无关」的题（例如原题考「基本原则」，不要换成「哪些纠纷可仲裁」）
4. stem 只放题干；选项放 options；必须自带 explanation。

知识点要义（命题边界，勿超出）：
{essence}

原错题（你的出题锚点，必须在此基础上变形）：
{original}

⚠️ 以下题目本知识点已经出过，**禁止与它们雷同**（必须换案情主体、换数字、换设问角度，不得只改个别词）：
{avoid}

只输出一个 JSON 对象，不要 markdown：
{{"stem": "...", "question_type": "{question_type}", "options": [{{"key":"A","text":"..."}}], "correct_answer": ["A"], "explanation": "..."}}"""


class LLMError(RuntimeError):
    pass


def _build_client():
    from openai import OpenAI

    if not settings.dashscope_api_key.strip():
        raise LLMError("未配置 DASHSCOPE_API_KEY，无法调用文本模型")
    return OpenAI(
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
    )


def _parse_json_object(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data, _ = json.JSONDecoder().raw_decode(text)
    if not isinstance(data, dict):
        raise LLMError(f"期望 JSON 对象，得到 {type(data)}")
    return data


def classify_question(
    question_text: str,
    candidates: list[dict],
    *,
    subject: str = "经济法",
    chapter: str | None = None,
    client=None,
) -> dict:
    """调用文本模型归类一题。返回 {knowledge_point_name, confidence}。"""
    client = client or _build_client()
    lines = []
    for c in candidates:
        hint = (c.get("essence") or "")[:120].replace("\n", " ")
        ch = c.get("chapter") or ""
        label = f"{ch} · {c['name']}" if ch else c["name"]
        lines.append(f"- {label}" + (f"（{hint}…）" if hint else ""))
    disambiguation = ""
    if subject == "经济法" and chapter:
        disambiguation = _CHAPTER_DISAMBIGUATION.get(chapter, "")
    prompt = _CLASSIFY_PROMPT.format(
        subject=subject,
        candidates="\n".join(lines),
        disambiguation=disambiguation,
        question=question_text,
    )
    resp = client.chat.completions.create(
        model=settings.qwen_text_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    content = resp.choices[0].message.content or ""
    data = _parse_json_object(content)
    return {
        "knowledge_point_name": str(data.get("knowledge_point_name", "")).strip(),
        "confidence": float(data.get("confidence", 0)),
    }


def infer_subject(
    question_text: str,
    chapter_hint: str = "",
    *,
    client=None,
) -> dict:
    """判断题目所属科目。返回 {subject, confidence}。"""
    client = client or _build_client()
    prompt = _INFER_SUBJECT_PROMPT.format(
        chapter_hint=chapter_hint or "（无）",
        question=question_text,
    )
    resp = client.chat.completions.create(
        model=settings.qwen_text_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    content = resp.choices[0].message.content or ""
    data = _parse_json_object(content)
    subject = str(data.get("subject", "")).strip()
    if subject not in SUBJECTS:
        for s in SUBJECTS:
            if s in subject:
                subject = s
                break
        else:
            subject = "经济法"
    return {"subject": subject, "confidence": float(data.get("confidence", 0))}


def polish_ocr_question(ocr: dict, *, client=None) -> dict:
    """润色 OCR 结果：补全缺字、整理题干与选项结构（不编造答案/解析）。"""
    client = client or _build_client()
    ocr_payload = {
        k: ocr.get(k)
        for k in (
            "stem",
            "question_type",
            "options",
            "correct_answer",
            "explanation",
            "chapter",
            "exam_point",
            "year",
            "answer_visible",
            "explanation_visible",
        )
    }
    prompt = _POLISH_PROMPT.format(ocr_json=json.dumps(ocr_payload, ensure_ascii=False))
    resp = client.chat.completions.create(
        model=settings.qwen_text_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    content = resp.choices[0].message.content or ""
    data = _parse_json_object(content)

    answer_visible = bool(ocr.get("answer_visible"))
    explanation_visible = bool(ocr.get("explanation_visible"))

    correct_answer = data.get("correct_answer") or []
    if isinstance(correct_answer, str):
        correct_answer = [correct_answer]
    if not answer_visible:
        correct_answer = []

    explanation = str(data.get("explanation") or "")
    if not explanation_visible:
        explanation = ""

    options = data.get("options") or ocr.get("options") or []
    qtype = data.get("question_type") or ocr.get("question_type") or "单选"
    if qtype not in ("单选", "多选", "判断"):
        qtype = "单选"

    return {
        "stem": str(data.get("stem") or ocr.get("stem") or "").strip(),
        "question_type": qtype,
        "options": options,
        "correct_answer": [str(a).strip() for a in correct_answer if str(a).strip()],
        "explanation": explanation.strip(),
        "chapter": str(data.get("chapter") or ocr.get("chapter") or "").strip(),
        "exam_point": str(data.get("exam_point") or ocr.get("exam_point") or "").strip(),
        "year": data.get("year") if data.get("year") is not None else ocr.get("year"),
        "answer_visible": answer_visible,
        "explanation_visible": explanation_visible,
        "confidence": float(ocr.get("confidence", 0.5)),
        "has_image": bool(ocr.get("has_image")),
        "text_polished": True,
    }


def enrich_missing_fields(
    question_text: str,
    question_type: str,
    *,
    subject: str,
    knowledge_point_name: str,
    essence: str,
    correct_answer: list[str],
    explanation: str,
    answer_visible: bool,
    explanation_visible: bool,
    client=None,
) -> dict:
    """图上缺答案/解析时，用文本模型补全。"""
    need_answer = not answer_visible or not correct_answer
    need_explanation = not explanation_visible or not (explanation or "").strip()
    if not need_answer and not need_explanation:
        return {
            "correct_answer": correct_answer,
            "explanation": explanation or "",
            "answer_inferred": False,
            "explanation_generated": False,
        }

    client = client or _build_client()
    prompt = _ENRICH_PROMPT.format(
        subject=subject,
        knowledge_point_name=knowledge_point_name,
        essence=essence or "（要义暂无）",
        question=question_text,
        question_type=question_type,
        has_answer="是" if answer_visible and correct_answer else "否",
        has_explanation="是" if explanation_visible and explanation else "否",
    )
    resp = client.chat.completions.create(
        model=settings.qwen_text_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    content = resp.choices[0].message.content or ""
    data = _parse_json_object(content)
    out_answer = correct_answer
    out_expl = explanation or ""
    answer_inferred = False
    explanation_generated = False
    if need_answer:
        raw_ans = data.get("correct_answer") or []
        if isinstance(raw_ans, str):
            raw_ans = [raw_ans]
        if raw_ans:
            out_answer = [str(a).strip() for a in raw_ans if str(a).strip()]
            answer_inferred = bool(data.get("answer_inferred", True))
    if need_explanation:
        out_expl = str(data.get("explanation") or "").strip()
        explanation_generated = bool(data.get("explanation_generated", True))
    return {
        "correct_answer": out_answer,
        "explanation": out_expl,
        "answer_inferred": answer_inferred,
        "explanation_generated": explanation_generated,
    }


def generate_similar_question(
    essence: str,
    original_text: str,
    question_type: str,
    knowledge_point_name: str,
    *,
    avoid_stems: list[str] | None = None,
    client=None,
) -> dict:
    """生成一道相似题原始 dict（未校验）。avoid_stems 是本知识点已出过的题干，用于防重复。"""
    client = client or _build_client()
    avoid = "\n".join(f"- {s}" for s in (avoid_stems or [])[:8]) or "（无）"
    prompt = _SIMILAR_PROMPT.format(
        essence=essence or "（要义暂无）",
        original=original_text,
        question_type=question_type,
        knowledge_point_name=knowledge_point_name,
        avoid=avoid,
    )
    resp = client.chat.completions.create(
        model=settings.qwen_text_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.85,
    )
    content = resp.choices[0].message.content or ""
    data = _parse_json_object(content)
    data["question_type"] = question_type
    return data
