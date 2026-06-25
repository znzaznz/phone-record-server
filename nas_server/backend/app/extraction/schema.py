"""题目草稿的 schema 与校验（VLM 返回 → 结构化题）。"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

QUESTION_TYPES = {"单选", "多选", "判断"}


class Option(BaseModel):
    key: str  # A / B / C / D / 对 / 错
    text: str


class QuestionDraft(BaseModel):
    """VLM 从一页识别出的单道题。校验失败者进人工确认队列。"""

    stem: str = Field(min_length=1)
    question_type: str
    options: list[Option] = Field(default_factory=list)
    correct_answer: list[str] = Field(default_factory=list)
    explanation: str | None = None
    chapter: str | None = None
    exam_point: str | None = None
    year: str | None = None
    difficulty: str | None = None
    has_image: bool = False
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("year", "difficulty", mode="before")
    @classmethod
    def _coerce_str(cls, v):
        # VLM 常把年份返回成整数（2024），统一转成字符串
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return str(int(v))
        v = str(v).strip()
        return v or None

    @field_validator("question_type")
    @classmethod
    def _valid_type(cls, v: str) -> str:
        if v not in QUESTION_TYPES:
            raise ValueError(f"未知题型: {v!r}，应为 {QUESTION_TYPES}")
        return v

    @field_validator("correct_answer", mode="before")
    @classmethod
    def _coerce_answer(cls, v):
        # 容忍模型返回字符串 "A" 或 "AC" 或列表
        if v is None:
            return []
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return []
            # "对"/"错" 整体保留；纯字母答案逐字母拆（"AC" -> ["A","C"]）
            if v in ("对", "错", "正确", "错误"):
                return [v]
            if all(c.isalpha() and c.isascii() for c in v):
                return [c.upper() for c in v]
            return [v]
        return v


def validate_draft(raw: dict) -> tuple[QuestionDraft | None, str | None]:
    """校验单条原始 dict。

    返回 (draft, None) 或 (None, 错误信息)。
    """
    try:
        return QuestionDraft.model_validate(raw), None
    except Exception as e:  # pydantic.ValidationError 及其它
        return None, str(e)
