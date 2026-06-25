"""API 请求 / 响应模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class OptionOut(BaseModel):
    key: str
    text: str


class QuestionPublic(BaseModel):
    """给前端刷题用的题目结构 —— 不含正确答案与解析。"""

    id: int
    stem: str
    question_type: str
    options: list[OptionOut] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)
    exam_point: str | None = None
    year: str | None = None
    knowledge_point_name: str | None = None
    source: str | None = None


class SimilarQuestionOut(QuestionPublic):
    """相似题接口：附带是否来自缓存、绑定的原错题 id。"""

    cached: bool = False
    origin_question_id: int


class AttemptRequest(BaseModel):
    question_id: int
    user_answer: list[str] = Field(default_factory=list)


class AttemptResult(BaseModel):
    is_correct: bool
    correct_answer: list[str]
    explanation: str | None = None


class MistakeItem(BaseModel):
    """错题本列表项：错题追踪字段 + 题目展示信息。"""

    question_id: int
    stem: str
    question_type: str
    exam_point: str | None = None
    year: str | None = None
    knowledge_point_name: str | None = None
    images: list[str] = Field(default_factory=list)
    wrong_answer: list[str] = Field(default_factory=list)
    correct_answer: list[str] = Field(default_factory=list)
    wrong_count: int
    correct_count: int
    first_wrong_at: str
    last_attempt_at: str
    mastery: str
    favorite: bool = False


class QuestionReviewItem(BaseModel):
    """待人工确认队列中的题目。"""

    id: int
    stem: str
    question_type: str
    exam_point: str | None = None
    knowledge_point_id: int | None = None
    knowledge_point_name: str | None = None
    needs_review: bool = True


class SetKnowledgePointRequest(BaseModel):
    knowledge_point_id: int
    clear_review: bool = False


class WeaknessItem(BaseModel):
    knowledge_point_id: int
    name: str
    chapter: str
    mastery_requirement: str | None = None
    attempt_count: int
    correct_count: int
    wrong_count: int
    mistake_count: int
    accuracy: float | None = None
    last_attempt_at: str | None = None
    priority: float
    tags: list[str] = Field(default_factory=list)


class SettingsOut(BaseModel):
    daily_target_count: int = 30


class SettingsUpdate(BaseModel):
    daily_target_count: int = Field(ge=5, le=100)


class DailyTaskSummary(BaseModel):
    task_date: str
    target_count: int
    total: int
    completed: int


class DailyTaskQuestion(QuestionPublic):
    completed: bool = False


class UploadDraftOut(BaseModel):
    id: int
    image_path: str
    confidence: float | None = None
    stem: str = ""
    question_type: str = "单选"
    options: list[OptionOut] = Field(default_factory=list)
    correct_answer: list[str] = Field(default_factory=list)
    explanation: str | None = None
    chapter: str | None = None
    exam_point: str | None = None
    subject: str | None = None
    knowledge_point_id: int | None = None
    classify_confidence: float | None = None
    needs_review: bool = False
    classify_note: str | None = None
    answer_inferred: bool = False
    explanation_generated: bool = False
    text_polished: bool = False


class ConfirmUploadRequest(BaseModel):
    stem: str
    question_type: str
    options: list[OptionOut] = Field(default_factory=list)
    correct_answer: list[str] = Field(default_factory=list)
    explanation: str | None = None
    chapter: str | None = None
    exam_point: str | None = None
    knowledge_point_id: int | None = None
