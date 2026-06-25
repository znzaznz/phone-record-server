"""SQLite 连接与初始化。

S0 只需保证：数据库文件能自动创建、连接正常。
后续切片（S1 起）在这里扩充题库 / 错题本 / 作答记录的 schema。
"""

import json
import sqlite3
from pathlib import Path

from .config import RESOURCE_DIR, settings

# 题库数据以 data/imports/*.jsonl 为可追踪源（SQLite 不进 git）。
# 全新 clone / 删库后，启动时按表逐表重建（仅当该表为空时载入）。
# 用 RESOURCE_DIR：打包后从包内只读资源读，开发时即仓库根。
SNAPSHOT_DIR = RESOURCE_DIR / "data" / "imports"
SNAPSHOT_TABLES = ["exam_points", "knowledge_points", "questions", "lecture_materials"]


def _table_empty(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone() is None


def seed_from_snapshot(
    conn: sqlite3.Connection, snapshot_dir: Path | None = None
) -> dict[str, int]:
    """从 data/imports/*.jsonl 重建题库数据，仅载入当前为空的表（幂等、不覆盖现有数据）。

    按 SNAPSHOT_TABLES 顺序载入以满足外键依赖（考点→知识点→题目）。
    返回 {表名: 载入行数}。
    """
    snapshot_dir = snapshot_dir or SNAPSHOT_DIR
    loaded: dict[str, int] = {}
    for table in SNAPSHOT_TABLES:
        path = snapshot_dir / f"{table}.jsonl"
        if not path.exists() or not _table_empty(conn, table):
            continue
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows:
            continue
        cols = list(rows[0].keys())
        conn.executemany(
            f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' for _ in cols)})",
            [[r.get(c) for c in cols] for r in rows],
        )
        loaded[table] = len(rows)
    if loaded:
        conn.commit()
    return loaded


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or settings.db_path
    # check_same_thread=False：FastAPI 同步端点在线程池执行，本地单用户场景安全
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _migrate_add_column(conn: sqlite3.Connection, table: str, column: str, decl: str) -> None:
    """幂等地给已存在的表补列（CREATE TABLE IF NOT EXISTS 不会改旧表）。"""
    cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def init_db(db_path: Path | None = None) -> None:
    """创建数据库文件并建立基础 schema（幂等）。"""
    path = db_path or settings.db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection(path)
    try:
        # 记录 schema 版本，后续 migration 用得上。S0 先建这张表证明 DB 可写。
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT OR IGNORE INTO schema_meta (key, value) VALUES ('version', '0')"
        )
        # 题库（S1）。选项 / 正确答案 / 配图路径以 JSON 文本存。
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS questions (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                chapter        TEXT,
                exam_point     TEXT,
                question_type  TEXT NOT NULL,            -- 单选 / 多选 / 判断
                difficulty     TEXT,
                year           TEXT,
                stem           TEXT NOT NULL,            -- 题干
                options        TEXT NOT NULL DEFAULT '[]', -- JSON: [{"key":"A","text":"..."}]
                correct_answer TEXT NOT NULL DEFAULT '[]', -- JSON: ["A"] / ["A","C"] / ["对"]
                explanation    TEXT,                     -- 解析
                images         TEXT NOT NULL DEFAULT '[]', -- JSON: 相对路径列表
                source         TEXT NOT NULL,            -- PDF导入 / 截图上传 / 相似题生成
                source_ref     TEXT,                     -- 来源定位：如 "<pdf名>#page=4"
                confidence     REAL,                     -- VLM 置信度 0~1
                needs_review   INTEGER NOT NULL DEFAULT 0, -- 1=进人工确认队列
                knowledge_point_id INTEGER REFERENCES knowledge_points(id), -- S6 归类填入
                created_at     TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        _migrate_add_column(conn, "questions", "knowledge_point_id", "INTEGER")
        # 作答记录（S2）。每次作答一行。
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS attempts (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id  INTEGER NOT NULL REFERENCES questions(id),
                user_answer  TEXT NOT NULL DEFAULT '[]',  -- JSON: 用户选择
                is_correct   INTEGER NOT NULL,            -- 1=对 0=错
                created_at   TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        # 知识点体系（S4）。两层：考点(节) -> 知识点。来源=官方考纲种子。
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS exam_points (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                chapter TEXT NOT NULL,           -- 章名，如 总论
                name    TEXT NOT NULL,           -- 考点(节)名，如 法律行为与代理
                seq     INTEGER NOT NULL,        -- 在大纲中的顺序
                UNIQUE (chapter, name)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_points (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                exam_point_id       INTEGER NOT NULL REFERENCES exam_points(id),
                name                TEXT NOT NULL,
                mastery_requirement TEXT,         -- 掌握/熟悉/了解（官方能力要求）
                essence             TEXT,         -- 要义总结（讲义概括，可空待补）
                seq                 INTEGER NOT NULL,
                UNIQUE (exam_point_id, name)
            )
            """
        )
        # 科目维度（中级会计三科：经济法/中级会计实务/财务管理）。
        # 旧库 exam_points 无此列，补列后回填为经济法（已有数据均属经济法）。
        _migrate_add_column(conn, "exam_points", "subject", "TEXT")
        conn.execute("UPDATE exam_points SET subject = '经济法' WHERE subject IS NULL")

        # 错题本（S3）。一题一行，作答出错时自动收录、重复出错累加。
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mistakes (
                question_id     INTEGER PRIMARY KEY REFERENCES questions(id),
                wrong_answer    TEXT NOT NULL DEFAULT '[]',  -- JSON: 最近一次错误答案
                correct_answer  TEXT NOT NULL DEFAULT '[]',  -- JSON: 正确答案快照
                wrong_count     INTEGER NOT NULL DEFAULT 0,
                correct_count   INTEGER NOT NULL DEFAULT 0,
                first_wrong_at  TEXT NOT NULL,               -- 第一次做错时间
                last_attempt_at TEXT NOT NULL,               -- 最近一次做题时间
                mastery         TEXT NOT NULL DEFAULT '未掌握',
                favorite        INTEGER NOT NULL DEFAULT 0   -- S11 收藏
            )
            """
        )
        _migrate_add_column(conn, "mistakes", "favorite", "INTEGER NOT NULL DEFAULT 0")

        # SM-2 复习状态（S9）
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS question_sm2 (
                question_id   INTEGER PRIMARY KEY REFERENCES questions(id),
                ease          REAL NOT NULL DEFAULT 2.5,
                interval_days INTEGER NOT NULL DEFAULT 0,
                repetition    INTEGER NOT NULL DEFAULT 0,
                due_date      TEXT
            )
            """
        )
        # 每日任务（S9）
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_tasks (
                task_date    TEXT PRIMARY KEY,
                target_count INTEGER NOT NULL DEFAULT 30,
                created_at   TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_task_items (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                task_date   TEXT NOT NULL,
                question_id INTEGER NOT NULL REFERENCES questions(id),
                seq         INTEGER NOT NULL,
                completed   INTEGER NOT NULL DEFAULT 0,
                UNIQUE (task_date, question_id)
            )
            """
        )
        # 截图上传草稿（S10）
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS upload_drafts (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                draft_json TEXT NOT NULL,
                confidence REAL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS lecture_materials (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                source         TEXT NOT NULL,
                source_pdf     TEXT NOT NULL,
                page           INTEGER NOT NULL,
                text           TEXT NOT NULL,
                readable_summary TEXT,
                images         TEXT NOT NULL DEFAULT '[]',
                page_image     TEXT,
                status         TEXT NOT NULL DEFAULT 'pass',
                review_reason  TEXT,
                created_at     TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE (source, source_pdf, page)
            )
            """
        )
        conn.execute(
            "INSERT OR IGNORE INTO schema_meta (key, value) VALUES ('daily_target_count', '30')"
        )
        conn.commit()
    finally:
        conn.close()


def check_connection(db_path: Path | None = None) -> bool:
    """健康检查用：能否连上并执行一条查询。"""
    try:
        conn = get_connection(db_path)
        try:
            conn.execute("SELECT 1").fetchone()
            return True
        finally:
            conn.close()
    except sqlite3.Error:
        return False
