"""应用配置：从项目根目录的 .env 读取。

key 只在后端持有，绝不进浏览器。这里只暴露「配置项是否已填」，
不向外打印 key 值本身。
"""

import os
import sys
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_FROZEN = getattr(sys, "frozen", False)  # PyInstaller 打包后为 True


def _resource_dir() -> Path:
    """只读资源根：打包后是 PyInstaller 解包目录（含 .env、data/imports），开发时是仓库根。"""
    if _FROZEN:
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parents[2]


def _data_home() -> Path:
    """可写数据根：打包后用 %APPDATA%/MistakeGenie（db、上传图片），开发时是仓库根。"""
    if _FROZEN:
        base = os.getenv("APPDATA") or str(Path.home())
        d = Path(base) / "MistakeGenie"
        d.mkdir(parents=True, exist_ok=True)
        return d
    return Path(__file__).resolve().parents[2]


# 资源根（只读）与数据根（可写）：开发时二者都是仓库根，行为不变
RESOURCE_DIR = _resource_dir()
PROJECT_ROOT = _data_home()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # 先读打包内的 .env，再让数据目录里的 .env 覆盖（用户可在 APPDATA 改 key）
        env_file=(RESOURCE_DIR / ".env", PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ===== 通义千问 / DashScope =====
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_vl_model: str = "qwen-vl-plus"
    qwen_text_model: str = "qwen-plus"

    # ===== 本地 Ollama =====
    vlm_provider: str = "dashscope"  # dashscope / ollama
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_vl_model: str = "qwen3-vl:8b"

    # ===== 数据库 =====
    db_path: Path = PROJECT_ROOT / "mistakegenie.db"

    # ===== 本地文件存储 =====
    # 题目配图等本地文件根目录；DB 里只存相对此目录的路径
    media_dir: Path = PROJECT_ROOT / "media"

    def config_presence(self) -> dict[str, bool]:
        """报告各关键配置项是否已填（不暴露 key 值本身）。"""
        return {
            "DASHSCOPE_API_KEY": bool(self.dashscope_api_key.strip()),
            "DASHSCOPE_BASE_URL": bool(self.dashscope_base_url.strip()),
            "QWEN_VL_MODEL": bool(self.qwen_vl_model.strip()),
            "QWEN_TEXT_MODEL": bool(self.qwen_text_model.strip()),
            "VLM_PROVIDER": bool(self.vlm_provider.strip()),
            "OLLAMA_BASE_URL": bool(self.ollama_base_url.strip()),
            "OLLAMA_VL_MODEL": bool(self.ollama_vl_model.strip()),
        }


settings = Settings()
