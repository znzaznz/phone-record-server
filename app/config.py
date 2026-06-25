import tempfile
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # 阿里云百炼 DashScope Fun-ASR
    # Key: https://help.aliyun.com/zh/model-studio/get-api-key
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com"
    dashscope_model: str = "fun-asr"
    dashscope_diarization_enabled: bool = True
    dashscope_speaker_count: int | None = None
    dashscope_poll_interval_seconds: float = 2.0
    dashscope_poll_timeout_seconds: float = 7200.0
    dashscope_retry_max_attempts: int = 3
    dashscope_retry_base_delay_seconds: float = 5.0
    # Local default; set SHARED_OUTPUT_DIR=/shared/output when mounting with OpenClaw
    shared_output_dir: Path = Path("shared/output")
    temp_upload_dir: Path = Path(tempfile.gettempdir()) / "audio-stt-uploads"


settings = Settings()
