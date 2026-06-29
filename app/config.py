import tempfile
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # 阿里云百炼 DashScope Fun-ASR
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com"
    dashscope_model: str = "fun-asr"
    dashscope_diarization_enabled: bool = True
    dashscope_speaker_count: int | None = None
    dashscope_poll_interval_seconds: float = 2.0
    dashscope_poll_timeout_seconds: float = 7200.0
    dashscope_retry_max_attempts: int = 3
    dashscope_retry_base_delay_seconds: float = 5.0

    # 取名 — 百炼 qwen（国内直连，不走代理）
    title_model: str = "qwen-plus"
    title_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    title_max_chars: int = 8000

    # 总结报告 — Gemini 3.5 Flash（仅此走代理）
    google_api_key: str = ""
    summary_enabled: bool = True
    summary_model: str = "gemini-3.5-flash"
    summary_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    summary_proxy_url: str = "http://192.168.31.9:7890"
    summary_max_chars: int = 24000

    public_base_url: str = "http://192.168.31.9:10083"

    shared_output_dir: Path = Path("shared/output")
    temp_upload_dir: Path = Path(tempfile.gettempdir()) / "audio-stt-uploads"

    # Issue 03: per-device relay credentials and phone -> computer pairings.
    # A file is preferred because it can be edited to revoke a device without
    # restarting the relay. JSON is convenient for container environment setup.
    remote_agent_auth_config_path: str = ""
    remote_agent_auth_config_json: str = ""
    remote_agent_allow_insecure_localhost: bool = False
    remote_agent_trust_forwarded_proto: bool = False
    remote_agent_db_path: Path = Path("shared/output/remote-agent.sqlite3")


settings = Settings()
