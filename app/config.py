import tempfile
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # SiliconFlow STT: https://docs.siliconflow.cn/en/api-reference/audio/create-audio-transcriptions
    stt_api_key: str = ""
    stt_base_url: str = "https://api.siliconflow.cn/v1"
    stt_model: str = "TeleAI/TeleSpeechASR"
    # 0 = upload whole file once; else slice to this many seconds per STT request (needs FFmpeg on PATH).
    stt_chunk_seconds: int = 300
    # If the tail shorter than this (seconds), merge into the previous chunk.
    stt_chunk_merge_tail_seconds: int = 2
    # Extra seconds appended after each chunk end; same window opens the next chunk → duplicate ASR text is merged away.
    # 0 = hard cuts, no overlap merge.
    stt_chunk_overlap_seconds: int = 5
    # Local default; set SHARED_OUTPUT_DIR=/shared/output when mounting with OpenClaw
    shared_output_dir: Path = Path("shared/output")
    temp_upload_dir: Path = Path(tempfile.gettempdir()) / "audio-stt-uploads"


settings = Settings()
