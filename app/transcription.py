import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from app.alibailian_asr import transcribe_local_file
from app.config import settings

logger = logging.getLogger(__name__)

ALLOWED_SUFFIXES = {".mp3", ".m4a", ".flac", ".wav"}


def validate_audio_filename(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_SUFFIXES))
        raise ValueError(f"Unsupported format '{suffix}'. Allowed: {allowed}")
    return suffix


def transcribe_file(path: Path) -> str:
    logger.info("Bailian DashScope ASR (model=%s)", settings.dashscope_model)
    return transcribe_local_file(path, settings)


def _output_model_label() -> str:
    return f"bailian/{settings.dashscope_model}"


def _transcript_markdown(
    task_id: UUID,
    original_filename: str,
    transcript: str,
    timestamp: str,
) -> str:
    model = _output_model_label()
    return (
        f"# Transcript\n\n"
        f"- **task_id**: `{task_id}`\n"
        f"- **file**: `{original_filename}`\n"
        f"- **time (UTC)**: {timestamp}\n"
        f"- **model**: `{model}`\n\n"
        f"---\n\n"
        f"{transcript}\n"
    )


def write_output_atomic(
    task_id: UUID,
    original_filename: str,
    transcript: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / str(task_id)
    md_path = base.with_suffix(".md")
    tmp_md = base.with_suffix(".md.tmp")

    ts = datetime.now(timezone.utc).isoformat()
    md_body = _transcript_markdown(task_id, original_filename, transcript, ts)
    tmp_md.write_text(md_body, encoding="utf-8")
    os.replace(tmp_md, md_path)


def run_pipeline(task_id: UUID, saved_path: Path, original_filename: str) -> None:
    output_dir = settings.shared_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = saved_path.suffix.lower() or Path(original_filename).suffix.lower()
    audio_out = output_dir / f"{task_id}{suffix}"
    try:
        transcript = transcribe_file(saved_path)
        write_output_atomic(task_id, original_filename, transcript, output_dir)
    except Exception:
        logger.exception("Transcription failed task_id=%s", task_id)
        raise
    finally:
        try:
            if saved_path.is_file():
                shutil.copy2(saved_path, audio_out)
        except OSError as e:
            logger.warning("Could not copy source audio to %s: %s", audio_out, e)
        try:
            saved_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove temp file %s", saved_path)
