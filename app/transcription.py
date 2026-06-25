import json
import logging
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

from app.alibailian_asr import transcribe_local_file
from app.config import settings
from app.naming import generate_title, sanitize_filename
from app.summary import generate_summary

logger = logging.getLogger(__name__)

ALLOWED_SUFFIXES = {".mp3", ".m4a", ".flac", ".wav"}

_CST = timezone(timedelta(hours=8))


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
    task_id: UUID, title: str, original_filename: str, transcript: str, timestamp: str
) -> str:
    return (
        f"# {title}\n\n"
        f"- **task_id**: `{task_id}`\n"
        f"- **原始文件**: `{original_filename}`\n"
        f"- **time (UTC)**: {timestamp}\n"
        f"- **model**: `{_output_model_label()}`\n\n"
        f"---\n\n"
        f"{transcript}\n"
    )


def _summary_markdown(
    task_id: UUID,
    title: str,
    stype: str,
    summary: str,
    timestamp: str,
) -> str:
    return (
        f"# {title} — 总结\n\n"
        f"- **类型**: {stype}\n"
        f"- **task_id**: `{task_id}`\n"
        f"- **time (UTC)**: {timestamp}\n"
        f"- **model**: `gemini/{settings.summary_model}`\n\n"
        f"---\n\n"
        f"{summary}\n"
    )


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _unique_base(audio_dir: Path, tx_dir: Path, sum_dir: Path, base: str, suffix: str) -> str:
    cand = base
    i = 2
    while (
        (tx_dir / f"{cand}.md").exists()
        or (sum_dir / f"{cand}.md").exists()
        or (audio_dir / f"{cand}{suffix}").exists()
    ):
        cand = f"{base}_{i}"
        i += 1
    return cand


def _write_marker(task_dir: Path, task_id: UUID, data: dict) -> None:
    _atomic_write(task_dir / f"{task_id}.json", json.dumps(data, ensure_ascii=False, indent=2))


def run_pipeline(task_id: UUID, saved_path: Path, original_filename: str) -> None:
    out = settings.shared_output_dir
    audio_dir = out / "audio"
    tx_dir = out / "transcripts"
    sum_dir = out / "summaries"
    task_dir = out / ".tasks"
    for d in (audio_dir, tx_dir, sum_dir, task_dir):
        d.mkdir(parents=True, exist_ok=True)

    suffix = saved_path.suffix.lower() or Path(original_filename).suffix.lower()
    date = datetime.now(_CST).strftime("%Y-%m-%d")

    try:
        transcript = transcribe_file(saved_path)

        title = sanitize_filename(generate_title(transcript))
        base = _unique_base(audio_dir, tx_dir, sum_dir, f"{date}_{title}", suffix)

        ts = datetime.now(timezone.utc).isoformat()
        _atomic_write(
            tx_dir / f"{base}.md",
            _transcript_markdown(task_id, title, original_filename, transcript, ts),
        )

        summary_path = ""
        summary_type = ""
        if result := generate_summary(transcript, title=title):
            summary_type, summary_body = result
            summary_path = f"summaries/{base}.md"
            _atomic_write(
                sum_dir / f"{base}.md",
                _summary_markdown(task_id, title, summary_type, summary_body, ts),
            )

        audio_name = f"{base}{suffix}"
        shutil.copy2(saved_path, audio_dir / audio_name)

        marker = {
            "task_id": str(task_id),
            "ready": True,
            "title": title,
            "transcript": f"transcripts/{base}.md",
            "audio": f"audio/{audio_name}",
            "audio_file": audio_name,
        }
        if summary_path:
            marker["summary"] = summary_path
            marker["summary_type"] = summary_type
        _write_marker(task_dir, task_id, marker)
        logger.info("完成 task=%s -> %s", task_id, base)
    except Exception as e:
        logger.exception("Transcription failed task_id=%s", task_id)
        try:
            fb_name = f"{date}_未转写_{str(task_id)[:8]}{suffix}"
            if saved_path.is_file():
                shutil.copy2(saved_path, audio_dir / fb_name)
        except OSError:
            fb_name = ""
        try:
            _write_marker(
                task_dir,
                task_id,
                {
                    "task_id": str(task_id),
                    "ready": False,
                    "error": str(e),
                    "audio_file": fb_name,
                    "audio": f"audio/{fb_name}" if fb_name else "",
                },
            )
        except OSError:
            pass
        raise
    finally:
        try:
            saved_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove temp file %s", saved_path)