import json
import logging
import mimetypes
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

ALLOWED_SUFFIXES = {".mp3", ".m4a", ".flac", ".wav"}


def validate_audio_filename(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_SUFFIXES))
        raise ValueError(f"Unsupported format '{suffix}'. Allowed: {allowed}")
    return suffix


def _run_ffmpeg(args: list[str]) -> None:
    try:
        subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "ffmpeg/ffprobe not found on PATH; install FFmpeg to use audio chunking."
        ) from e
    except subprocess.CalledProcessError as e:
        err = (e.stderr or e.stdout or "").strip()
        raise RuntimeError(f"ffmpeg failed: {err or e.returncode}") from e


def _ffprobe_duration_seconds(path: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(out.stdout.strip())


def _ffmpeg_extract_wav(src: Path, dest: Path, start_sec: float, duration_sec: float) -> None:
    _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_sec:.6f}",
            "-i",
            str(src),
            "-t",
            f"{duration_sec:.6f}",
            "-vn",
            "-acodec",
            "pcm_s16le",
            str(dest),
        ]
    )


def _chunk_ranges_ms(total_ms: int, chunk_ms: int, min_tail_ms: int) -> list[tuple[int, int]]:
    if total_ms <= chunk_ms:
        return [(0, total_ms)]
    ranges: list[tuple[int, int]] = []
    pos = 0
    while pos < total_ms:
        end = min(pos + chunk_ms, total_ms)
        tail = total_ms - end
        if tail > 0 and tail < min_tail_ms:
            end = total_ms
        elif end - pos < min_tail_ms and ranges:
            start, _ = ranges.pop()
            ranges.append((start, total_ms))
            break
        ranges.append((pos, end))
        pos = end
    return ranges


def _chunk_ranges_overlap_ms(
    total_ms: int, chunk_ms: int, overlap_ms: int, min_tail_ms: int
) -> list[tuple[int, int]]:
    """Step by chunk_ms; segment i is [i*chunk_ms, min((i+1)*chunk_ms + overlap_ms, total)]."""
    if total_ms <= chunk_ms:
        return [(0, total_ms)]
    ranges: list[tuple[int, int]] = []
    pos = 0
    while pos < total_ms:
        end = min(pos + chunk_ms + overlap_ms, total_ms)
        tail = total_ms - end
        if tail > 0 and tail < min_tail_ms:
            end = total_ms
        elif end - pos < min_tail_ms and ranges:
            start, _ = ranges.pop()
            ranges.append((start, total_ms))
            break
        ranges.append((pos, end))
        if end >= total_ms:
            break
        pos += chunk_ms
    return ranges


def _norm_chars(s: str) -> str:
    return "".join(s.split())


def _prefix_len_for_norm_count(s: str, norm_count: int) -> int:
    seen = 0
    for i, ch in enumerate(s):
        if not ch.isspace():
            seen += 1
            if seen == norm_count:
                return i + 1
    return len(s)


def _longest_equal_suffix_prefix(left: str, right: str, max_chars: int) -> int:
    m = min(len(left), len(right), max_chars)
    for k in range(m, 0, -1):
        if left[-k:] == right[:k]:
            return k
    return 0


def _merge_transcripts_with_overlap(parts: list[str], overlap_seconds: int) -> str:
    if not parts:
        return ""
    max_chars = max(80, overlap_seconds * 50)
    max_norm = max(40, overlap_seconds * 35)
    out = parts[0]
    for nxt in parts[1:]:
        k = _longest_equal_suffix_prefix(out, nxt, max_chars)
        if k > 0:
            out = out + nxt[k:]
            continue
        nl = _norm_chars(out)
        nr = _norm_chars(nxt)
        kn = _longest_equal_suffix_prefix(nl, nr, max_norm)
        if kn >= 2:
            cut = _prefix_len_for_norm_count(nxt, kn)
            out = out + nxt[cut:]
        else:
            out = out + "\n" + nxt
    return out


def _transcribe_one_file(path: Path, filename_for_api: str) -> str:
    base = settings.stt_base_url.rstrip("/")
    url = f"{base}/audio/transcriptions"
    mime, _ = mimetypes.guess_type(filename_for_api)
    content_type = mime or "application/octet-stream"
    max_attempts = 4
    body: object | None = None
    for attempt in range(max_attempts):
        try:
            with path.open("rb") as audio_file:
                files = {"file": (filename_for_api, audio_file, content_type)}
                data = {"model": settings.stt_model}
                with httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
                    r = client.post(
                        url,
                        headers={"Authorization": f"Bearer {settings.stt_api_key}"},
                        files=files,
                        data=data,
                    )
                    r.raise_for_status()
                    body = r.json()
            break
        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            if (
                attempt + 1 < max_attempts
                and code in (429, 500, 502, 503, 504)
            ):
                wait = 2**attempt
                logger.warning(
                    "STT HTTP %s (attempt %s/%s), retry in %ss: %s",
                    code,
                    attempt + 1,
                    max_attempts,
                    wait,
                    url,
                )
                time.sleep(wait)
            else:
                raise
        except httpx.RequestError as e:
            if attempt + 1 < max_attempts:
                wait = 2**attempt
                logger.warning(
                    "STT request error (attempt %s/%s), retry in %ss: %s",
                    attempt + 1,
                    max_attempts,
                    wait,
                    e,
                )
                time.sleep(wait)
            else:
                raise
    else:
        raise RuntimeError("STT request failed after retries")

    assert body is not None
    try:
        return str(body["text"]).strip()
    except (KeyError, TypeError) as e:
        raise RuntimeError(f"Unexpected transcription response: {body!r}") from e


def transcribe_file(path: Path) -> str:
    if not settings.stt_api_key:
        raise RuntimeError("STT_API_KEY is not set")

    if settings.stt_chunk_seconds <= 0:
        return _transcribe_one_file(path, path.name)

    try:
        duration_sec = _ffprobe_duration_seconds(path)
    except FileNotFoundError as e:
        raise RuntimeError(
            "ffprobe not found on PATH; install FFmpeg or set STT_CHUNK_SECONDS=0."
        ) from e
    except (subprocess.CalledProcessError, ValueError) as e:
        raise RuntimeError(f"Could not read audio duration: {path}") from e

    duration_ms = int(round(duration_sec * 1000))
    chunk_ms = settings.stt_chunk_seconds * 1000
    min_tail_ms = max(0, settings.stt_chunk_merge_tail_seconds) * 1000

    if duration_ms <= chunk_ms:
        logger.info("Audio shorter than chunk size; single request (~%s ms)", duration_ms)
        return _transcribe_one_file(path, path.name)

    overlap_ms = max(0, settings.stt_chunk_overlap_seconds) * 1000
    if overlap_ms > 0:
        ranges = _chunk_ranges_overlap_ms(duration_ms, chunk_ms, overlap_ms, min_tail_ms)
        logger.info(
            "Transcribing %s overlapping chunks (~%s ms overlap)",
            len(ranges),
            overlap_ms,
        )
    else:
        ranges = _chunk_ranges_ms(duration_ms, chunk_ms, min_tail_ms)
        logger.info("Transcribing %s chunks (total ~%s ms)", len(ranges), duration_ms)
    parts: list[str] = []
    for i, (start_ms, end_ms) in enumerate(ranges):
        start_sec = start_ms / 1000.0
        dur_sec = (end_ms - start_ms) / 1000.0
        suffix = f"_part{i + 1:04d}.wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            _ffmpeg_extract_wav(path, tmp_path, start_sec, dur_sec)
            text = _transcribe_one_file(tmp_path, tmp_path.name)
            if text:
                parts.append(text)
        finally:
            tmp_path.unlink(missing_ok=True)

    if overlap_ms > 0 and len(parts) > 1:
        return _merge_transcripts_with_overlap(parts, settings.stt_chunk_overlap_seconds)
    return "\n".join(parts)


def write_output_atomic(
    task_id: UUID,
    original_filename: str,
    transcript: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / str(task_id)
    json_path = base.with_suffix(".json")
    tmp_json = base.with_suffix(".json.tmp")
    done_path = base.with_suffix(".done")

    payload = {
        "task_id": str(task_id),
        "original_filename": original_filename,
        "transcript": transcript,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": settings.stt_model,
    }
    data = json.dumps(payload, ensure_ascii=False, indent=2)
    tmp_json.write_text(data, encoding="utf-8")
    os.replace(tmp_json, json_path)
    done_path.touch()


def run_pipeline(task_id: UUID, saved_path: Path, original_filename: str) -> None:
    try:
        transcript = transcribe_file(saved_path)
        write_output_atomic(task_id, original_filename, transcript, settings.shared_output_dir)
    except Exception:
        logger.exception("Transcription failed task_id=%s", task_id)
        raise
    finally:
        try:
            saved_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove temp file %s", saved_path)
