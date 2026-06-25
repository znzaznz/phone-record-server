import json
import logging
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote
from uuid import UUID

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from app.config import settings
from app.docx_export import md_file_to_docx_bytes, validate_stem
from app.transcription import ALLOWED_SUFFIXES, run_pipeline, validate_audio_filename

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.temp_upload_dir.mkdir(parents=True, exist_ok=True)
    out = settings.shared_output_dir
    for sub in ("audio", "transcripts", "summaries", ".tasks"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Audio STT Producer", version="0.3.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> dict[str, str]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    try:
        validate_audio_filename(file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    task_id = uuid.uuid4()
    suffix = Path(file.filename).suffix.lower()
    dest = settings.temp_upload_dir / f"{task_id}{suffix}"

    try:
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out)
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Failed to store upload") from e

    background_tasks.add_task(run_pipeline, task_id, dest, file.filename)
    return {
        "task_id": str(task_id),
        "status": "accepted",
        "message": (
            "Transcription queued; 完成后逐字稿在 transcripts/，总结在 summaries/。"
            "可轮询 /tasks/{task_id} 查进度。"
        ),
    }


class TaskStatusResponse(BaseModel):
    task_id: str
    ready: bool
    audio_saved: bool
    audio_file: str = ""
    title: str = ""
    transcript: str = ""
    summary: str = ""


@app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
def task_status(task_id: str) -> TaskStatusResponse:
    try:
        tid = UUID(task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid task_id") from e

    marker = settings.shared_output_dir / ".tasks" / f"{tid}.json"
    if marker.is_file():
        try:
            d = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            d = {}
        return TaskStatusResponse(
            task_id=task_id,
            ready=bool(d.get("ready")),
            audio_saved=bool(d.get("audio_file")),
            audio_file=str(d.get("audio_file") or ""),
            title=str(d.get("title") or ""),
            transcript=str(d.get("transcript") or ""),
            summary=str(d.get("summary") or ""),
        )
    return TaskStatusResponse(task_id=task_id, ready=False, audio_saved=False)


def _docx_response(md_path: Path, download_name: str) -> Response:
    if not md_path.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    try:
        data = md_file_to_docx_bytes(md_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail="docx conversion failed") from e
    # filename 可能含中文，用 RFC 5987 避免 latin-1 报错
    disp = f"attachment; filename*=UTF-8''{quote(download_name)}"
    return Response(
        content=data,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": disp},
    )


@app.get("/download/transcript/{stem}.docx")
def download_transcript_docx(stem: str) -> Response:
    try:
        stem = validate_stem(stem)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    md = settings.shared_output_dir / "transcripts" / f"{stem}.md"
    return _docx_response(md, f"{stem}_原稿.docx")


@app.get("/download/summary/{stem}.docx")
def download_summary_docx(stem: str) -> Response:
    try:
        stem = validate_stem(stem)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    md = settings.shared_output_dir / "summaries" / f"{stem}.md"
    return _docx_response(md, f"{stem}_总结.docx")