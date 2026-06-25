import json
import logging
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.config import settings
from app.transcription import ALLOWED_SUFFIXES, run_pipeline, validate_audio_filename

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.temp_upload_dir.mkdir(parents=True, exist_ok=True)
    (settings.shared_output_dir / "audio").mkdir(parents=True, exist_ok=True)
    (settings.shared_output_dir / "transcripts").mkdir(parents=True, exist_ok=True)
    (settings.shared_output_dir / ".tasks").mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Audio STT Producer", version="0.2.0", lifespan=lifespan)


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
            "Transcription queued; 完成后逐字稿在 transcripts/{日期}_{标题}.md，"
            "音频在 audio/{日期}_{标题} 同名文件。可轮询 /tasks/{task_id} 查进度与标题。"
        ),
    }


class TaskStatusResponse(BaseModel):
    task_id: str
    ready: bool
    audio_saved: bool
    audio_file: str = ""
    title: str = ""


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
        )
    # 没有标记 = 还没开始/处理中
    return TaskStatusResponse(task_id=task_id, ready=False, audio_saved=False)
