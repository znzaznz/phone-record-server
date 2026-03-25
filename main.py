import logging
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile

from app.config import settings
from app.transcription import run_pipeline, validate_audio_filename

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.temp_upload_dir.mkdir(parents=True, exist_ok=True)
    settings.shared_output_dir.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Audio STT Producer", version="0.1.0", lifespan=lifespan)


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
        "message": "Transcription queued; OpenClaw may watch .done under shared output",
    }


@app.get("/tasks/{task_id}")
def task_status(task_id: str) -> dict[str, str | bool]:
    try:
        tid = UUID(task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid task_id") from e
    out = settings.shared_output_dir
    done = out / f"{tid}.done"
    data = out / f"{tid}.json"
    return {
        "task_id": task_id,
        "done": done.is_file(),
        "json_ready": data.is_file(),
    }
