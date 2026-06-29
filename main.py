import json
import logging
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.responses import Response
from pydantic import BaseModel

from app.config import settings
from app.transcription import ALLOWED_SUFFIXES, run_pipeline, validate_audio_filename
from app.remote_relay import remote_relay

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.temp_upload_dir.mkdir(parents=True, exist_ok=True)
    out = settings.shared_output_dir
    for sub in ("audio", "transcripts", "summaries", ".tasks"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Audio STT Producer", version="0.3.2", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/remote/ws")
async def remote_agent_ws(
    websocket: WebSocket,
    token: str = "",
    role: str = "",
    device_id: str = "",
) -> None:
    await remote_relay.connect(
        websocket,
        expected_token=settings.remote_agent_token,
        token=token,
        role=role,
        device_id=device_id,
    )


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


def _parse_task_id(task_id: str) -> str:
    try:
        return str(UUID(task_id.strip()))
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid task_id") from e


def _load_marker(task_id: str) -> dict:
    marker = settings.shared_output_dir / ".tasks" / f"{task_id}.json"
    if not marker.is_file():
        raise HTTPException(status_code=404, detail="task not found")
    try:
        return json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise HTTPException(status_code=500, detail="invalid task marker") from e


def _md_response(md_path: Path, download_name: str) -> Response:
    if not md_path.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    text = md_path.read_text(encoding="utf-8")
    disp = f'attachment; filename="{download_name}"'
    return Response(
        content=text.encode("utf-8"),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": disp},
    )


@app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
def task_status(task_id: str) -> TaskStatusResponse:
    tid = _parse_task_id(task_id)
    marker = settings.shared_output_dir / ".tasks" / f"{tid}.json"
    if marker.is_file():
        try:
            d = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            d = {}
        return TaskStatusResponse(
            task_id=tid,
            ready=bool(d.get("ready")),
            audio_saved=bool(d.get("audio_file")),
            audio_file=str(d.get("audio_file") or ""),
            title=str(d.get("title") or ""),
            transcript=str(d.get("transcript") or ""),
            summary=str(d.get("summary") or ""),
        )
    return TaskStatusResponse(task_id=tid, ready=False, audio_saved=False)


@app.get("/download/transcript/{task_id}.md")
def download_transcript_md(task_id: str) -> Response:
    tid = _parse_task_id(task_id)
    d = _load_marker(tid)
    rel = str(d.get("transcript") or "").strip()
    if not rel:
        raise HTTPException(status_code=404, detail="transcript not ready")
    md = settings.shared_output_dir / rel
    short = tid.replace("-", "")[:12]
    return _md_response(md, f"transcript_{short}.md")


@app.get("/download/summary/{task_id}.md")
def download_summary_md(task_id: str) -> Response:
    tid = _parse_task_id(task_id)
    d = _load_marker(tid)
    rel = str(d.get("summary") or "").strip()
    if not rel:
        raise HTTPException(status_code=404, detail="summary not available")
    md = settings.shared_output_dir / rel
    short = tid.replace("-", "")[:12]
    return _md_response(md, f"summary_{short}.md")
