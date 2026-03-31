"""阿里云百炼 Fun-ASR 录音文件识别（异步）+ 可选说话人分离。

文档: https://help.aliyun.com/zh/model-studio/fun-asr-recorded-speech-recognition-restful-api
临时上传: https://www.alibabacloud.com/help/zh/model-studio/get-temporary-file-url
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def _dashscope_api_key(settings: Any) -> str:
    k = (getattr(settings, "dashscope_api_key", "") or getattr(settings, "stt_api_key", "") or "").strip()
    if not k:
        raise RuntimeError("百炼 ASR 需要 DASHSCOPE_API_KEY 或 STT_API_KEY")
    return k


def _get_upload_policy(
    client: httpx.Client,
    api_key: str,
    base: str,
    model: str,
) -> dict[str, Any]:
    r = client.get(
        f"{base}/api/v1/uploads",
        params={"action": "getPolicy", "model": model},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60.0,
    )
    r.raise_for_status()
    body = r.json()
    if "data" not in body:
        raise RuntimeError(f"getPolicy unexpected response: {body!r}")
    return body["data"]


def _upload_to_instant_oss(
    client: httpx.Client,
    policy: dict[str, Any],
    path: Path,
) -> str:
    file_name = f"{uuid.uuid4().hex}_{path.name}"
    upload_dir = policy["upload_dir"]
    object_key = f"{upload_dir}/{file_name}"
    upload_host = policy["upload_host"]
    with path.open("rb") as f:
        files = [
            ("OSSAccessKeyId", (None, policy["oss_access_key_id"])),
            ("Signature", (None, policy["signature"])),
            ("policy", (None, policy["policy"])),
            ("x-oss-object-acl", (None, policy["x_oss_object_acl"])),
            ("x-oss-forbid-overwrite", (None, policy["x_oss_forbid_overwrite"])),
            ("key", (None, object_key)),
            ("success_action_status", (None, "200")),
            ("file", (file_name, f)),
        ]
        r = client.post(upload_host, files=files, timeout=httpx.Timeout(600.0, connect=60.0))
    if r.status_code != 200:
        raise RuntimeError(f"百炼临时 OSS 上传失败 HTTP {r.status_code}: {r.text[:500]}")
    return f"oss://{object_key}"


def _submit_asr_job(
    client: httpx.Client,
    api_key: str,
    base: str,
    model: str,
    oss_url: str,
    diarization: bool,
    speaker_count: int | None,
) -> str:
    parameters: dict[str, Any] = {"channel_id": [0]}
    if diarization:
        parameters["diarization_enabled"] = True
        if speaker_count is not None:
            parameters["speaker_count"] = int(speaker_count)

    payload = {
        "model": model,
        "input": {"file_urls": [oss_url]},
        "parameters": parameters,
    }
    r = client.post(
        f"{base}/api/v1/services/audio/asr/transcription",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
            "X-DashScope-OssResourceResolve": "enable",
        },
        json=payload,
        timeout=120.0,
    )
    r.raise_for_status()
    body = r.json()
    out = body.get("output") or {}
    task_id = out.get("task_id")
    if not task_id:
        raise RuntimeError(f"百炼提交转写无 task_id: {body!r}")
    return str(task_id)


def _poll_until_done(
    client: httpx.Client,
    api_key: str,
    base: str,
    task_id: str,
    poll_interval: float,
    max_wait: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        r = client.get(
            f"{base}/api/v1/tasks/{task_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60.0,
        )
        r.raise_for_status()
        body = r.json()
        output = body.get("output") or {}
        status = output.get("task_status", "")
        if status == "FAILED":
            raise RuntimeError(f"百炼任务失败: {body!r}")
        if status == "SUCCEEDED":
            return output
        time.sleep(poll_interval)
    raise RuntimeError(f"百炼任务超时（>{max_wait}s）task_id={task_id}")


def _collect_sentences(result_json: dict[str, Any]) -> list[dict[str, Any]]:
    sents: list[dict[str, Any]] = []
    for tr in result_json.get("transcripts") or []:
        for s in tr.get("sentences") or []:
            sents.append(s)
    sents.sort(key=lambda x: int(x.get("begin_time", 0)))
    return sents


def _format_with_speakers(sentences: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    cur: int | None = None
    buf: list[str] = []
    for s in sentences:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        sp_raw = s.get("speaker_id", 0)
        sp = int(sp_raw) if sp_raw is not None else 0
        if cur is None:
            cur = sp
        if sp != cur:
            lines.append(f"**说话人 {cur + 1}**\n\n{''.join(buf).strip()}\n")
            cur = sp
            buf = [text]
        else:
            buf.append(text)
    if buf and cur is not None:
        lines.append(f"**说话人 {cur + 1}**\n\n{''.join(buf).strip()}\n")
    return "\n".join(lines).strip()


def _format_plain(sentences: list[dict[str, Any]]) -> str:
    return "".join(s.get("text") or "" for s in sentences).strip()


def transcribe_local_file(path: Path, settings: Any) -> str:
    api_key = _dashscope_api_key(settings)
    base = str(getattr(settings, "dashscope_base_url", "https://dashscope.aliyuncs.com")).rstrip("/")
    model = getattr(settings, "dashscope_model", "fun-asr")
    diarization = bool(getattr(settings, "dashscope_diarization_enabled", True))
    speaker_count = getattr(settings, "dashscope_speaker_count", None)
    poll_iv = float(getattr(settings, "dashscope_poll_interval_seconds", 2.0))
    poll_max = float(getattr(settings, "dashscope_poll_timeout_seconds", 7200.0))

    with httpx.Client() as client:
        logger.info("百炼: 获取上传凭证 model=%s", model)
        policy = _get_upload_policy(client, api_key, base, model)
        logger.info("百炼: 上传音频 %s", path.name)
        oss_url = _upload_to_instant_oss(client, policy, path)
        logger.info("百炼: 提交转写任务")
        task_id = _submit_asr_job(
            client, api_key, base, model, oss_url, diarization, speaker_count
        )
        logger.info("百炼: 轮询 task_id=%s", task_id)
        output = _poll_until_done(client, api_key, base, task_id, poll_iv, poll_max)
        results = output.get("results") or []
        if not results:
            raise RuntimeError(f"百炼无 results: {output!r}")
        first = results[0]
        if first.get("subtask_status") != "SUCCEEDED":
            raise RuntimeError(f"百炼子任务未成功: {first!r}")
        turl = first.get("transcription_url")
        if not turl:
            raise RuntimeError(f"无 transcription_url: {first!r}")
        tr = client.get(str(turl), timeout=120.0)
        tr.raise_for_status()
        data = tr.json()
        sentences = _collect_sentences(data)
        if diarization and sentences and "speaker_id" in sentences[0]:
            return _format_with_speakers(sentences)
        return _format_plain(sentences)
