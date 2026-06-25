#!/bin/sh
set -eu

# my-nas-server (mistakegenie) :8756 — 用自己的 CWD 跑，确保 `app` 解析到 nas 的包
( cd /app/nas_server/backend && MISTAKEGENIE_HOST=0.0.0.0 python run_server.py ) &

# phone-record hermes codex adapter :18644
python /app/hermes_codex_adapter.py &

# phone-record STT producer :8000（前台进程）
# ponytail: 一个容器跑 3 个进程、无 supervisor。某个崩了其它的会继续静默跑；
# 真要进程级自愈再上 s6/supervisord。
exec uvicorn main:app --host 0.0.0.0 --port 8000
