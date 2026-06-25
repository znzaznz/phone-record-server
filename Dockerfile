FROM python:3.12-slim-bookworm

WORKDIR /app

# nas_server 的依赖是 phone 的超集（fastapi/uvicorn[standard]/multipart/httpx/
# pydantic-settings + openai/pydantic/python-dotenv），一次装完两边都够。
COPY nas_server/requirements-server.txt /tmp/requirements-server.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ -r /tmp/requirements-server.txt

# phone-record: STT producer + hermes codex adapter
COPY app ./app
COPY main.py .
COPY hermes_codex_adapter.py .
COPY start_with_adapter.sh .

# my-nas-server (mistakegenie) 放进自己的子目录；启动时用各自的 CWD 跑，
# 两个同名 app 包互不污染。
COPY nas_server ./nas_server

ENV PYTHONUNBUFFERED=1
ENV FRONTEND_DIR=/app/nas_server/frontend_dist

EXPOSE 8000 18644 8756

CMD ["sh", "/app/start_with_adapter.sh"]
