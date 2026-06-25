"""打包后后端入口：直接拉起 uvicorn（无 reload）。PyInstaller 以此为主脚本。

端口用冷门的 8756（避开 8000 等常被占用的端口）；可用 MISTAKEGENIE_PORT 覆盖。
前端 .env.production 的 VITE_API_BASE 必须与此端口一致。
"""

import os

import uvicorn

from app.main import app

if __name__ == "__main__":
    # 桌面默认只绑回环；容器里需 0.0.0.0 才能被端口转发到（设 MISTAKEGENIE_HOST=0.0.0.0）
    host = os.getenv("MISTAKEGENIE_HOST", "127.0.0.1")
    port = int(os.getenv("MISTAKEGENIE_PORT", "8756"))
    uvicorn.run(app, host=host, port=port, log_level="info")
