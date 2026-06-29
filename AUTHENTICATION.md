# 远程代理鉴权配置

中继使用每设备独立 token，配置文件格式见 `remote-agent-auth.example.json`。
生产环境建议把实际配置放在持久卷，并设置：

```dotenv
REMOTE_AGENT_AUTH_CONFIG_PATH=/run/secrets/remote-agent-auth.json
REMOTE_AGENT_TRUST_FORWARDED_PROTO=true
REMOTE_AGENT_ALLOW_INSECURE_LOCALHOST=false
```

`REMOTE_AGENT_TRUST_FORWARDED_PROTO=true` 仅用于会覆盖 `X-Forwarded-Proto`
的可信 TLS 反向代理；外部连接必须是 WSS。也可以用
`REMOTE_AGENT_AUTH_CONFIG_JSON` 直接传入同结构 JSON，但文件方式支持在线吊销：
将某设备的 `revoked` 改为 `true` 并原子替换配置文件，该连接会在约 0.5 秒内断开，
其它设备不受影响。审计记录写在 `REMOTE_AGENT_DB_PATH` 指定 SQLite 的
`audit_log` 表中，token 不会写入日志。

本机联调可设置：

```dotenv
REMOTE_AGENT_AUTH_CONFIG_PATH=./remote-agent-auth.local.json
REMOTE_AGENT_ALLOW_INSECURE_LOCALHOST=true
REMOTE_AGENT_TRUST_FORWARDED_PROTO=false
```

此例外只允许 `ws://127.0.0.1:8000`、`ws://localhost:8000` 或
`ws://[::1]:8000`。
