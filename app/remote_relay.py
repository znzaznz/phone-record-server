import asyncio
import hmac
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect, status


@dataclass(frozen=True)
class DeviceCredential:
    role: str
    token: str
    revoked: bool = False


class DeviceAuthPolicy:
    """Reloadable per-device credentials and phone/computer pairing policy."""

    def __init__(self, *, config_path: str = "", config_json: str = "") -> None:
        if config_path and config_json:
            raise ValueError(
                "set only one of REMOTE_AGENT_AUTH_CONFIG_PATH or "
                "REMOTE_AGENT_AUTH_CONFIG_JSON"
            )
        self.config_path = Path(config_path).expanduser() if config_path else None
        self.config_json = config_json
        # Validate configured input at startup. No config is allowed but denies all.
        self._load()

    def _load(self) -> tuple[dict[str, DeviceCredential], dict[str, set[str]]]:
        if self.config_path is not None:
            raw = self.config_path.read_text(encoding="utf-8")
        elif self.config_json:
            raw = self.config_json
        else:
            return {}, {}
        document = json.loads(raw)
        if not isinstance(document, dict):
            raise ValueError("auth config root must be an object")
        raw_devices = document.get("devices")
        raw_pairings = document.get("pairings")
        if not isinstance(raw_devices, dict) or not isinstance(raw_pairings, dict):
            raise ValueError("auth config requires object fields: devices and pairings")

        devices: dict[str, DeviceCredential] = {}
        tokens: set[str] = set()
        for device_id, value in raw_devices.items():
            if not isinstance(device_id, str) or not device_id or not isinstance(value, dict):
                raise ValueError("every device must have a non-empty string id and object value")
            role = value.get("role")
            token = value.get("token")
            if role not in {"phone", "computer"} or not isinstance(token, str) or not token:
                raise ValueError(f"device {device_id!r} requires role and non-empty token")
            if token in tokens:
                raise ValueError("device tokens must be unique")
            tokens.add(token)
            devices[device_id] = DeviceCredential(
                role=role, token=token, revoked=value.get("revoked") is True
            )

        pairings: dict[str, set[str]] = {}
        for phone_id, computer_ids in raw_pairings.items():
            if not isinstance(phone_id, str) or not isinstance(computer_ids, list):
                raise ValueError("pairings must map phone ids to computer id arrays")
            if phone_id not in devices or devices[phone_id].role != "phone":
                raise ValueError(f"pairing source {phone_id!r} is not a configured phone")
            targets = {str(item) for item in computer_ids}
            if any(item not in devices or devices[item].role != "computer" for item in targets):
                raise ValueError(f"pairing {phone_id!r} contains an unknown computer")
            pairings[phone_id] = targets
        return devices, pairings

    def authorize(self, device_id: str, role: str, token: str) -> bool:
        try:
            devices, _ = self._load()
        except (OSError, ValueError):
            return False
        credential = devices.get(device_id)
        return bool(
            credential
            and not credential.revoked
            and credential.role == role
            and hmac.compare_digest(token, credential.token)
        )

    def paired(self, phone_id: str, computer_id: str) -> bool:
        try:
            _, pairings = self._load()
        except (OSError, ValueError):
            return False
        return computer_id in pairings.get(phone_id, set())


class RelayEventStore:
    """Small SQLite event log used by the relay across process restarts."""

    def __init__(self, path: Path, *, max_events_per_session: int = 5000) -> None:
        self.path = path
        self.max_events_per_session = max_events_per_session
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS sessions (
                    computer_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    workspace TEXT NOT NULL DEFAULT '',
                    engine_session_id TEXT NOT NULL DEFAULT '',
                    last_active TEXT NOT NULL,
                    last_seq INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (computer_id, session_id)
                );
                CREATE TABLE IF NOT EXISTS events (
                    computer_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (computer_id, session_id, seq)
                );
                CREATE INDEX IF NOT EXISTS events_session_seq
                    ON events(computer_id, session_id, seq);
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    device_id TEXT NOT NULL,
                    computer_id TEXT NOT NULL DEFAULT '',
                    event_type TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS audit_created_at
                    ON audit_log(created_at);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def append(self, computer_id: str, event: dict[str, Any]) -> bool:
        session_id = str(event["session_id"])
        seq = int(event["seq"])
        now = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        with self._connect() as db:
            inserted = db.execute(
                """INSERT OR IGNORE INTO events
                   (computer_id, session_id, seq, payload, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (computer_id, session_id, seq, payload, now),
            ).rowcount == 1
            if not inserted:
                return False
            db.execute(
                """INSERT INTO sessions
                   (computer_id, session_id, title, workspace,
                    engine_session_id, last_active, last_seq)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(computer_id, session_id) DO UPDATE SET
                     title = CASE WHEN excluded.title != ''
                                  THEN excluded.title ELSE sessions.title END,
                     workspace = CASE WHEN excluded.workspace != ''
                                      THEN excluded.workspace ELSE sessions.workspace END,
                     engine_session_id = CASE WHEN excluded.engine_session_id != ''
                                              THEN excluded.engine_session_id
                                              ELSE sessions.engine_session_id END,
                     last_active = excluded.last_active,
                     last_seq = MAX(sessions.last_seq, excluded.last_seq)""",
                (
                    computer_id,
                    session_id,
                    str(event.get("title") or ""),
                    str(event.get("workspace") or ""),
                    str(event.get("engine_session_id") or ""),
                    now,
                    seq,
                ),
            )
            if self.max_events_per_session > 0:
                db.execute(
                    """DELETE FROM events
                       WHERE computer_id = ? AND session_id = ? AND seq <= ?""",
                    (computer_id, session_id, seq - self.max_events_per_session),
                )
        return True

    def events_after(
        self, computer_id: str, session_id: str, last_seq: int
    ) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                """SELECT payload FROM events
                   WHERE computer_id = ? AND session_id = ? AND seq > ?
                   ORDER BY seq""",
                (computer_id, session_id, last_seq),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def list_sessions(self, computer_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(
                """SELECT session_id, title, workspace, engine_session_id,
                          last_active, last_seq
                   FROM sessions WHERE computer_id = ?
                   ORDER BY last_active DESC""",
                (computer_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_session(self, computer_id: str, session_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            db.row_factory = sqlite3.Row
            row = db.execute(
                """SELECT session_id, title, workspace, engine_session_id,
                          last_active, last_seq
                   FROM sessions WHERE computer_id = ? AND session_id = ?""",
                (computer_id, session_id),
            ).fetchone()
        return dict(row) if row else None

    def audit(
        self,
        *,
        device_id: str,
        computer_id: str = "",
        event_type: str,
        outcome: str,
        summary: str = "",
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as db:
            db.execute(
                """INSERT INTO audit_log
                   (created_at, device_id, computer_id, event_type, outcome, summary)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (now, device_id, computer_id, event_type, outcome, summary),
            )

    def audit_entries(self) -> list[dict[str, Any]]:
        with self._connect() as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(
                """SELECT created_at, device_id, computer_id, event_type,
                          outcome, summary FROM audit_log ORDER BY id"""
            ).fetchall()
        return [dict(row) for row in rows]


class RemoteRelay:
    """Authenticated relay plus durable per-session replay log."""

    def __init__(self) -> None:
        self._computers: dict[str, WebSocket] = {}
        self._phones: dict[str, WebSocket] = {}
        self._lock = asyncio.Lock()
        self._store: RelayEventStore | None = None
        self._auth = DeviceAuthPolicy()

    def initialize(
        self,
        database_path: Path,
        *,
        auth_config_path: str = "",
        auth_config_json: str = "",
    ) -> None:
        if self._store is None or self._store.path != database_path:
            self._store = RelayEventStore(database_path)
        self._auth = DeviceAuthPolicy(
            config_path=auth_config_path, config_json=auth_config_json
        )

    @property
    def store(self) -> RelayEventStore:
        if self._store is None:
            raise RuntimeError("remote relay store is not initialized")
        return self._store

    async def connect(
        self,
        websocket: WebSocket,
        *,
        token: str,
        role: str,
        device_id: str,
    ) -> None:
        if role not in {"phone", "computer"} or not device_id:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        if not self._auth.authorize(device_id, role, token):
            self.store.audit(
                device_id=device_id,
                event_type="connect",
                outcome="denied",
                summary=f"role={role}",
            )
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        await websocket.accept()
        self.store.audit(
            device_id=device_id,
            event_type="connect",
            outcome="allowed",
            summary=f"role={role}",
        )
        peers = self._phones if role == "phone" else self._computers
        async with self._lock:
            old = peers.get(device_id)
            peers[device_id] = websocket
        if old is not None and old is not websocket:
            await old.close(code=status.WS_1000_NORMAL_CLOSURE)

        try:
            await websocket.send_json(
                {"type": "registered", "role": role, "device_id": device_id}
            )
            receive_task = asyncio.create_task(websocket.receive_json())
            while True:
                done, _ = await asyncio.wait({receive_task}, timeout=0.5)
                if not self._auth.authorize(device_id, role, token):
                    self.store.audit(
                        device_id=device_id,
                        event_type="revocation",
                        outcome="disconnected",
                        summary=f"role={role}",
                    )
                    receive_task.cancel()
                    await asyncio.gather(receive_task, return_exceptions=True)
                    await websocket.close(
                        code=status.WS_1008_POLICY_VIOLATION, reason="device_revoked"
                    )
                    return
                if not done:
                    continue
                message = receive_task.result()
                if role == "phone":
                    await self._from_phone(websocket, device_id, message)
                else:
                    await self._from_computer(websocket, device_id, message)
                receive_task = asyncio.create_task(websocket.receive_json())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            async with self._lock:
                if peers.get(device_id) is websocket:
                    peers.pop(device_id, None)

    async def _from_phone(
        self, websocket: WebSocket, device_id: str, message: dict[str, Any]
    ) -> None:
        target = str(message.get("target") or "")
        kind = message.get("type")
        if not self._auth.paired(device_id, target):
            self.store.audit(
                device_id=device_id,
                computer_id=target,
                event_type=str(kind or "unknown"),
                outcome="denied_unpaired",
                summary=self._message_summary(message),
            )
            await websocket.send_json(
                {"type": "relay_error", "error": "not_paired", "target": target}
            )
            return
        self.store.audit(
            device_id=device_id,
            computer_id=target,
            event_type=str(kind or "unknown"),
            outcome="allowed",
            summary=self._message_summary(message),
        )
        if kind == "list_sessions":
            async with self._lock:
                await websocket.send_json(
                    {
                        "type": "sessions",
                        "source": target,
                        "items": self.store.list_sessions(target),
                    }
                )
            return
        if kind == "attach":
            session_id = str(message.get("session_id") or "")
            try:
                last_seq = max(0, int(message.get("last_seq") or 0))
            except (TypeError, ValueError):
                last_seq = 0
            async with self._lock:
                events = self.store.events_after(target, session_id, last_seq)
                for event in events:
                    await websocket.send_json({**event, "source": target, "replay": True})
                await websocket.send_json(
                    {
                        "type": "attach_complete",
                        "source": target,
                        "session_id": session_id,
                        "last_seq": events[-1]["seq"] if events else last_seq,
                    }
                )
            return

        forwarded = {**message, "source": device_id}
        if kind == "resume_session" and message.get("session_id"):
            saved = self.store.get_session(target, str(message["session_id"]))
            if saved:
                for field in ("engine_session_id", "workspace", "title", "last_seq"):
                    if not forwarded.get(field):
                        forwarded[field] = saved[field]
        async with self._lock:
            destination = self._computers.get(target)
            if destination is None:
                await websocket.send_json(
                    {"type": "relay_error", "error": "target_offline", "target": target}
                )
                return
            try:
                await destination.send_json(forwarded)
            except (WebSocketDisconnect, RuntimeError):
                if self._computers.get(target) is destination:
                    self._computers.pop(target, None)
                await websocket.send_json(
                    {"type": "relay_error", "error": "target_offline", "target": target}
                )

    async def _from_computer(
        self, websocket: WebSocket, device_id: str, message: dict[str, Any]
    ) -> None:
        target = str(message.get("target") or "")
        kind = str(message.get("type") or "unknown")
        if not self._auth.paired(target, device_id):
            self.store.audit(
                device_id=target,
                computer_id=device_id,
                event_type=kind,
                outcome="denied_unpaired",
                summary=self._message_summary(message),
            )
            await websocket.send_json(
                {"type": "relay_error", "error": "not_paired", "target": target}
            )
            return
        self.store.audit(
            device_id=target,
            computer_id=device_id,
            event_type=kind,
            outcome="allowed",
            summary=self._message_summary(message),
        )
        forwarded = {**message, "source": device_id}
        async with self._lock:
            if message.get("session_id") and message.get("seq") is not None:
                inserted = self.store.append(device_id, message)
                if not inserted:
                    return
            destination = self._phones.get(target)
            if destination is not None:
                try:
                    await destination.send_json(forwarded)
                except (WebSocketDisconnect, RuntimeError):
                    # A phone disappearing must never tear down the computer socket.
                    if self._phones.get(target) is destination:
                        self._phones.pop(target, None)

    @staticmethod
    def _message_summary(message: dict[str, Any]) -> str:
        fields = []
        for key in ("session_id", "workspace", "status", "engine", "name"):
            value = message.get(key)
            if value not in (None, ""):
                fields.append(f"{key}={value}")
        detail = message.get("prompt") or message.get("message") or message.get("content")
        if detail not in (None, ""):
            text = str(detail).replace("\r", " ").replace("\n", " ")[:160]
            fields.append(f"detail={text}")
        return "; ".join(fields)[:500]


remote_relay = RemoteRelay()
