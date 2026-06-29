import asyncio
import hmac
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect, status


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


class RemoteRelay:
    """Authenticated relay plus durable per-session replay log."""

    def __init__(self) -> None:
        self._computers: dict[str, WebSocket] = {}
        self._phones: dict[str, WebSocket] = {}
        self._lock = asyncio.Lock()
        self._store: RelayEventStore | None = None

    def initialize(self, database_path: Path) -> None:
        if self._store is None or self._store.path != database_path:
            self._store = RelayEventStore(database_path)

    @property
    def store(self) -> RelayEventStore:
        if self._store is None:
            raise RuntimeError("remote relay store is not initialized")
        return self._store

    async def connect(
        self,
        websocket: WebSocket,
        *,
        expected_token: str,
        token: str,
        role: str,
        device_id: str,
    ) -> None:
        if not expected_token or not hmac.compare_digest(token, expected_token):
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        if role not in {"phone", "computer"} or not device_id:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        await websocket.accept()
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
            while True:
                message = await websocket.receive_json()
                if role == "phone":
                    await self._from_phone(websocket, device_id, message)
                else:
                    await self._from_computer(device_id, message)
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
        self, device_id: str, message: dict[str, Any]
    ) -> None:
        target = str(message.get("target") or "")
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


remote_relay = RemoteRelay()
