import asyncio
import hmac
from fastapi import WebSocket, WebSocketDisconnect, status


class RemoteRelay:
    """In-memory exchange for the issue-01 single-NAS happy path."""

    def __init__(self) -> None:
        self._computers: dict[str, WebSocket] = {}
        self._phones: dict[str, WebSocket] = {}
        self._lock = asyncio.Lock()

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
                target = str(message.get("target") or "")
                destinations = self._computers if role == "phone" else self._phones
                destination = destinations.get(target)
                if destination is None:
                    await websocket.send_json(
                        {"type": "relay_error", "error": "target_offline", "target": target}
                    )
                    continue
                forwarded = dict(message)
                forwarded["source"] = device_id
                await destination.send_json(forwarded)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            async with self._lock:
                if peers.get(device_id) is websocket:
                    peers.pop(device_id, None)


remote_relay = RemoteRelay()
