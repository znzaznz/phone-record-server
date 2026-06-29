import json

from fastapi.testclient import TestClient
import pytest
from starlette.websockets import WebSocketDisconnect

from app.config import settings
from app.remote_relay import remote_relay
from main import app


AUTH = {
    "devices": {
        "phone-1": {"role": "phone", "token": "phone-1-token"},
        "phone-2": {"role": "phone", "token": "phone-2-token"},
        "home-pc": {"role": "computer", "token": "home-pc-token"},
        "office-pc": {"role": "computer", "token": "office-pc-token"},
    },
    "pairings": {"phone-1": ["home-pc"], "phone-2": ["home-pc"]},
}


@pytest.fixture(autouse=True)
def isolated_relay(tmp_path):
    originals = (
        settings.remote_agent_db_path,
        settings.remote_agent_auth_config_path,
        settings.remote_agent_auth_config_json,
        settings.remote_agent_trust_forwarded_proto,
    )
    config_path = tmp_path / "remote-agent-auth.json"
    config_path.write_text(json.dumps(AUTH), encoding="utf-8")
    settings.remote_agent_db_path = tmp_path / "relay.sqlite3"
    settings.remote_agent_auth_config_path = str(config_path)
    settings.remote_agent_auth_config_json = ""
    settings.remote_agent_trust_forwarded_proto = True
    try:
        yield config_path
    finally:
        (
            settings.remote_agent_db_path,
            settings.remote_agent_auth_config_path,
            settings.remote_agent_auth_config_json,
            settings.remote_agent_trust_forwarded_proto,
        ) = originals


def secure_client() -> TestClient:
    return TestClient(app, headers={"x-forwarded-proto": "https"})


def ws(device_id: str, role: str, token: str) -> str:
    return f"/remote/ws?token={token}&role={role}&device_id={device_id}"


def test_rejects_wrong_token_and_role_impersonation() -> None:
    with secure_client() as client:
        for path in (
            ws("phone-1", "phone", "wrong"),
            ws("phone-1", "computer", "phone-1-token"),
            ws("home-pc", "computer", "phone-1-token"),
        ):
            with pytest.raises(WebSocketDisconnect) as caught:
                with client.websocket_connect(path) as socket:
                    socket.receive_json()
            assert caught.value.code == 1008


def test_rejects_plain_ws_except_explicit_loopback_mode() -> None:
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect) as caught:
            with client.websocket_connect(
                ws("phone-1", "phone", "phone-1-token")
            ) as socket:
                socket.receive_json()
        assert caught.value.code == 1008


def test_allows_explicit_plain_ws_loopback_mode() -> None:
    original_local = settings.remote_agent_allow_insecure_localhost
    original_forwarded = settings.remote_agent_trust_forwarded_proto
    settings.remote_agent_allow_insecure_localhost = True
    settings.remote_agent_trust_forwarded_proto = False
    try:
        with TestClient(app, headers={"host": "127.0.0.1:8000"}) as client:
            with client.websocket_connect(
                ws("phone-1", "phone", "phone-1-token")
            ) as socket:
                assert socket.receive_json()["device_id"] == "phone-1"
    finally:
        settings.remote_agent_allow_insecure_localhost = original_local
        settings.remote_agent_trust_forwarded_proto = original_forwarded


def test_routes_independent_device_tokens_and_writes_audit() -> None:
    with secure_client() as client:
        with client.websocket_connect(
            ws("phone-1", "phone", "phone-1-token")
        ) as phone, client.websocket_connect(
            ws("home-pc", "computer", "home-pc-token")
        ) as computer:
            assert phone.receive_json()["device_id"] == "phone-1"
            assert computer.receive_json()["device_id"] == "home-pc"
            phone.send_json(
                {
                    "type": "start_session",
                    "target": "home-pc",
                    "workspace": "code",
                    "prompt": "hello",
                }
            )
            command = computer.receive_json()
            assert command["source"] == "phone-1"
            computer.send_json(
                {
                    "type": "status",
                    "target": "phone-1",
                    "session_id": "s1",
                    "seq": 1,
                    "status": "running",
                }
            )
            assert phone.receive_json()["source"] == "home-pc"

    entries = remote_relay.store.audit_entries()
    command_entry = next(row for row in entries if row["event_type"] == "start_session")
    assert command_entry["device_id"] == "phone-1"
    assert command_entry["computer_id"] == "home-pc"
    assert command_entry["outcome"] == "allowed"
    assert "detail=hello" in command_entry["summary"]
    assert command_entry["created_at"].endswith("+00:00")


def test_pairing_whitelist_blocks_phone_and_computer_directions() -> None:
    with secure_client() as client:
        with client.websocket_connect(
            ws("phone-1", "phone", "phone-1-token")
        ) as phone, client.websocket_connect(
            ws("office-pc", "computer", "office-pc-token")
        ) as computer:
            phone.receive_json()
            computer.receive_json()
            phone.send_json({"type": "list_sessions", "target": "office-pc"})
            assert phone.receive_json() == {
                "type": "relay_error",
                "error": "not_paired",
                "target": "office-pc",
            }
            computer.send_json(
                {"type": "status", "target": "phone-1", "session_id": "x", "seq": 1}
            )
            assert computer.receive_json()["error"] == "not_paired"

    denied = [
        row for row in remote_relay.store.audit_entries()
        if row["outcome"] == "denied_unpaired"
    ]
    assert len(denied) == 2


def test_file_revocation_disconnects_only_revoked_device(isolated_relay) -> None:
    with secure_client() as client:
        with client.websocket_connect(
            ws("phone-1", "phone", "phone-1-token")
        ) as phone1, client.websocket_connect(
            ws("phone-2", "phone", "phone-2-token")
        ) as phone2, client.websocket_connect(
            ws("home-pc", "computer", "home-pc-token")
        ) as computer:
            phone1.receive_json()
            phone2.receive_json()
            computer.receive_json()
            revoked = json.loads(json.dumps(AUTH))
            revoked["devices"]["phone-1"]["revoked"] = True
            isolated_relay.write_text(json.dumps(revoked), encoding="utf-8")

            with pytest.raises(WebSocketDisconnect) as caught:
                phone1.receive_json()
            assert caught.value.code == 1008

            computer.send_json(
                {
                    "type": "status",
                    "target": "phone-2",
                    "session_id": "still-online",
                    "seq": 1,
                    "status": "done",
                }
            )
            assert phone2.receive_json()["status"] == "done"

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                ws("phone-1", "phone", "phone-1-token")
            ) as rejected:
                rejected.receive_json()


def test_persists_lists_replays_and_resumes_after_computer_disconnect() -> None:
    with secure_client() as client:
        with client.websocket_connect(
            ws("phone-1", "phone", "phone-1-token")
        ) as phone:
            phone.receive_json()
            with client.websocket_connect(
                ws("home-pc", "computer", "home-pc-token")
            ) as computer:
                computer.receive_json()
                for seq, content in ((1, "one"), (2, "two"), (3, "three")):
                    computer.send_json(
                        {
                            "type": "assistant_text",
                            "target": "phone-1",
                            "session_id": "session-1",
                            "seq": seq,
                            "content": content,
                            "title": "repair disconnect",
                            "workspace": "C:/code/project-a",
                            "engine_session_id": "claude-1",
                        }
                    )
                    assert phone.receive_json()["seq"] == seq

            phone.send_json({"type": "list_sessions", "target": "home-pc"})
            sessions = phone.receive_json()
            assert sessions["items"][0]["last_seq"] == 3
            phone.send_json(
                {
                    "type": "attach",
                    "target": "home-pc",
                    "session_id": "session-1",
                    "last_seq": 1,
                }
            )
            assert [phone.receive_json()["seq"], phone.receive_json()["seq"]] == [2, 3]
            assert phone.receive_json()["type"] == "attach_complete"

            with client.websocket_connect(
                ws("home-pc", "computer", "home-pc-token")
            ) as computer:
                computer.receive_json()
                phone.send_json(
                    {
                        "type": "resume_session",
                        "target": "home-pc",
                        "session_id": "session-1",
                    }
                )
                resumed = computer.receive_json()
                assert resumed["engine_session_id"] == "claude-1"
                assert resumed["workspace"] == "C:/code/project-a"


def test_phone_disconnect_does_not_stop_event_persistence() -> None:
    with secure_client() as client:
        with client.websocket_connect(
            ws("home-pc", "computer", "home-pc-token")
        ) as computer:
            computer.receive_json()
            with client.websocket_connect(
                ws("phone-1", "phone", "phone-1-token")
            ) as phone:
                phone.receive_json()
            computer.send_json(
                {
                    "type": "status",
                    "status": "done",
                    "target": "phone-1",
                    "session_id": "offline",
                    "seq": 1,
                }
            )
            with client.websocket_connect(
                ws("phone-1", "phone", "phone-1-token")
            ) as reconnected:
                reconnected.receive_json()
                reconnected.send_json(
                    {
                        "type": "attach",
                        "target": "home-pc",
                        "session_id": "offline",
                        "last_seq": 0,
                    }
                )
                assert reconnected.receive_json()["replay"] is True
