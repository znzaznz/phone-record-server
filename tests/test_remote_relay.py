from fastapi.testclient import TestClient
import pytest
from starlette.websockets import WebSocketDisconnect

from app.config import settings
from main import app


@pytest.fixture(autouse=True)
def isolated_relay_database(tmp_path):
    original = settings.remote_agent_db_path
    settings.remote_agent_db_path = tmp_path / "relay.sqlite3"
    try:
        yield
    finally:
        settings.remote_agent_db_path = original


def test_rejects_wrong_shared_token() -> None:
    original = settings.remote_agent_token
    settings.remote_agent_token = "test-token"
    try:
        with TestClient(app) as client:
            with pytest.raises(WebSocketDisconnect) as caught:
                with client.websocket_connect(
                    "/remote/ws?token=wrong&role=phone&device_id=phone-1"
                ) as websocket:
                    websocket.receive_json()
            assert caught.value.code == 1008
    finally:
        settings.remote_agent_token = original


def test_routes_phone_commands_and_computer_events() -> None:
    original = settings.remote_agent_token
    settings.remote_agent_token = "test-token"
    try:
        with TestClient(app) as client:
            with client.websocket_connect(
                "/remote/ws?token=test-token&role=phone&device_id=phone-1"
            ) as phone:
                assert phone.receive_json()["type"] == "registered"
                with client.websocket_connect(
                    "/remote/ws?token=test-token&role=computer&device_id=home-pc"
                ) as computer:
                    assert computer.receive_json()["type"] == "registered"
                    phone.send_json(
                        {
                            "type": "start_session",
                            "target": "home-pc",
                            "engine": "claude",
                            "workspace": "code",
                            "prompt": "hello",
                        }
                    )
                    command = computer.receive_json()
                    assert command["source"] == "phone-1"
                    assert command["prompt"] == "hello"

                    computer.send_json(
                        {
                            "type": "status",
                            "target": "phone-1",
                            "session_id": "s1",
                            "seq": 1,
                            "status": "running",
                        }
                    )
                    event = phone.receive_json()
                    assert event["source"] == "home-pc"
                    assert event["seq"] == 1
    finally:
        settings.remote_agent_token = original


def test_persists_lists_and_replays_after_computer_disconnect(tmp_path) -> None:
    original_token = settings.remote_agent_token
    original_db = settings.remote_agent_db_path
    settings.remote_agent_token = "test-token"
    settings.remote_agent_db_path = tmp_path / "relay.sqlite3"
    try:
        with TestClient(app) as client:
            with client.websocket_connect(
                "/remote/ws?token=test-token&role=phone&device_id=phone-1"
            ) as phone:
                phone.receive_json()
                with client.websocket_connect(
                    "/remote/ws?token=test-token&role=computer&device_id=home-pc"
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
                                "title": "修复断线",
                                "workspace": "C:/code/project-a",
                                "engine_session_id": "claude-1",
                            }
                        )
                        assert phone.receive_json()["seq"] == seq

                # The computer is offline; both the index and event log come from NAS.
                phone.send_json({"type": "list_sessions", "target": "home-pc"})
                sessions = phone.receive_json()
                assert sessions["type"] == "sessions"
                assert sessions["items"] == [
                    {
                        "session_id": "session-1",
                        "title": "修复断线",
                        "workspace": "C:/code/project-a",
                        "engine_session_id": "claude-1",
                        "last_active": sessions["items"][0]["last_active"],
                        "last_seq": 3,
                    }
                ]

                phone.send_json(
                    {
                        "type": "attach",
                        "target": "home-pc",
                        "session_id": "session-1",
                        "last_seq": 1,
                    }
                )
                replayed = [phone.receive_json(), phone.receive_json()]
                assert [event["seq"] for event in replayed] == [2, 3]
                assert all(event["replay"] is True for event in replayed)
                assert phone.receive_json() == {
                    "type": "attach_complete",
                    "source": "home-pc",
                    "session_id": "session-1",
                    "last_seq": 3,
                }
        assert settings.remote_agent_db_path.is_file()
    finally:
        settings.remote_agent_token = original_token
        settings.remote_agent_db_path = original_db


def test_resume_by_session_id_is_enriched_from_persisted_index(tmp_path) -> None:
    original_token = settings.remote_agent_token
    original_db = settings.remote_agent_db_path
    settings.remote_agent_token = "test-token"
    settings.remote_agent_db_path = tmp_path / "relay.sqlite3"
    try:
        with TestClient(app) as client:
            with client.websocket_connect(
                "/remote/ws?token=test-token&role=phone&device_id=phone-1"
            ) as phone, client.websocket_connect(
                "/remote/ws?token=test-token&role=computer&device_id=home-pc"
            ) as computer:
                phone.receive_json()
                computer.receive_json()
                computer.send_json(
                    {
                        "type": "status",
                        "status": "done",
                        "target": "phone-1",
                        "session_id": "session-1",
                        "seq": 7,
                        "title": "历史任务",
                        "workspace": "project-a",
                        "engine_session_id": "claude-1",
                    }
                )
                phone.receive_json()
                phone.send_json(
                    {
                        "type": "resume_session",
                        "target": "home-pc",
                        "session_id": "session-1",
                    }
                )
                command = computer.receive_json()
                assert command["engine_session_id"] == "claude-1"
                assert command["workspace"] == "project-a"
                assert command["last_seq"] == 7
                assert command["source"] == "phone-1"
    finally:
        settings.remote_agent_token = original_token
        settings.remote_agent_db_path = original_db


def test_phone_disconnect_does_not_stop_computer_events(tmp_path) -> None:
    original_token = settings.remote_agent_token
    settings.remote_agent_token = "test-token"
    try:
        with TestClient(app) as client:
            with client.websocket_connect(
                "/remote/ws?token=test-token&role=computer&device_id=home-pc"
            ) as computer:
                computer.receive_json()
                with client.websocket_connect(
                    "/remote/ws?token=test-token&role=phone&device_id=phone-1"
                ) as phone:
                    phone.receive_json()

                # The task continues to emit while no phone socket exists.
                computer.send_json(
                    {
                        "type": "status",
                        "status": "done",
                        "target": "phone-1",
                        "session_id": "session-offline",
                        "seq": 1,
                        "title": "离线完成",
                        "workspace": "project-a",
                        "engine_session_id": "claude-offline",
                    }
                )
                with client.websocket_connect(
                    "/remote/ws?token=test-token&role=phone&device_id=phone-1"
                ) as reconnected:
                    reconnected.receive_json()
                    reconnected.send_json(
                        {
                            "type": "attach",
                            "target": "home-pc",
                            "session_id": "session-offline",
                            "last_seq": 0,
                        }
                    )
                    replay = reconnected.receive_json()
                    assert replay["seq"] == 1
                    assert replay["status"] == "done"
                    assert replay["replay"] is True
    finally:
        settings.remote_agent_token = original_token
