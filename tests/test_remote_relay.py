from fastapi.testclient import TestClient
import pytest
from starlette.websockets import WebSocketDisconnect

from app.config import settings
from main import app


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
