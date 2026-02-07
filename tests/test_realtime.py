"""Tests for RealtimeClient (WebSocket)."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from vector_db_client import (
    RealtimeClient,
    Point,
    WsAckMessage,
    WsEventMessage,
    WsErrorMessage,
)
from vector_db_client.exceptions import VectorDBError


@pytest.fixture
def rt_client():
    """Create a RealtimeClient (not connected)."""
    return RealtimeClient(base_url="http://localhost:8080", api_key="sk-test")


def test_build_ws_url(rt_client):
    """Test that the WS URL is built correctly with token."""
    url = rt_client._build_ws_url()
    assert url.startswith("ws://")
    assert "/api/v1/ws" in url
    assert "token=sk-test" in url


def test_build_ws_url_https():
    """Test that https base URL uses wss scheme."""
    rt = RealtimeClient(base_url="https://my-server.com", api_key="sk-key")
    url = rt._build_ws_url()
    assert url.startswith("wss://")
    assert "token=sk-key" in url


def test_build_ws_url_no_api_key():
    """Test WS URL without API key."""
    rt = RealtimeClient(base_url="http://localhost:8080")
    url = rt._build_ws_url()
    assert "token=" not in url


def test_ensure_connected_raises_when_not_connected(rt_client):
    """Test that operations raise when not connected."""
    with pytest.raises(RuntimeError, match="not connected"):
        rt_client._ensure_connected()


@pytest.mark.asyncio
async def test_upsert_sends_correct_message(rt_client):
    """Test that upsert sends the right JSON and waits for ack."""
    # Mock WebSocket
    mock_ws = AsyncMock()
    sent_messages = []

    async def capture_send(msg):
        sent_messages.append(json.loads(msg))

    mock_ws.send = capture_send
    mock_ws.close = AsyncMock()

    rt_client._ws = mock_ws
    rt_client._connected = True

    # Simulate the ack being received immediately
    ack_data = WsAckMessage(upserted=2, failed=0, took_ms=3)

    async def fake_send_and_wait(msg):
        sent_messages.append(msg)
        return ack_data

    rt_client._send_and_wait_ack = fake_send_and_wait

    points = [
        Point(id="p1", vector=[0.1, 0.2], metadata={"text": "hello"}),
        Point(id="p2", vector=[0.3, 0.4], metadata={}),
    ]

    result = await rt_client.upsert("my-col", points)

    assert result.upserted == 2
    assert result.failed == 0
    assert result.took_ms == 3

    # Verify the message structure
    msg = sent_messages[0]
    assert msg["type"] == "upsert"
    assert msg["collection"] == "my-col"
    assert len(msg["points"]) == 2


@pytest.mark.asyncio
async def test_subscribe_sends_correct_message(rt_client):
    """Test that subscribe sends the right JSON."""
    ack_data = WsAckMessage(upserted=0, failed=0, took_ms=0)
    sent = []

    async def fake_send_and_wait(msg):
        sent.append(msg)
        return ack_data

    rt_client._ws = AsyncMock()
    rt_client._connected = True
    rt_client._send_and_wait_ack = fake_send_and_wait

    result = await rt_client.subscribe("docs", events=["upsert", "delete"])

    assert result.upserted == 0
    msg = sent[0]
    assert msg["type"] == "subscribe"
    assert msg["collection"] == "docs"
    assert msg["events"] == ["upsert", "delete"]


@pytest.mark.asyncio
async def test_subscribe_without_events(rt_client):
    """Test subscribe without event filter."""
    ack_data = WsAckMessage(upserted=0, failed=0, took_ms=0)
    sent = []

    async def fake_send_and_wait(msg):
        sent.append(msg)
        return ack_data

    rt_client._ws = AsyncMock()
    rt_client._connected = True
    rt_client._send_and_wait_ack = fake_send_and_wait

    await rt_client.subscribe("docs")

    msg = sent[0]
    assert "events" not in msg


@pytest.mark.asyncio
async def test_listen_loop_dispatches_event():
    """Test that the listen loop dispatches event messages to callbacks."""
    rt = RealtimeClient(base_url="http://localhost:8080")
    rt._connected = True

    events_received = []
    rt.on_event(lambda e: events_received.append(e))

    # Simulate WS messages
    event_json = json.dumps({
        "type": "event",
        "collection": "docs",
        "action": "upsert",
        "point_ids": ["p1"],
        "timestamp": 1700000000,
    })

    class FakeWS:
        def __init__(self):
            self.messages = [event_json]
            self.idx = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.idx < len(self.messages):
                msg = self.messages[self.idx]
                self.idx += 1
                return msg
            raise StopAsyncIteration

        async def close(self):
            pass

    rt._ws = FakeWS()
    await rt._listen_loop()

    assert len(events_received) == 1
    assert events_received[0].collection == "docs"
    assert events_received[0].action == "upsert"
    assert events_received[0].point_ids == ["p1"]


@pytest.mark.asyncio
async def test_listen_loop_dispatches_error():
    """Test that the listen loop dispatches error messages to callbacks."""
    rt = RealtimeClient(base_url="http://localhost:8080")
    rt._connected = True

    errors_received = []
    rt.on_error(lambda e: errors_received.append(e))

    error_json = json.dumps({
        "type": "error",
        "message": "collection not found",
        "code": 404,
    })

    class FakeWS:
        def __aiter__(self):
            return self

        async def __anext__(self):
            if not hasattr(self, '_sent'):
                self._sent = True
                return error_json
            raise StopAsyncIteration

        async def close(self):
            pass

    rt._ws = FakeWS()
    await rt._listen_loop()

    assert len(errors_received) == 1
    assert errors_received[0].code == 404


@pytest.mark.asyncio
async def test_listen_loop_handles_pong():
    """Test that pong resolves pending pong future."""
    rt = RealtimeClient(base_url="http://localhost:8080")
    rt._connected = True

    loop = asyncio.get_event_loop()
    rt._pending_pong = loop.create_future()

    pong_json = json.dumps({"type": "pong"})

    class FakeWS:
        def __aiter__(self):
            return self

        async def __anext__(self):
            if not hasattr(self, '_sent'):
                self._sent = True
                return pong_json
            raise StopAsyncIteration

        async def close(self):
            pass

    rt._ws = FakeWS()
    await rt._listen_loop()

    assert rt._pending_pong.done()


@pytest.mark.asyncio
async def test_listen_loop_responds_to_server_ping():
    """Test that server-initiated ping gets a pong response."""
    rt = RealtimeClient(base_url="http://localhost:8080")
    rt._connected = True

    sent_messages = []
    ping_json = json.dumps({"type": "ping"})

    class FakeWS:
        def __aiter__(self):
            return self

        async def __anext__(self):
            if not hasattr(self, '_sent'):
                self._sent = True
                return ping_json
            raise StopAsyncIteration

        async def send(self, msg):
            sent_messages.append(json.loads(msg))

        async def close(self):
            pass

    rt._ws = FakeWS()
    await rt._listen_loop()

    assert len(sent_messages) == 1
    assert sent_messages[0]["type"] == "pong"


@pytest.mark.asyncio
async def test_context_manager():
    """Test RealtimeClient as async context manager."""
    rt = RealtimeClient(base_url="http://localhost:8080", api_key="sk-test")

    mock_ws = AsyncMock()
    mock_ws.close = AsyncMock()

    class FakeWS:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def close(self):
            pass

    with patch("websockets.connect", new_callable=AsyncMock, return_value=FakeWS()):
        async with rt as client:
            assert client._connected is True

        assert client._connected is False
