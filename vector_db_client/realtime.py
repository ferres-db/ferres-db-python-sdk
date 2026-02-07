"""Real-time WebSocket client for FerresDB."""

import asyncio
import json
from typing import List, Optional, Callable, Any
from urllib.parse import urljoin, urlparse, urlunparse

import structlog

from .models import (
    Point,
    WsAckMessage,
    WsEventMessage,
    WsErrorMessage,
)

logger = structlog.get_logger()


class RealtimeClient:
    """WebSocket client for real-time streaming with FerresDB.

    Supports real-time point ingestion, collection event subscriptions,
    and application-level heartbeat.

    Usage::

        async with RealtimeClient("http://localhost:8080", api_key="sk-xxx") as rt:
            # Upsert points in real-time
            ack = await rt.upsert("my_collection", [Point("id-1", [0.1, 0.2], {})])
            print(f"Upserted {ack.upserted} points in {ack.took_ms}ms")

            # Subscribe to events
            rt.on_event(lambda evt: print(f"Event: {evt.action} on {evt.collection}"))
            await rt.subscribe("my_collection", events=["upsert", "delete"])

            # Keep listening for events
            await asyncio.sleep(60)
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the RealtimeClient.

        Args:
            base_url: Base URL of the FerresDB server (e.g., "http://localhost:8080")
            api_key: Optional API key for authentication (passed as ``?token=`` query param)
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._ws = None
        self._listen_task: Optional[asyncio.Task] = None
        self._pending_ack: Optional[asyncio.Future] = None
        self._pending_pong: Optional[asyncio.Future] = None
        self._event_callbacks: List[Callable[[WsEventMessage], Any]] = []
        self._error_callbacks: List[Callable[[WsErrorMessage], Any]] = []
        self._close_callbacks: List[Callable[[], Any]] = []
        self._connected = False
        self._logger = logger.bind(client="RealtimeClient")

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Establish the WebSocket connection.

        Raises:
            ConnectionError: If the connection cannot be established.
        """
        try:
            import websockets
        except ImportError:
            raise ImportError(
                "The 'websockets' package is required for RealtimeClient. "
                "Install it with: pip install websockets>=12.0"
            )

        ws_url = self._build_ws_url()
        self._logger.info("connecting", url=ws_url)

        try:
            self._ws = await websockets.connect(
                ws_url,
                max_size=10 * 1024 * 1024,  # 10 MB, matches server limit
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to WebSocket: {e}") from e

        self._connected = True
        self._listen_task = asyncio.create_task(self._listen_loop())
        self._logger.info("connected")

    async def close(self) -> None:
        """Close the WebSocket connection gracefully."""
        self._connected = False
        if self._listen_task is not None:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            self._listen_task = None
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
        self._logger.info("disconnected")

    async def __aenter__(self) -> "RealtimeClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    # ── Public API ─────────────────────────────────────────────────────────

    async def upsert(
        self,
        collection: str,
        points: List[Point],
    ) -> WsAckMessage:
        """Send an upsert message and wait for the server acknowledgement.

        Args:
            collection: Target collection name.
            points: List of points to upsert.

        Returns:
            WsAckMessage with upserted/failed counts and timing.
        """
        self._ensure_connected()
        msg = {
            "type": "upsert",
            "collection": collection,
            "points": [p.to_dict() for p in points],
        }
        return await self._send_and_wait_ack(msg)

    async def subscribe(
        self,
        collection: str,
        events: Optional[List[str]] = None,
    ) -> WsAckMessage:
        """Subscribe to real-time events for a collection.

        After subscribing, ``on_event`` callbacks will be invoked whenever
        the collection is modified (via REST or WebSocket).

        Args:
            collection: Collection to subscribe to.
            events: Optional list of event types to filter: ``["upsert"]``,
                ``["delete"]``, or both. If omitted, all events are received.

        Returns:
            WsAckMessage confirming the subscription.
        """
        self._ensure_connected()
        msg: dict = {
            "type": "subscribe",
            "collection": collection,
        }
        if events:
            msg["events"] = events
        return await self._send_and_wait_ack(msg)

    async def ping(self) -> None:
        """Send an application-level ping and wait for the pong response."""
        self._ensure_connected()
        loop = asyncio.get_event_loop()
        self._pending_pong = loop.create_future()
        await self._ws.send(json.dumps({"type": "ping"}))
        try:
            await asyncio.wait_for(self._pending_pong, timeout=10.0)
        except asyncio.TimeoutError:
            self._pending_pong = None
            raise TimeoutError("Pong not received within 10 seconds")

    # ── Event callbacks ────────────────────────────────────────────────────

    def on_event(self, callback: Callable[[WsEventMessage], Any]) -> None:
        """Register a callback for collection change events."""
        self._event_callbacks.append(callback)

    def on_error(self, callback: Callable[[WsErrorMessage], Any]) -> None:
        """Register a callback for WebSocket error messages."""
        self._error_callbacks.append(callback)

    def on_close(self, callback: Callable[[], Any]) -> None:
        """Register a callback invoked when the connection closes."""
        self._close_callbacks.append(callback)

    # ── Internal ───────────────────────────────────────────────────────────

    def _build_ws_url(self) -> str:
        """Convert base HTTP URL to WebSocket URL with auth token."""
        parsed = urlparse(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        path = "/api/v1/ws"
        query = f"token={self.api_key}" if self.api_key else ""
        return urlunparse((scheme, parsed.netloc, path, "", query, ""))

    def _ensure_connected(self) -> None:
        if not self._connected or self._ws is None:
            raise RuntimeError(
                "RealtimeClient is not connected. Call connect() or use 'async with'."
            )

    async def _send_and_wait_ack(self, msg: dict) -> WsAckMessage:
        """Send a message and wait for an ack response."""
        loop = asyncio.get_event_loop()
        self._pending_ack = loop.create_future()
        await self._ws.send(json.dumps(msg))
        try:
            result = await asyncio.wait_for(self._pending_ack, timeout=30.0)
        except asyncio.TimeoutError:
            self._pending_ack = None
            raise TimeoutError("Server did not acknowledge within 30 seconds")
        return result

    async def _listen_loop(self) -> None:
        """Background task that reads messages from the WebSocket."""
        try:
            async for raw in self._ws:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    self._logger.warning("invalid_json", raw=raw[:200])
                    continue

                msg_type = data.get("type")

                if msg_type == "ack":
                    ack = WsAckMessage.from_dict(data)
                    if self._pending_ack and not self._pending_ack.done():
                        self._pending_ack.set_result(ack)

                elif msg_type == "pong":
                    if self._pending_pong and not self._pending_pong.done():
                        self._pending_pong.set_result(None)

                elif msg_type == "event":
                    event = WsEventMessage.from_dict(data)
                    for cb in self._event_callbacks:
                        try:
                            result = cb(event)
                            if asyncio.iscoroutine(result):
                                await result
                        except Exception:
                            self._logger.exception("event_callback_error")

                elif msg_type == "error":
                    error = WsErrorMessage.from_dict(data)
                    # If we're waiting for an ack, reject it with the error
                    if self._pending_ack and not self._pending_ack.done():
                        from .exceptions import VectorDBError
                        self._pending_ack.set_exception(
                            VectorDBError(error.message, code=error.code)
                        )
                    for cb in self._error_callbacks:
                        try:
                            result = cb(error)
                            if asyncio.iscoroutine(result):
                                await result
                        except Exception:
                            self._logger.exception("error_callback_error")

                elif msg_type == "ping":
                    # Server-initiated ping → respond with pong
                    await self._ws.send(json.dumps({"type": "pong"}))

                else:
                    self._logger.warning("unknown_message_type", type=msg_type)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._logger.warning("listen_loop_ended", error=str(e))
        finally:
            self._connected = False
            for cb in self._close_callbacks:
                try:
                    result = cb()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    self._logger.exception("close_callback_error")
