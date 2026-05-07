"""Tradovate market-data WebSocket client.

Implements the SockJS-style framing the API uses, handles auth + heartbeats,
and exposes a chart-subscription primitive that yields parsed `Bar`s.

Reference: https://api.tradovate.com/#section/Real-time-Data
Wire format (client -> server commands):

    <endpoint>\\n<requestId>\\n<query>\\n<body>

Wire format (server -> client frames, single text per WS message):

    "o"           — connection opened
    "h"           — heartbeat
    "a[<json>]"   — array of app messages (each element is a JSON-encoded dict)
    "c[<code>,<reason>]" — close
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

from websockets.asyncio.client import ClientConnection
from websockets.asyncio.client import connect as ws_connect

from futures_bot.types import Bar

log = logging.getLogger(__name__)

MD_WS_URL = "wss://md.tradovateapi.com/v1/websocket"


class MDProtocolError(RuntimeError):
    pass


# --- pure helpers (unit-tested without a network) ----------------------------


def format_command(endpoint: str, request_id: int, body: Any = None, query: str = "") -> str:
    """Encode a single client command into the wire format."""
    if body is None:
        body_str = ""
    elif isinstance(body, str):
        body_str = body
    else:
        body_str = json.dumps(body, separators=(",", ":"))
    return f"{endpoint}\n{request_id}\n{query}\n{body_str}"


def parse_server_frame(frame: str) -> tuple[str, Any]:
    """Decode a single server frame.

    Returns a (kind, payload) tuple where `kind` is one of:
        "open", "heartbeat", "messages", "close", "unknown".
    For "messages" the payload is a list[dict].
    For "close" the payload is (code:int|None, reason:str|None).
    """
    if not frame:
        return ("unknown", frame)
    head = frame[0]
    if head == "o":
        return ("open", None)
    if head == "h":
        return ("heartbeat", None)
    if head == "a":
        try:
            arr = json.loads(frame[1:])
        except json.JSONDecodeError as e:
            raise MDProtocolError(f"bad app frame: {frame!r}") from e
        return ("messages", [json.loads(m) if isinstance(m, str) else m for m in arr])
    if head == "c":
        try:
            code, reason = json.loads(frame[1:])
        except Exception:
            code, reason = None, frame[1:]
        return ("close", (code, reason))
    return ("unknown", frame)


def decode_chart_packet(packet: dict) -> list[Bar]:
    """Convert a Tradovate `chart` event payload into Bars.

    Tradovate packs prices as int offsets in tick units relative to a chart
    base price: `actual = bp + offset * ts`. Some payloads return raw floats
    instead — handle both.
    """
    out: list[Bar] = []
    for chart in packet.get("charts", []):
        bp = float(chart.get("bp", 0.0))
        ts = float(chart.get("ts", 1.0))
        for bar in chart.get("bars", []):
            ts_str = bar.get("timestamp")
            if not ts_str:
                continue
            ts_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            # If close looks like a small int offset, unpack; otherwise trust as price.
            o = _unpack_price(bar.get("open"), bp, ts)
            h = _unpack_price(bar.get("high"), bp, ts)
            lo = _unpack_price(bar.get("low"), bp, ts)
            c = _unpack_price(bar.get("close"), bp, ts)
            vol = float(bar.get("upVolume", 0)) + float(bar.get("downVolume", 0))
            out.append(Bar(ts=ts_dt, open=o, high=h, low=lo, close=c, volume=vol))
    return out


def _unpack_price(raw: Any, bp: float, ts: float) -> float:
    if raw is None:
        return bp
    val = float(raw)
    # Heuristic: Tradovate packed offsets are within ~few thousand ticks of bp.
    # If `bp` is non-zero AND `val` is within a plausible offset range, unpack.
    # Otherwise treat as a full price.
    if bp != 0 and abs(val) < 1_000_000 and abs(val * ts) < bp:
        return bp + val * ts
    return val


# --- live client -------------------------------------------------------------


class TradovateMD:
    def __init__(self, access_token: str, url: str = MD_WS_URL) -> None:
        self._token = access_token
        self._url = url
        self._ws: ClientConnection | None = None
        self._next_id = 0
        self._authorized = asyncio.Event()
        self._reader: asyncio.Task[None] | None = None
        # subscriptionId -> queue of bars
        self._chart_queues: dict[int, asyncio.Queue[list[Bar] | None]] = {}
        # requestId -> future for command responses
        self._pending: dict[int, asyncio.Future[dict]] = {}

    # --- lifecycle ---------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and self._authorized.is_set()

    async def connect(self) -> None:
        self._ws = await ws_connect(self._url, ping_interval=None)
        self._reader = asyncio.create_task(self._reader_loop(), name="tradovate-md-reader")
        try:
            await asyncio.wait_for(self._authorized.wait(), timeout=15.0)
        except TimeoutError as e:
            await self.close()
            raise MDProtocolError("authorization timed out") from e

    async def close(self) -> None:
        for q in self._chart_queues.values():
            q.put_nowait(None)
        if self._reader:
            self._reader.cancel()
        if self._ws:
            await self._ws.close()
            self._ws = None

    # --- chart subscription ------------------------------------------------

    async def stream_chart(
        self,
        symbol: str,
        bar_seconds: int,
        history_elements: int = 1,
    ) -> AsyncIterator[Bar]:
        """Subscribe to a chart and yield Bars (history first, then live updates)."""
        sub_id = await self._subscribe_chart(symbol, bar_seconds, history_elements)
        q = self._chart_queues[sub_id]
        try:
            while True:
                bars = await q.get()
                if bars is None:
                    return
                for bar in bars:
                    yield bar
        finally:
            await self._cancel_chart(sub_id)

    async def _subscribe_chart(self, symbol: str, bar_seconds: int, history: int) -> int:
        underlying, element_size = _bars_to_chart_desc(bar_seconds)
        body = {
            "symbol": symbol,
            "chartDescription": {
                "underlyingType": underlying,
                "elementSize": element_size,
                "elementSizeUnit": "UnderlyingUnits",
                "withHistogram": False,
            },
            "timeRange": {"asMuchAsElements": max(1, history)},
        }
        resp = await self._send_and_wait("md/getChart", body)
        # Tradovate returns either {"subscriptionId": N} or {"realtimeId": N, "historicalId": N}.
        sub_id = resp.get("subscriptionId") or resp.get("realtimeId") or resp.get("historicalId")
        if sub_id is None:
            raise MDProtocolError(f"md/getChart did not return a subscription id: {resp}")
        sub_id = int(sub_id)
        self._chart_queues[sub_id] = asyncio.Queue()
        return sub_id

    async def _cancel_chart(self, sub_id: int) -> None:
        try:
            await self._send_and_wait("md/cancelChart", {"subscriptionId": sub_id})
        except Exception:
            log.debug("cancelChart failed (subscription may have already closed)", exc_info=True)
        self._chart_queues.pop(sub_id, None)

    # --- command plumbing --------------------------------------------------

    async def _send_and_wait(self, endpoint: str, body: Any) -> dict:
        if self._ws is None:
            raise RuntimeError("not connected")
        self._next_id += 1
        rid = self._next_id
        fut: asyncio.Future[dict] = asyncio.get_event_loop().create_future()
        self._pending[rid] = fut
        await self._ws.send(format_command(endpoint, rid, body=body))
        try:
            return await asyncio.wait_for(fut, timeout=15.0)
        finally:
            self._pending.pop(rid, None)

    async def _send_authorize(self) -> None:
        if self._ws is None:
            raise RuntimeError("not connected")
        self._next_id += 1
        rid = self._next_id
        await self._ws.send(format_command("authorize", rid, body=self._token))

    # --- reader loop -------------------------------------------------------

    async def _reader_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                kind, payload = parse_server_frame(raw)
                if kind == "open":
                    await self._send_authorize()
                elif kind == "heartbeat":
                    continue
                elif kind == "close":
                    log.warning("MD socket closed by server: %s", payload)
                    return
                elif kind == "messages":
                    for msg in payload:
                        self._dispatch_message(msg)
                else:
                    log.debug("unknown MD frame: %s", raw[:80])
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("MD reader loop crashed")
            for q in self._chart_queues.values():
                q.put_nowait(None)

    def _dispatch_message(self, msg: dict) -> None:
        # Command response: has "i" (request id) and "s" (status).
        if "i" in msg and "s" in msg:
            rid = int(msg["i"])
            data = msg.get("d", {})
            fut = self._pending.get(rid)
            if fut and not fut.done():
                if int(msg["s"]) >= 400:
                    fut.set_exception(MDProtocolError(f"command {rid} failed: {data}"))
                else:
                    fut.set_result(data if isinstance(data, dict) else {"d": data})
            # The first successful response is to authorize — flip the gate.
            if not self._authorized.is_set() and int(msg["s"]) < 400:
                self._authorized.set()
            return
        # Event: {"e": "<kind>", "d": {...}}
        event = msg.get("e")
        if event == "chart":
            self._dispatch_chart(msg.get("d", {}))
        # Other events (props, tick, etc.) are ignored for now.

    def _dispatch_chart(self, packet: dict) -> None:
        bars = decode_chart_packet(packet)
        if not bars:
            return
        # Each chart entry has its own id; route bars to the right queue.
        for chart in packet.get("charts", []):
            cid = chart.get("id")
            if cid is None:
                continue
            q = self._chart_queues.get(int(cid))
            if q is None:
                continue
            chart_bars = decode_chart_packet({"charts": [chart]})
            if chart_bars:
                q.put_nowait(chart_bars)


def _bars_to_chart_desc(bar_seconds: int) -> tuple[str, int]:
    """Map our bar_seconds to a Tradovate (underlyingType, elementSize) pair.

    Currently supports minute-multiples (e.g. 60 -> 1m, 300 -> 5m). Sub-minute
    bars require tick aggregation, which we don't do yet.
    """
    if bar_seconds <= 0 or bar_seconds % 60 != 0:
        raise NotImplementedError(
            f"bar_seconds={bar_seconds} not supported yet; use a multiple of 60"
        )
    return "MinuteBar", bar_seconds // 60
