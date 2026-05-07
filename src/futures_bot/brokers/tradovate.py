"""Tradovate REST + (stubbed) WebSocket broker.

Auth flow reference: https://api.tradovate.com/#tag/Authentication

The demo environment uses `demo.tradovateapi.com`; live uses `live.tradovateapi.com`.
This module implements auth + a small, typed surface used by the runner.
WebSocket streaming is intentionally a stub — wire it up per the strategy you ship.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from futures_bot.brokers.base import Broker
from futures_bot.config import Secrets
from futures_bot.types import Bar, Order, OrderStatus, OrderType, Position, Side

log = logging.getLogger(__name__)

_REST_BASES = {
    "demo": "https://demo.tradovateapi.com/v1",
    "live": "https://live.tradovateapi.com/v1",
}


class TradovateAuthError(RuntimeError):
    pass


class TradovateBroker(Broker):
    def __init__(self, secrets: Secrets, account_spec: str | None = None) -> None:
        self._secrets = secrets
        self._account_spec = account_spec  # account name or id; None == first account
        self._base_url = _REST_BASES[secrets.tradovate_env]
        self._client: httpx.AsyncClient | None = None
        self._access_token: str | None = None
        self._token_expiry: datetime | None = None
        self._account_id: int | None = None

    # --- lifecycle -----------------------------------------------------------

    async def connect(self) -> None:
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=15.0)
        await self._authenticate()
        await self._resolve_account()
        log.info("Tradovate connected (env=%s, account_id=%s)",
                 self._secrets.tradovate_env, self._account_id)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # --- auth ----------------------------------------------------------------

    async def _authenticate(self) -> None:
        s = self._secrets
        if not (s.tradovate_username and s.tradovate_password):
            raise TradovateAuthError("TRADOVATE_USERNAME/PASSWORD not set")
        body = {
            "name": s.tradovate_username,
            "password": s.tradovate_password,
            "appId": s.tradovate_app_id,
            "appVersion": s.tradovate_app_version,
            "cid": int(s.tradovate_cid) if s.tradovate_cid else None,
            "sec": s.tradovate_secret or None,
        }
        body = {k: v for k, v in body.items() if v is not None}
        resp = await self._http().post("/auth/accesstokenrequest", json=body)
        resp.raise_for_status()
        data = resp.json()
        if "accessToken" not in data:
            raise TradovateAuthError(f"auth failed: {data}")
        self._access_token = data["accessToken"]
        # Tradovate returns expiration as ISO-8601 in `expirationTime`.
        expires = data.get("expirationTime")
        if expires:
            self._token_expiry = datetime.fromisoformat(expires.replace("Z", "+00:00"))
        else:
            self._token_expiry = datetime.now(UTC) + timedelta(minutes=60)

    def _auth_headers(self) -> dict[str, str]:
        if not self._access_token:
            raise TradovateAuthError("not authenticated")
        return {"Authorization": f"Bearer {self._access_token}"}

    async def _ensure_token_fresh(self) -> None:
        if self._token_expiry is None:
            await self._authenticate()
            return
        if datetime.now(UTC) >= self._token_expiry - timedelta(minutes=2):
            await self._authenticate()

    async def _resolve_account(self) -> None:
        resp = await self._http().get("/account/list", headers=self._auth_headers())
        resp.raise_for_status()
        accounts = resp.json()
        if not accounts:
            raise TradovateAuthError("no Tradovate accounts available")
        if self._account_spec:
            for a in accounts:
                if str(a.get("id")) == self._account_spec or a.get("name") == self._account_spec:
                    self._account_id = a["id"]
                    return
            raise TradovateAuthError(f"account '{self._account_spec}' not found")
        self._account_id = accounts[0]["id"]

    # --- helpers -------------------------------------------------------------

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("call connect() first")
        return self._client

    async def _get(self, path: str, **kwargs: Any) -> Any:
        await self._ensure_token_fresh()
        resp = await self._http().get(path, headers=self._auth_headers(), **kwargs)
        resp.raise_for_status()
        return resp.json()

    async def _post(self, path: str, json: dict[str, Any]) -> Any:
        await self._ensure_token_fresh()
        resp = await self._http().post(path, headers=self._auth_headers(), json=json)
        resp.raise_for_status()
        return resp.json()

    async def _resolve_contract_id(self, symbol: str) -> int:
        data = await self._get("/contract/find", params={"name": symbol})
        if not data or "id" not in data:
            raise ValueError(f"contract not found: {symbol}")
        return int(data["id"])

    # --- account / positions -------------------------------------------------

    async def get_account_equity(self) -> float:
        if self._account_id is None:
            raise RuntimeError("not connected")
        data = await self._get(
            "/cashBalance/getcashbalancesnapshot",
            params={"accountId": self._account_id},
        )
        # Tradovate exposes `totalCashValue` on the snapshot.
        return float(data.get("totalCashValue", 0.0))

    async def get_position(self, symbol: str) -> Position:
        contract_id = await self._resolve_contract_id(symbol)
        positions = await self._get("/position/list")
        for p in positions:
            if p.get("contractId") == contract_id and p.get("accountId") == self._account_id:
                qty = int(p.get("netPos", 0))
                return Position(
                    symbol=symbol,
                    qty=qty,
                    avg_price=float(p.get("netPrice") or 0.0),
                )
        return Position(symbol=symbol)

    # --- orders --------------------------------------------------------------

    async def place_order(
        self,
        symbol: str,
        side: Side,
        qty: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
    ) -> Order:
        if qty <= 0:
            raise ValueError("qty must be positive")
        body: dict[str, Any] = {
            "accountId": self._account_id,
            "action": "Buy" if side is Side.BUY else "Sell",
            "symbol": symbol,
            "orderQty": qty,
            "orderType": _order_type_to_tradovate(order_type),
            "isAutomated": True,
        }
        if order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
            if limit_price is None:
                raise ValueError("limit_price required")
            body["price"] = limit_price
        if order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
            if stop_price is None:
                raise ValueError("stop_price required")
            body["stopPrice"] = stop_price
        data = await self._post("/order/placeorder", body)
        order_id = str(data.get("orderId") or data.get("id") or uuid.uuid4())
        return Order(
            id=order_id,
            symbol=symbol,
            side=side,
            qty=qty,
            type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            status=OrderStatus.WORKING,
        )

    async def cancel_order(self, order_id: str) -> None:
        await self._post("/order/cancelorder", {"orderId": int(order_id)})

    # --- market data ---------------------------------------------------------

    async def historical_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        bar_seconds: int,
    ) -> list[Bar]:
        # NOTE: Tradovate's historical chart data flows over a separate
        # WebSocket (md.tradovateapi.com /v1/websocket) using `md/getChart`.
        # Wire that up before relying on this in production. Returning [] keeps
        # the surface honest for the scaffold.
        log.warning("historical_bars not wired up — returning []")
        return []

    async def stream_bars(self, symbol: str, bar_seconds: int) -> AsyncIterator[Bar]:  # type: ignore[override]
        # Same note as above — replace with a websockets.connect() loop that
        # subscribes to md/subscribeQuote or md/subscribeChart and aggregates
        # ticks into bars of `bar_seconds`.
        raise NotImplementedError("stream_bars: implement websocket market data feed")
        if False:  # pragma: no cover - keeps return type AsyncIterator[Bar]
            yield  # type: ignore[unreachable]


def _order_type_to_tradovate(t: OrderType) -> str:
    return {
        OrderType.MARKET: "Market",
        OrderType.LIMIT: "Limit",
        OrderType.STOP: "Stop",
        OrderType.STOP_LIMIT: "StopLimit",
    }[t]
