from __future__ import annotations

import json

import pytest

from futures_bot.brokers.tradovate_md import (
    MDProtocolError,
    decode_chart_packet,
    format_command,
    parse_server_frame,
)


def test_format_command_with_dict_body():
    out = format_command("md/getChart", 7, body={"symbol": "MESM6"})
    assert out == 'md/getChart\n7\n\n{"symbol":"MESM6"}'


def test_format_command_with_string_body():
    # `authorize` takes the raw token as the body, not JSON.
    out = format_command("authorize", 1, body="tok_abc")
    assert out == "authorize\n1\n\ntok_abc"


def test_format_command_no_body():
    out = format_command("md/cancelChart", 3)
    assert out == "md/cancelChart\n3\n\n"


def test_parse_open_and_heartbeat():
    assert parse_server_frame("o") == ("open", None)
    assert parse_server_frame("h") == ("heartbeat", None)


def test_parse_app_messages():
    inner = json.dumps({"i": 1, "s": 200, "d": {"ok": True}})
    frame = "a" + json.dumps([inner])
    kind, payload = parse_server_frame(frame)
    assert kind == "messages"
    assert payload == [{"i": 1, "s": 200, "d": {"ok": True}}]


def test_parse_close_frame():
    kind, payload = parse_server_frame('c[1000,"normal"]')
    assert kind == "close"
    assert payload == (1000, "normal")


def test_parse_invalid_app_frame_raises():
    with pytest.raises(MDProtocolError):
        parse_server_frame("a{not-json}")


def test_decode_chart_packet_unpacks_packed_offsets():
    # bp=4500, ts=0.25 — open offset 1 -> 4500.25, etc.
    packet = {
        "charts": [
            {
                "id": 99,
                "bp": 4500,
                "ts": 0.25,
                "bars": [
                    {
                        "timestamp": "2025-01-01T13:30:00Z",
                        "open": 1,
                        "high": 10,
                        "low": -1,
                        "close": 4,
                        "upVolume": 30,
                        "downVolume": 20,
                    }
                ],
            }
        ]
    }
    bars = decode_chart_packet(packet)
    assert len(bars) == 1
    b = bars[0]
    assert b.open == 4500.25
    assert b.high == 4502.5
    assert b.low == 4499.75
    assert b.close == 4501.0
    assert b.volume == 50.0


def test_decode_chart_packet_handles_full_prices():
    # bp=0 means no packing; values are full prices.
    packet = {
        "charts": [
            {
                "id": 1,
                "bp": 0,
                "ts": 0.25,
                "bars": [
                    {
                        "timestamp": "2025-01-01T13:30:00Z",
                        "open": 4500.25,
                        "high": 4502.5,
                        "low": 4499.75,
                        "close": 4501.0,
                    }
                ],
            }
        ]
    }
    bars = decode_chart_packet(packet)
    assert bars[0].close == 4501.0
