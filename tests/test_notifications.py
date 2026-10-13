import os
from unittest.mock import patch, MagicMock
import pytest

from notifications.telegram import (
    escape_mdv2, format_event, _TelegramBucket, _Sender, TelegramNotifier,
    _normalize_chat_id,
)


class _Clock:
    def __init__(self, virtual=False): self._v = virtual; self._t = 1000.0
    def now(self): return self._t
    @property
    def mode(self): return "virtual" if self._v else "real"


class _Metrics:
    def __init__(self): self.c = {}
    def inc(self, k, n=1, **__): self.c[k] = self.c.get(k, 0) + n
    def set(self, *_, **__): pass
    def observe(self, *_, **__): pass
    def snapshot(self): return dict(self.c)


def _bus():
    from core.bus import MessageBus
    from core.event_log import InMemoryEventLog
    from core.schema_registry import register_default_schemas
    register_default_schemas()
    return MessageBus(event_log=InMemoryEventLog())


# ---------- Formatting ---------------------------------------------------- #

def test_escape_mdv2_escapes_specials():
    assert escape_mdv2("a_b*c[d]") == r"a\_b\*c\[d\]"
    assert escape_mdv2("hello world") == "hello world"


def test_format_order_proposal_contains_fields():
    msg = format_event("order_proposal", {
        "symbol": "EURUSD", "side": "buy", "direction": 1.0,
        "qty": 1.0, "source": "release_drift",
    }, ts=1_700_000_000.0)
    assert "EURUSD" in msg
    assert "release\\_drift" in msg
    assert "ORDER PROPOSAL" in msg


def test_format_risk_veto_shows_decision():
    msg = format_event("risk_veto", {
        "symbol": "EURUSD", "decision": "deny", "reasons": ["cap breached"],
    }, ts=1_700_000_000.0)
    assert "DENY" in msg
    assert "cap breached" in msg


def test_format_order_fill_close_shows_pnl():
    msg = format_event("order_fill", {
        "symbol": "X", "side": "buy", "filled_notional": 100.0,
        "price": 1.2345, "closed": True, "pnl": 12.34,
    }, ts=1_700_000_000.0)
    assert "CLOSE" in msg
    assert "+12.34" in msg


def test_format_generic_fallback():
    msg = format_event("some_unknown_event", {"a": 1, "b": "x"}, ts=1_700_000_000.0)
    assert "SOME\\_UNKNOWN\\_EVENT" in msg
    assert "`1`" in msg


# ---------- Rate limiter -------------------------------------------------- #

def test_bucket_allows_burst_then_throttles():
    b = _TelegramBucket(rate_per_min=60.0, burst=3.0)
    for _ in range(3):
        assert b.take() == 0.0
    wait = b.take()
    assert wait > 0.0


# ---------- Sender -------------------------------------------------------- #

def test_sender_enqueue_and_send(monkeypatch):
    sender = _Sender("fake:token", "-100test", rate_per_min=600.0, queue_size=10)

    calls = []
    class _R:
        status_code = 200
        headers = {}
        text = "ok"

    def fake_post(url, data=None, timeout=None, **kwargs):
        calls.append((url, data))
        return _R()

    monkeypatch.setattr("notifications.telegram.requests.post", fake_post)
    sender.enqueue("hello", "info")
    import time; time.sleep(0.5)
    sender.close()
    assert calls, "expected at least one POST"
    assert calls[0][1]["chat_id"] == "-100test"
    assert calls[0][1]["parse_mode"] == "MarkdownV2"


def test_sender_drops_oldest_when_full(monkeypatch):
    sender = _Sender("fake:token", "-100test", rate_per_min=1.0, queue_size=2)
    sender._stop.set()
    for i in range(4):
        sender.enqueue(f"m{i}", "info")
    assert sender.dropped >= 1


# ---------- Component ----------------------------------------------------- #

def test_notifier_inert_without_credentials(monkeypatch):
    from core.config import Config
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    cfg = Config.load()
    cfg.set("notifications.telegram.enabled", True)
    n = TelegramNotifier(_bus(), cfg, _Clock(), _Metrics())
    n.start()
    assert n._sender is None


def test_notifier_disabled_by_config():
    from core.config import Config
    cfg = Config.load()
    cfg.set("notifications.telegram.enabled", False)
    n = TelegramNotifier(_bus(), cfg, _Clock(), _Metrics())
    n.start()
    assert n._sender is None


def test_notifier_replay_mode_suppresses(monkeypatch):
    from core.config import Config
    from core.events import make_event
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake:token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100test")
    cfg = Config.load()
    cfg.set("notifications.telegram.enabled", True)
    cfg.set("notifications.telegram.allow_in_replay", False)

    n = TelegramNotifier(_bus(), cfg, _Clock(virtual=True), _Metrics())
    n.start()
    assert n._sender is not None

    sent = []
    n._sender.enqueue = lambda text, sev: (sent.append((text, sev)) or True)
    n._on_event(make_event(
        "order_proposal", "test",
        payload={"symbol": "EURUSD", "side": "buy", "direction": 1.0,
                 "qty": 1.0, "source": "test"},
        timestamp=1_700_000_000.0,
    ))
    assert sent == []
    n._sender.close()


def test_notifier_emits_on_order_proposal(monkeypatch):
    from core.config import Config
    from core.events import make_event
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake:token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100test")
    cfg = Config.load()
    cfg.set("notifications.telegram.enabled", True)

    n = TelegramNotifier(_bus(), cfg, _Clock(), _Metrics())
    n.start()

    captured = []
    n._sender.enqueue = lambda text, sev: (captured.append((text, sev)) or True)
    n._on_event(make_event(
        "order_proposal", "test",
        payload={"symbol": "EURUSD", "side": "buy", "direction": 1.0,
                 "qty": 1.0, "source": "release_drift"},
        timestamp=1_700_000_000.0,
    ))
    assert captured and "EURUSD" in captured[0][0]
    n._sender.close()


def test_notifier_ignores_info_when_min_severity_warning(monkeypatch):
    from core.config import Config
    from core.events import make_event
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake:token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100test")
    cfg = Config.load()
    cfg.set("notifications.telegram.enabled", True)
    cfg.set("notifications.telegram.min_severity", "warning")

    n = TelegramNotifier(_bus(), cfg, _Clock(), _Metrics())
    n.start()
    captured = []
    n._sender.enqueue = lambda text, sev: (captured.append((text, sev)) or True)

    n._on_event(make_event("order_proposal", "test",
                           payload={"symbol": "X"}, timestamp=1.0))
    n._on_event(make_event("order_reject", "test",
                           payload={"symbol": "X", "reason": "requote"},
                           timestamp=2.0))
    assert len(captured) == 1
    assert "REJECT" in captured[0][0]
    n._sender.close()


# ---------- Relay channel ------------------------------------------------- #

@pytest.mark.parametrize("raw,expected", [
    ("@Cryptonaute_Engager", "@Cryptonaute_Engager"),
    ("Cryptonaute_Engager", "@Cryptonaute_Engager"),
    ("t.me/Cryptonaute_Engager", "@Cryptonaute_Engager"),
    ("https://t.me/Cryptonaute_Engager", "@Cryptonaute_Engager"),
    ("-1001234567890", "-1001234567890"),
    ("", ""),
    (None, ""),
])
def test_normalize_chat_id(raw, expected):
    assert _normalize_chat_id(raw) == expected


def test_notifier_routes_only_confirmed_events_to_relay(monkeypatch):
    from core.config import Config
    from core.events import make_event
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake:token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-100main")
    monkeypatch.setenv("RELAY_CHANNEL", "Cryptonaute_Engager")
    cfg = Config.load()
    cfg.set("notifications.telegram.enabled", True)

    n = TelegramNotifier(_bus(), cfg, _Clock(), _Metrics())
    n.start()
    assert n._sender is not None
    assert n._relay_sender is not None
    assert n._relay_sender.chat_id == "@Cryptonaute_Engager"

    main, relay = [], []
    n._sender.enqueue = lambda text, sev: (main.append(text) or True)
    n._relay_sender.enqueue = lambda text, sev: (relay.append(text) or True)

    # A proposal is not a confirmed outcome — main only.
    n._on_event(make_event("order_proposal", "test",
                           payload={"symbol": "EURUSD", "side": "buy",
                                    "direction": 1.0, "qty": 1.0, "source": "t"},
                           timestamp=1.0))
    assert len(main) == 1 and relay == []

    # A fill is a confirmed outcome — main AND relay.
    n._on_event(make_event("order_fill", "test",
                           payload={"symbol": "EURUSD", "side": "buy",
                                    "filled_notional": 100.0, "price": 1.1,
                                    "closed": True, "pnl": 1.0},
                           timestamp=2.0))
    assert len(main) == 2 and len(relay) == 1
    assert "ORDER FILL" in relay[0]

    n._sender.close()
    n._relay_sender.close()
