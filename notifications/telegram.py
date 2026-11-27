# notifications/telegram.py
"""
Telegram notification layer.

Subscribes to the bus and forwards every significant event to a Telegram
channel. Runs off the critical path (async, best-effort, never raises).
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import requests

from core.component import Component
from core.events import NormalizedEvent

log = logging.getLogger("notifications.telegram")

TG_API = "https://api.telegram.org/bot{token}/sendMessage"

_SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}

_EVENT_SEVERITY: dict[str, str] = {
    "circuit_breaker":   "critical",
    "risk_breach":       "critical",
    "veto_signature_rejected": "critical",
    "execution_veto_unverified": "critical",
    "order_reject":      "warning",
    "release_actual":    "info",
    "nowcast":           "info",
    "order_proposal":    "info",
    "risk_veto":         "info",
    "order_fill":        "info",
    "order_exit":        "info",
}


class _TelegramBucket:
    def __init__(self, rate_per_min: float = 20.0, burst: float = 3.0) -> None:
        self.rate_per_sec = rate_per_min / 60.0
        self.capacity = burst
        self.tokens = burst
        self.last = time.monotonic()
        self._lock = threading.Lock()

    def take(self) -> float:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.last
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_per_sec)
            self.last = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return 0.0
            wait = (1.0 - self.tokens) / self.rate_per_sec if self.rate_per_sec > 0 else 1.0
            self.tokens = 0.0
            return wait


_MDV2_SPECIAL = r"_*[]()~`>#+-=|{}.!"


def escape_mdv2(s: str) -> str:
    out = []
    for ch in str(s):
        if ch in _MDV2_SPECIAL:
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)


def _normalize_chat_id(raw: Any) -> str:
    """Normalize a Telegram channel reference to a Bot-API-acceptable form.

    Accepts ``@channel``, ``channel``, ``t.me/channel``, ``https://t.me/channel``
    or a numeric id (``-100...`` / ``123``). Returns ``@channel`` for usernames,
    the bare id for numeric ids, or ``""`` when nothing usable is supplied.
    """
    s = str(raw or "").strip()
    if not s:
        return ""
    low = s.lower()
    for pre in ("https://t.me/", "http://t.me/", "https://telegram.me/",
                "http://telegram.me/", "t.me/", "telegram.me/"):
        if low.startswith(pre):
            s = s[len(pre):]
            break
    s = s.strip().lstrip("@").strip()
    if not s:
        return ""
    if s.lstrip("-").isdigit():
        return s
    return "@" + s


def _code(content: str) -> str:
    """Wrap content in inline code, escaping MarkdownV2 reserved chars inside."""
    return "`" + escape_mdv2(content) + "`"


def _pnl_badge(pnl) -> str:
    if pnl is None:
        return _code("PnL: —")
    try:
        p = float(pnl)
    except (TypeError, ValueError):
        return _code(f"PnL: {str(pnl)}")
    badge = "🟢" if p >= 0 else "🔴"
    return f"{badge} {_code(f'PnL: {p:+.2f}')}"


def format_event(event_type: str, payload: dict[str, Any], ts: float) -> str:
    import datetime as _dt
    stamp = _code(_dt.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S UTC"))

    def _num(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    if event_type == "order_proposal":
        sym = str(payload.get("symbol", "?"))
        side = str(payload.get("side", "?"))
        direction = _num(payload.get("direction", 0.0))
        qty = _num(payload.get("qty", 0.0))
        src = str(payload.get("source", "strategy"))
        conf = payload.get("confidence")
        conf_txt = f"\n_conf:_ {_code(f'{float(conf):.2f}')}" if conf is not None else ""
        return (
            f"*📊 ORDER PROPOSAL*\n"
            f"🕒 {stamp}\n"
            f"🪙 {_code(sym)}\n"
            f"📈 {_code(side)} ({_code(f'{direction:+.0f}')})\n"
            f"📦 {_code(f'{qty}')}\n"
            f"🔧 {_code(src)}{conf_txt}"
        )

    if event_type == "risk_veto":
        sym = str(payload.get("symbol", "?"))
        decision = str(payload.get("decision", "?"))
        reasons = payload.get("reasons") or []
        reasons_txt = "; ".join(str(r) for r in reasons) or "none"
        rule = str(payload.get("rule", "—"))
        return (
            f"*🛡️ RISK VETO* — {_code(decision.upper())}\n"
            f"🕒 {stamp}\n"
            f"🪙 {_code(sym)}\n"
            f"📋 {_code(rule)}\n"
            f"🚫 {_code(reasons_txt)}"
        )

    if event_type == "order_fill":
        sym = str(payload.get("symbol", "?"))
        side = str(payload.get("side", "?"))
        vol = _num(payload.get("filled_notional", payload.get("volume", 0.0)))
        px = _num(payload.get("price", 0.0))
        closed = payload.get("closed", False)
        pnl = payload.get("pnl")
        ot = str(payload.get("order_type", "market"))
        ticket = str(payload.get("ticket", "—"))
        head = "🟢 ORDER FILL (CLOSE)" if closed else "🟢 ORDER FILL (OPEN)"
        return (
            f"*{escape_mdv2(head)}*\n"
            f"🕒 {stamp}\n"
            f"🪙 {_code(sym)}\n"
            f"📈 {_code(side)}\n"
            f"💵 {_code(f'{vol:.2f}')}\n"
            f"💰 {_code(f'{px:.6f}')}\n"
            f"🏷️ {_code(ot)}\n"
            f"🎫 {_code(ticket)}\n"
            f"{_pnl_badge(pnl)}"
        )

    if event_type == "order_exit":
        sym = str(payload.get("symbol", "?"))
        side = str(payload.get("side", "?"))
        vol = _num(payload.get("filled_notional", payload.get("volume", 0.0)))
        px = _num(payload.get("price", 0.0))
        pnl = payload.get("pnl")
        reason = str(payload.get("reason", "—"))
        ticket = str(payload.get("ticket", "—"))
        return (
            f"*🏁 ORDER EXIT*\n"
            f"🕒 {stamp}\n"
            f"🪙 {_code(sym)}\n"
            f"📈 {_code(side)}\n"
            f"💵 {_code(f'{vol:.2f}')}\n"
            f"💰 {_code(f'{px:.6f}')}\n"
            f"{_pnl_badge(pnl)}\n"
            f"🎫 {_code(ticket)}\n"
            f"📋 {_code(reason)}"
        )

    if event_type == "order_reject":
        sym = str(payload.get("symbol", "?"))
        reason = str(payload.get("reason", "?"))
        code = str(payload.get("code", "—"))
        return (
            f"*🔴 ORDER REJECT*\n"
            f"🕒 {stamp}\n"
            f"🪙 {_code(sym)}\n"
            f"📋 {_code(code)}\n"
            f"🚫 {_code(reason)}"
        )

    if event_type == "release_actual":
        rid = str(payload.get("release_id", "?"))
        act = _num(payload.get("actual", 0.0))
        cons = _num(payload.get("consensus", 0.0))
        surp = _num(payload.get("surprise", 0.0))
        unit = str(payload.get("unit", ""))
        unit_txt = f" {_code(unit)}" if unit else ""
        return (
            f"*📰 RELEASE ACTUAL*\n"
            f"🕒 {stamp}\n"
            f"📅 {_code(rid)}\n"
            f"📊 {_code(f'{act}')}{unit_txt}\n"
            f"🎯 {_code(f'{cons}')}{unit_txt}\n"
            f"⚡ {_code(f'{surp:+.2f}σ')}"
        )

    if event_type == "nowcast":
        rid = str(payload.get("release_id", "?"))
        mean = _num(payload.get("mean", 0.0))
        sd = _num(payload.get("stddev", 0.0))
        edge = _num(payload.get("edge_vs_consensus", 0.0))
        return (
            f"*📈 NOWCAST*\n"
            f"🕒 {stamp}\n"
            f"📅 {_code(rid)}\n"
            f"🔮 {_code(f'{mean:.4f} σ={sd:.4f}')}\n"
            f"⚡ {_code(f'{edge:+.4f}')} vs consensus"
        )

    if event_type in ("circuit_breaker", "risk_breach"):
        state = str(payload.get("state") or payload.get("reason", "?"))
        val = _num(payload.get("value", 0.0))
        limit = payload.get("limit", "—")
        limit_txt = f"\n📏 {_code(str(limit))}" if limit is not None else ""
        return (
            f"*🚨 CIRCUIT BREAKER*\n"
            f"🕒 {stamp}\n"
            f"📛 {_code(state)}\n"
            f"📉 {_code(f'{val:.2f}')}{limit_txt}"
        )

    pairs = []
    for k, v in list(payload.items())[:10]:
        pairs.append(f"{escape_mdv2(str(k))}: {_code(str(v)[:100])}")
    body = "\n".join(pairs)
    return f"*ℹ️ {escape_mdv2(event_type.upper())}*\n🕒 {stamp}\n{body}"


@dataclass
class _Msg:
    text: str
    severity: str = "info"


class _Sender:
    def __init__(self, token: str, chat_id: str, *, rate_per_min: float = 20.0,
                 queue_size: int = 500) -> None:
        self.token = token
        self.chat_id = chat_id
        self.url = TG_API.format(token=token)
        self.bucket = _TelegramBucket(rate_per_min=rate_per_min)
        self.q: queue.Queue[_Msg | None] = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self.sent = 0
        self.dropped = 0
        self.errors = 0
        # Start the worker only after the counters exist (the thread reads them).
        self._thread = threading.Thread(target=self._run, name="tg-sender", daemon=True)
        self._thread.start()

    def enqueue(self, text: str, severity: str) -> bool:
        try:
            self.q.put_nowait(_Msg(text, severity))
            return True
        except queue.Full:
            try:
                self.q.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
            try:
                self.q.put_nowait(_Msg(text, severity))
                return True
            except queue.Full:
                self.dropped += 1
                return False

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            wait = self.bucket.take()
            if wait > 0:
                time.sleep(wait)
            try:
                r = requests.post(
                    self.url,
                    data={
                        "chat_id": self.chat_id,
                        "text": item.text,
                        "parse_mode": "MarkdownV2",
                        "disable_web_page_preview": "true",
                    },
                    timeout=8.0,
                )
                if r.status_code == 200:
                    self.sent += 1
                elif r.status_code == 429:
                    ra = float(r.headers.get("Retry-After", "2") or 2)
                    time.sleep(min(ra, 30.0))
                    self.enqueue(item.text, item.severity)
                else:
                    self.errors += 1
                    log.warning("telegram send failed %s: %s", r.status_code, r.text[:200])
            except Exception as exc:
                self.errors += 1
                log.debug("telegram send exception: %s", exc)

    def close(self) -> None:
        self._stop.set()
        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)


class TelegramNotifier(Component):
    def __init__(self, bus, cfg, clock, metrics) -> None:
        super().__init__(bus, cfg, clock, metrics, name="notifications.telegram")
        n = cfg.get("notifications.telegram", {}) or {}
        self.enabled = bool(n.get("enabled", False))
        self.allow_in_replay = bool(n.get("allow_in_replay", False))
        self.min_severity = str(n.get("min_severity", "info")).lower()
        self.events = set(n.get("events") or [
            "order_proposal", "risk_veto", "order_fill", "order_reject",
            "circuit_breaker", "release_actual", "nowcast", "order_exit",
        ])

        token_env = str(n.get("bot_token_env", "TELEGRAM_BOT_TOKEN"))
        chat_env = str(n.get("chat_id_env", "TELEGRAM_CHAT_ID"))
        relay_env = str(n.get("relay_channel_env", "RELAY_CHANNEL"))
        self.token = cfg.secret(token_env)
        self.chat_id = n.get("chat_id") or cfg.secret(chat_env)
        # Relay/aggregator channel — receives ONLY confirmed signals + outcomes.
        self.relay_channel = n.get("relay_channel") or cfg.secret(relay_env)
        # Relay mirrors every subscribed event unless explicitly restricted.
        self.relay_events = set(n.get("relay_events") or self.events)

        rate = float(n.get("rate_per_min", 20.0))
        qsize = int(n.get("queue_size", 500))
        self._sender: _Sender | None = None
        self._relay_sender: _Sender | None = None
        if self.enabled and self.token:
            if self.chat_id:
                self._sender = _Sender(
                    token=self.token,
                    chat_id=str(self.chat_id),
                    rate_per_min=rate,
                    queue_size=qsize,
                )
            relay_ref = _normalize_chat_id(self.relay_channel)
            if relay_ref and relay_ref != _normalize_chat_id(self.chat_id):
                self._relay_sender = _Sender(
                    token=self.token,
                    chat_id=relay_ref,
                    rate_per_min=rate,
                    queue_size=qsize,
                )

    def start(self) -> None:
        self._running = True
        if not self.enabled:
            log.info("Telegram notifier disabled")
            return
        if self._sender is None and self._relay_sender is None:
            log.warning("Telegram notifier: missing bot_token or chat_id — inert")
            return

        for et in self.events:
            try:
                self.bus.subscribe(et, self._on_event)
            except Exception:
                pass
        self.bus.subscribe(None, self._on_any)

        self._send("🟢 *Cryptonaute online* — notifications active",
                   severity="info")
        self._send("🟢 *Cryptonaute online* — confirmed signals relay active",
                   severity="info", relay=True)
        log.info("Telegram notifier online (chat=%s, relay=%s, events=%d)",
                 str(self.chat_id)[:8] + "…",
                 _normalize_chat_id(self.relay_channel) or "-",
                 len(self.events))

    def _on_any(self, ev: NormalizedEvent) -> None:
        try:
            et = ev.type if isinstance(ev.type, str) else ev.type.value
        except Exception:
            return
        sev = _EVENT_SEVERITY.get(et, "info")
        if _SEVERITY_ORDER.get(sev, 0) < _SEVERITY_ORDER.get(self.min_severity, 0):
            return
        if sev == "critical" and et not in self.events:
            self._on_event(ev)

    def _on_event(self, ev: NormalizedEvent) -> None:
        if self._sender is None and self._relay_sender is None:
            return
        if getattr(self.clock, "mode", "real") == "virtual" and not self.allow_in_replay:
            return
        try:
            et = ev.type if isinstance(ev.type, str) else ev.type.value
        except Exception:
            return
        if et not in self.events and _EVENT_SEVERITY.get(et, "info") != "critical":
            return
        sev = _EVENT_SEVERITY.get(et, "info")
        if _SEVERITY_ORDER.get(sev, 0) < _SEVERITY_ORDER.get(self.min_severity, 0):
            return
        text = format_event(et, dict(ev.payload), float(ev.timestamp))
        if self._sender is not None:
            if self._sender.enqueue(text, sev):
                self.metrics.inc("telegram_queued", event_type=et)
            else:
                self.metrics.inc("telegram_dropped", event_type=et)
        # Relay gets ONLY confirmed signals + outcomes (fills/exits).
        if self._relay_sender is not None and et in self.relay_events:
            if self._relay_sender.enqueue(text, sev):
                self.metrics.inc("telegram_relay_queued", event_type=et)
            else:
                self.metrics.inc("telegram_relay_dropped", event_type=et)

    def _send(self, text: str, severity: str = "info", relay: bool = False) -> None:
        sender = self._relay_sender if relay else self._sender
        if sender is not None:
            sender.enqueue(text, severity)

    def stop(self) -> None:
        if self._sender is not None:
            self._send("🔴 *Cryptonaute offline* — notifications paused",
                       severity="info")
        if self._relay_sender is not None:
            self._send("🔴 *Cryptonaute offline* — relay paused",
                       severity="info", relay=True)
        time.sleep(0.5)
        for sender in (self._sender, self._relay_sender):
            if sender is not None:
                sender.close()

    def stats(self) -> dict[str, int]:
        s = self._sender
        r = self._relay_sender
        return {
            "sent": s.sent if s else 0,
            "dropped": s.dropped if s else 0,
            "errors": s.errors if s else 0,
            "relay_sent": r.sent if r else 0,
            "relay_dropped": r.dropped if r else 0,
            "relay_errors": r.errors if r else 0,
        }
