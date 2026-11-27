"""Normalization layer (Layer 1 — §2.3).

Every raw market/news item is coerced into the canonical :class:`core.events.NormalizedEvent`
schema (timestamp, source, type, payload, symbol_impact) before it enters the bus. Keeping
normalization in one place means the analytics/fusion layers never special-case the origin of
an item (MT5 tick vs Binance tick, etc.).
"""

from __future__ import annotations

import logging

from core.events import EventType, make_event

log = logging.getLogger("data.normalizer")


def normalize_tick(source, symbol, *, bid, ask, last=None, volume=0.0, time=None, sequence=0):
    payload = {"symbol": symbol, "bid": float(bid), "ask": float(ask),
               "mid": (float(bid) + float(ask)) / 2.0, "last": float(last) if last is not None else None,
               "volume": float(volume)}
    return make_event(EventType.TICK, source, payload, symbol_impact=[symbol], timestamp=time, sequence=sequence)


def normalize_candle(source, symbol, timeframe, *, open_, high, low, close, volume=0.0, time=None, closed=True, sequence=0):
    payload = {"symbol": symbol, "timeframe": timeframe, "open": float(open_), "high": float(high),
               "low": float(low), "close": float(close), "volume": float(volume), "closed": closed}
    return make_event(EventType.CANDLE, source, payload, symbol_impact=[symbol], timestamp=time, sequence=sequence)


def normalize_order_book(source, symbol, *, bids, asks, time=None, sequence=0):
    mid = (bids[0][0] + asks[0][0]) / 2.0 if (bids and asks) else None
    payload = {"symbol": symbol, "bids": bids, "asks": asks, "mid": mid}
    return make_event(EventType.ORDER_BOOK, source, payload, symbol_impact=[symbol], timestamp=time, sequence=sequence)


def normalize_news(source, *, title, summary="", url="", language="en", time=None, tags=None):
    payload = {"title": title, "summary": summary, "url": url, "language": language, "tags": tags or []}
    return make_event(EventType.NEWS, source, payload, symbol_impact=[], timestamp=time)


def normalize_news_score(source, *, headline, symbol_impact, score, confidence, explanation="", time=None):
    payload = {"headline": headline, "score": float(score), "confidence": float(confidence), "explanation": explanation}
    return make_event(EventType.NEWS_SCORE, source, payload, symbol_impact=symbol_impact, timestamp=time)


def normalize_macro(source, *, event, country, impact, actual=None, forecast=None, previous=None, time=None, symbol_impact=None):
    payload = {"event": event, "country": country, "impact": impact, "actual": actual, "forecast": forecast,
               "previous": previous, "surprise": (actual - forecast) if (actual is not None and forecast is not None) else None}
    return make_event(EventType.MACRO, source, payload, symbol_impact=symbol_impact or [], timestamp=time)
