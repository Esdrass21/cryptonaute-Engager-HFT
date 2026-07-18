"""Sample-data generator.

Produces a deterministic JSONL recording of normalized events (ticks + candles + news) that
can be fed to :class:`validation.replay_harness.ReplayHarness` and to the test-suite. This is
the "existing history" the architecture says to backtest first (§8: phase 1).

Usage
-----
    python -m data_ingestion.generate_sample --out data/sample.jsonl --steps 2000
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from core.config import Clock, Config
from core.events import NormalizedEvent

from .market_feeds.feeds import SimMarketFeed
from .news_feeds.feeds import SimNewsFeed
from .normalizer import normalize_news

log = logging.getLogger("data.generate_sample")


def generate(cfg: Config, steps: int, seed: int = 42, trend_strength: float = 0.0,
             mean_reversion: float = 0.01) -> list:
    """Run the sim feeds headless and capture emitted events as a list.

    ``trend_strength`` injects persistent, detectable trends (fair microstructure) so the
    validation gate has a legitimate edge to discover. 0.0 = pure GBM (no edge).
    ``mean_reversion`` is the OU pull toward a slowly-drifting anchor (keeps prices bounded).
    """
    from core.bus import MessageBus
    from core.metrics import Metrics

    if trend_strength:
        cfg.set("data.market.sim.trend_strength", trend_strength)
    if mean_reversion is not None:
        cfg.set("data.market.sim.mean_reversion", mean_reversion)
    captured: list = []
    clock = Clock()
    metrics = Metrics()
    sink = MessageBus(backend="memory")
    # Record ONLY raw inputs (ticks / candles / order book / news). The pipeline's own
    # decision outputs (composite, proposals, vetoes, fills) must NOT be persisted — replay
    # re-derives them from the raw inputs. Persisting them would double-feed the pipeline
    # and cause a proposal/veto storm at replay time.
    _RAW_TYPES = {
        "tick", "candle", "order_book", "news", "control",
    }
    sink.subscribe(None, lambda ev: captured.append(ev) if ev.type in _RAW_TYPES else None)

    market = SimMarketFeed(sink, cfg, clock, metrics, seed=seed)
    market.start()
    news = SimNewsFeed(sink, cfg, clock, metrics, seed=seed + 1)
    news.start()

    for _ in range(steps):
        clock.advance(float(cfg.get("data.market.sim.interval_s", 1.0)))
        market.run_once()
    for i in range(steps // 50):
        clock.advance(1.0)
        sink.publish(normalize_news("sim.news", title=f"EURUSD moves on macro data {i}",
                                    summary="simulated headline", time=clock.now()))
    return captured


def write_jsonl(events, path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev.to_dict()) + "\n")
    return len(events)


def load_jsonl(path) -> list:
    path = Path(path)
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(NormalizedEvent.from_dict(json.loads(line)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sample.jsonl")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trendy", dest="trendy", type=float, default=0.0,
                    help="Inject persistent trends (e.g. 0.003). 0 = pure GBM (no edge).")
    ap.add_argument("--reversion", dest="reversion", type=float, default=0.01,
                    help="OU mean-reversion speed toward drifting anchor (lower = stronger trends).")
    args = ap.parse_args()
    cfg = Config.load(args.config) if args.config else Config.load()
    events = generate(cfg, args.steps, seed=args.seed, trend_strength=args.trendy,
                     mean_reversion=args.reversion)
    n = write_jsonl(events, args.out)
    log.info("Wrote %d events to %s", n, args.out)


if __name__ == "__main__":
    main()
