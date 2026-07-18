"""Re-export market feed implementations (Layer 1 market_feeds package)."""

from .feeds import (
    CryptoMarketFeed,
    MarketFeed,
    MT5MarketFeed,
    SimMarketFeed,
    build_market_feed,
)

__all__ = [
    "MarketFeed",
    "SimMarketFeed",
    "MT5MarketFeed",
    "CryptoMarketFeed",
    "build_market_feed",
]
