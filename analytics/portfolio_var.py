r"""Portfolio VaR (Layer 6 -- S7.2).

Parametric (variance-covariance / variance) Value-at-Risk on the live
portfolio curve, consumed by the risk stack and downstream dashboards:

    VaR(confidence, horizon) = - ( mean(r) * h + z(confidence) * std(r) * sqrt(h) )

where :math:`r` is the **weighted portfolio daily return** and :math:`z` the
normal quantile (1.645 @ 95%, 2.326 @ 99%).

Pipeline
--------
1. Subscribe to ``CANDLE`` on every symbol in the portfolio.
2. Maintain a rolling position state (symbol :subscripts: `:weight:`).
3. Build the continuously-compounded **portfolio return** stream
   :math:`r_t = \sum_i w_i\,r_{i,t}`.
4. Keep a rolling window of :math:`r_t` (default 252 trading days).
5. Emit ``PORTFOLIO_VaR`` every ``report_s`` with::

       VaR95  = - (mean(r)*h + 1.645*std(r)*sqrt(h))
       VaR99  = - (mean(r)*h + 2.326*std(r)*sqrt(h))
       vol    = std(r)
       sharpe = mean(r) / (std(r) + eps)

Design note
-----------
VaR is a **long-run** statistic: a trailing window of 252 days keeps it
statistically meaningful and decoupled from one-off market moves.  It never
blocks a single order (the veto service is the hard-cap authority) — it is an
**observability / governance** layer that feeds risk dashboards and the
strategy router's position-sizing loop.
"""

from __future__ import annotations

import logging
from collections import deque
from math import sqrt

import numpy as np

from core.component import Component
from core.events import EventType, make_event

log = logging.getLogger("risk.portfolio_var")

_Z95 = 1.6448536269514722
_Z99 = 2.3263478740408408


class PortfolioVaR(Component):
    """Compute parametric VaR on the live portfolio curve."""

    def __init__(
        self,
        bus,
        cfg,
        clock,
        metrics,
        *,
        window: int = 252,          # trading days
        confidence: float = 0.95,   # 0.95 | 0.99
        report_s: float = 600.0,    # how often to emit PORTFOLIO_VaR
        weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__(bus, cfg, clock, metrics, name="risk.portfolio_var")
        self.window = max(10, int(window))
        self.confidence = float(confidence)
        self.report_s = float(report_s)
        self._weights = dict(weights or {})
        # symbol -> deque of closed prices (for return computation)
        self._close_hist: dict[str, deque[float]] = {}
        # rolling portfolio returns
        self._returns: deque[float] = deque(maxlen=self.window)
        self._prev_close: dict[str, float] = {}
        self._last_report = 0.0

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        self._running = True
        self.bus.subscribe(EventType.CANDLE, self._on_candle)
        log.info(
            "PortfolioVaR online (window=%d days, conf=%.0f%%, report_s=%.0f)",
            self.window, self.confidence * 100, self.report_s,
        )

    # ------------------------------------------------------------------ #
    def _on_candle(self, ev) -> None:
        p = ev.payload
        sym = p.get("symbol")
        if sym not in self._weights:
            # ignore symbols outside the portfolio mask
            return
        close = float(p["close"])
        dq = self._close_hist.setdefault(sym, deque(maxlen=self.window))
        dq.append(close)
        if len(dq) < 2:
            self._prev_close[sym] = close
            return
        prev = self._prev_close.get(sym)
        if prev is None or prev == 0:
            self._prev_close[sym] = close
            return
        r = float((close - prev) / prev)          # simple daily return
        self._prev_close[sym] = close
        w = self._weights[sym]
        self._returns.append(w * r)
        self.metrics.inc("vaR_returns", symbol=sym, weight=w)

    # ------------------------------------------------------------------ #
    def tick(self) -> None:
        """Public heart-beat (called by the runner each report_s)."""
        if self._running and self.clock.now() - self._last_report >= self.report_s \
                and len(self._returns) >= 20:
            self._emit()
            self._last_report = self.clock.now()

    # ------------------------------------------------------------------ #
    def _emit(self) -> None:
        r = np.asarray(list(self._returns), dtype=float)
        n = float(len(r))
        mean = float(np.mean(r))
        std = float(np.std(r, ddof=1)) if n > 1 else 0.0
        vol = std
        sharpe = (mean / (std + 1e-12)) * sqrt(252)
        h = 1.0  # daily horizon
        var95 = -(mean * h + _Z95 * std * sqrt(h))
        var99 = -(mean * h + _Z99 * std * sqrt(h))

        self.emit(
            make_event(
                EventType.PORTFOLIO_VaR, "risk.portfolio_var",
                payload={
                    "confidence": self.confidence,
                    "horizon_days": h,
                    "mean_return": mean,
                    "volatility": vol,
                    "sharpe": sharpe,
                    "var95": var95,
                    "var99": var99,
                    "n": n,
                    "window": self.window,
                },
                timestamp=self.clock.now(),
            ),
        )
        self.metrics.set("portfolio_var95", var95)
        self.metrics.set("portfolio_var99", var99)
        self.metrics.set("portfolio_vol", vol)
        self.metrics.set("portfolio_sharpe", sharpe)
        log.debug("PORTFOLIO_VaR report: VaR95=%.4f VaR99=%.4f vol=%.4f",
                  var95, var99, vol)

    # ------------------------------------------------------------------ #
    def summary(self) -> dict:
        if not self._returns:
            return {"portfolio_var": None, "n": 0}
        r = np.asarray(list(self._returns), dtype=float)
        return {
            "portfolio_var": float(-np.mean(r) - _Z95 * float(np.std(r, ddof=1))),
            "n": float(len(r)),
        }
