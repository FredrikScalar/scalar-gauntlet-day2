"""Monitor live performance against an intraday strategy training baseline.

Usage from the repository root:
    python starter/monitor_live.py --strategy windfall2
    python starter/monitor_live.py --strategy windfall2 --live-blotter path/to/live.csv

The supplied historical sample is treated as training/backtest only. A live
blotter must be supplied before the script produces a live comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from repricer import load_blotter, load_market
from report_v1 import _daily_pnl, _reprice, _t_statistics

FEE_EUR_MWH = 0.12
MIN_WATCH_DAYS = 20
FAILURE_DAYS = 30
MIN_BASELINE_DAYS = 30


def _parse_dev_end(value: Any) -> pd.Timestamp | None:
    if not value or not isinstance(value, str) or "to" not in value:
        return None
    return pd.Timestamp(value.split("to", 1)[1].strip(), tz="UTC")


def _windowed_daily(trades: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.Series:
    daily = _daily_pnl(trades, "net_executable_price", None)
    if start is not None:
        daily = daily[daily.index >= start]
    if end is not None:
        daily = daily[daily.index <= end]
    return daily


def _trade_metrics(trades: pd.DataFrame, daily: pd.Series) -> dict[str, Any]:
    if trades.empty or daily.empty:
        return {"days": int(len(daily)), "mean_daily_pnl_eur": 0.0, "pnl_eur": 0.0, "profit_per_mwh": 0.0, "hit_rate": 0.0, "tstat": 0.0, "hac_tstat": 0.0, "max_drawdown_eur": 0.0}
    net_pnl = trades["signed_exec_pnl"] - FEE_EUR_MWH * trades["qty_mw"]
    volume = float(trades["qty_mw"].sum())
    tstats = _t_statistics(daily)
    curve = daily.cumsum()
    return {
        "days": int(len(daily)),
        "active_days": int((daily != 0).sum()),
        "mean_daily_pnl_eur": round(float(daily.mean()), 3),
        "pnl_eur": round(float(daily.sum()), 2),
        "volume_mwh": round(volume, 2),
        "profit_per_mwh": round(float(net_pnl.sum() / volume), 4) if volume else 0.0,
        "hit_rate": round(float((daily > 0).mean()), 3),
        "tstat": tstats["tstat"],
        "hac_tstat": tstats["hac_tstat"],
        "max_drawdown_eur": round(float((curve - curve.cummax()).min()), 2),
    }


def _baseline_ratio(live: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float | None]:
    def ratio(key: str) -> float | None:
        denominator = float(baseline.get(key, 0) or 0)
        return round(float(live.get(key, 0)) / denominator, 3) if denominator else None

    return {
        "mean_daily_pnl_ratio": ratio("mean_daily_pnl_eur"),
        "profit_per_mwh_ratio": ratio("profit_per_mwh"),
        "hit_rate_difference": round(float(live.get("hit_rate", 0)) - float(baseline.get("hit_rate", 0)), 3),
    }


def _status(live: dict[str, Any], baseline: dict[str, Any]) -> tuple[str, str]:
    if int(baseline.get("days", 0)) < MIN_BASELINE_DAYS:
        return "NO_BASELINE", f"Need at least {MIN_BASELINE_DAYS} training delivery days for a valid comparison."
    days = int(live.get("days", 0))
    if days < MIN_WATCH_DAYS:
        return "INSUFFICIENT_DATA", f"Monitor at least {MIN_WATCH_DAYS} delivery days before judging live performance."
    baseline_mean = float(baseline.get("mean_daily_pnl_eur", 0))
    live_mean = float(live.get("mean_daily_pnl_eur", 0))
    hac_tstat = float(live.get("hac_tstat", 0))
    if days >= FAILURE_DAYS and live_mean <= 0 and hac_tstat <= -1.645:
        return "FAIL", f"At least {FAILURE_DAYS} days show negative mean P&L with a negative 5% HAC test."
    if days >= MIN_WATCH_DAYS and (live_mean < 0.5 * baseline_mean or hac_tstat < 1.645):
        return "WATCH", "Live performance is below half the training mean or lacks positive 10% HAC evidence."
    return "HEALTHY", "Live performance remains consistent with the training baseline under current evidence."


def analyze(strategy: str, registry: dict[str, dict], market: dict[str, pd.DataFrame], live_blotter_path: str | Path | None = None) -> dict[str, Any]:
    if strategy not in registry:
        raise KeyError(f"Unknown strategy: {strategy}")
    entry = registry[strategy]
    training_blotter_path = Path(__file__).resolve().parents[1] / "blotters" / Path(entry["blotter"]).name
    training_trades = _reprice(load_blotter(training_blotter_path), market)
    dev_end = _parse_dev_end(entry.get("dev_window"))
    backtest = entry.get("backtest_window", {})
    backtest_start = pd.Timestamp(backtest["start"], tz="UTC") if backtest.get("start") else None
    backtest_end = pd.Timestamp(backtest["end"], tz="UTC") if backtest.get("end") else None
    baseline = _windowed_daily(training_trades, backtest_start, dev_end) if dev_end is not None else pd.Series(dtype=float)
    if not live_blotter_path:
        raise ValueError("No live test window exists yet; provide --live-blotter for live monitoring.")
    live_trades = _reprice(load_blotter(live_blotter_path), market)
    live = _windowed_daily(live_trades, None, None)
    baseline_metrics = _trade_metrics(training_trades[training_trades["product_delivery"].dt.normalize().isin(baseline.index)], baseline)
    live_metrics = _trade_metrics(live_trades[live_trades["product_delivery"].dt.normalize().isin(live.index)], live)
    status, reason = _status(live_metrics, baseline_metrics)
    return {
        "strategy": strategy,
        "status": status,
        "reason": reason,
        "assumptions": {"fee_eur_mwh_per_trade": FEE_EUR_MWH, "min_watch_days": MIN_WATCH_DAYS, "failure_days": FAILURE_DAYS, "baseline_window": "declared development/backtest period", "live_window": "post-development period or supplied live blotter"},
        "baseline": baseline_metrics,
        "live_or_test": live_metrics,
        "comparison": _baseline_ratio(live_metrics, baseline_metrics),
        "baseline_available": bool(dev_end is not None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare live/test intraday strategy performance with its training baseline.")
    parser.add_argument("--strategy", required=True, choices=sorted(path.stem for path in (Path(__file__).resolve().parents[1] / "registry").glob("*.yaml")))
    parser.add_argument("--live-blotter", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    registry = {path.stem: yaml.safe_load(path.read_text()) for path in (root / "registry").glob("*.yaml")}
    result = analyze(args.strategy, registry, load_market(root / "data"), args.live_blotter)
    print(json.dumps(result, indent=2, default=str))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
