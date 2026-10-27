import json
import numpy as np
import pandas as pd
from typing import Optional

# Column index of each trading mode inside a stored daily row:
# [day_idx, net_all, net_long, net_short, trades, min_eq]
MODE_COLUMN = {"all": 1, "long": 2, "short": 3}

# Pandas period codes: weeks run Monday-Sunday, months and years are calendar based.
PERIOD_FREQ = {"W": "W-SUN", "M": "M", "Y": "Y"}

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


class TimeSeriesAnalyzer:
    """
    Expands the compact P/L series stored by the scanner and aggregates them
    across the filtered tests (weekly / monthly / yearly bands and heatmaps).
    """

    @staticmethod
    def expand(df: pd.DataFrame, mode: str = "all") -> pd.DataFrame:
        """
        Turns the PnL_Series JSON column of the given rows into a long DataFrame of the
        active days: Test_ID, date, net, trades, min_eq. Each test's testing period is kept
        in the frame's attrs ("ranges": {Test_ID: (start, end)}) for zero-filling later,
        and the starting capitals in attrs["caps"] ({Test_ID: capital}).
        """
        columns = ["Test_ID", "date", "net", "trades", "min_eq"]
        empty = pd.DataFrame(columns=columns)
        empty.attrs["ranges"] = {}
        empty.attrs["caps"] = {}
        if df is None or df.empty or "PnL_Series" not in df.columns:
            return empty

        value_col = MODE_COLUMN.get(mode, 1)
        # Starting capital: the scanner's "Starting Capital" (from the .optres settings or the report).
        capitals = {}
        if "Starting Capital" in df.columns:
            capitals = {t: float(c) for t, c in zip(df["Test_ID"], df["Starting Capital"]) if pd.notna(c)}
        ranges, caps = {}, {}
        ids, dates, nets, trades, min_eqs = [], [], [], [], []

        for test_id, raw in zip(df["Test_ID"].to_numpy(), df["PnL_Series"].to_numpy()):
            if not isinstance(raw, str) or not raw:
                continue
            try:
                payload = json.loads(raw)
            except ValueError:
                continue

            start = np.datetime64(payload["start"], "D")
            ranges[test_id] = (pd.Timestamp(start), pd.Timestamp(np.datetime64(payload.get("end", payload["start"]), "D")))
            caps[test_id] = capitals.get(test_id, payload.get("cap"))
            rows = payload.get("d") or []
            if not rows:
                continue

            arr = np.array([[r[0], r[value_col], r[4], np.nan if r[5] is None else r[5]] for r in rows], dtype=float)
            ids.append(np.full(len(arr), test_id))
            dates.append(start + arr[:, 0].astype(int) * np.timedelta64(1, "D"))
            nets.append(arr[:, 1])
            trades.append(arr[:, 2].astype(int))
            min_eqs.append(arr[:, 3])

        if not ranges:
            return empty

        if ids:
            long_df = pd.DataFrame(
                {
                    "Test_ID": np.concatenate(ids),
                    "date": pd.to_datetime(np.concatenate(dates)),
                    "net": np.concatenate(nets),
                    "trades": np.concatenate(trades),
                    "min_eq": np.concatenate(min_eqs),
                }
            )
        else:
            long_df = pd.DataFrame(columns=columns)
        long_df.attrs["ranges"] = ranges
        long_df.attrs["caps"] = caps
        return long_df

    @staticmethod
    def per_test_periods(long_df: pd.DataFrame, freq: str = "W", cumulative: bool = False) -> pd.DataFrame:
        """
        Sums each test's P/L into the requested period (W / M / Y). Periods inside a test's
        own testing period without any trade get 0, periods outside of it are left out.
        """
        ranges = long_df.attrs.get("ranges", {})
        if not ranges:
            return pd.DataFrame(columns=["Test_ID", "period_start", "net"])

        pfreq = PERIOD_FREQ.get(freq, "W-SUN")
        if long_df.empty:
            sums = pd.Series(dtype=float)
        else:
            periods = long_df["date"].dt.to_period(pfreq)
            sums = long_df.groupby([long_df["Test_ID"], periods])["net"].sum()

        # Every (test, period) pair inside the test's own range, built from period ordinals.
        test_ids = np.array(list(ranges.keys()))
        first = pd.PeriodIndex([s for s, _ in ranges.values()], freq=pfreq).asi8
        last = pd.PeriodIndex([e for _, e in ranges.values()], freq=pfreq).asi8
        counts = last - first + 1
        offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        full_index = pd.MultiIndex.from_arrays(
            [np.repeat(test_ids, counts), pd.PeriodIndex.from_ordinals(np.repeat(first, counts) + offsets, freq=pfreq)],
            names=["Test_ID", "period"],
        )
        if len(sums):
            sums.index = sums.index.set_names(["Test_ID", "period"])
        per_test = sums.reindex(full_index, fill_value=0.0).rename("net").reset_index()

        if cumulative:
            per_test["net"] = per_test.groupby("Test_ID")["net"].cumsum()
        per_test["period_start"] = per_test["period"].dt.start_time
        return per_test[["Test_ID", "period_start", "net"]]

    @staticmethod
    def aggregate(long_df: pd.DataFrame, freq: str = "W", cumulative: bool = False) -> pd.DataFrame:
        """
        Per period statistics across the tests: mean, std, min, max, median,
        loss_ratio (% of tests with a negative period) and n (tests covering the period).
        """
        per_test = TimeSeriesAnalyzer.per_test_periods(long_df, freq, cumulative)
        if per_test.empty:
            return pd.DataFrame(columns=["period_start", "mean", "std", "min", "max", "median", "loss_ratio", "n"])

        per_test["is_loss"] = per_test["net"] < 0
        grouped = per_test.groupby("period_start")
        stats = grouped["net"].agg(["mean", "std", "min", "max", "median", "count"]).rename(columns={"count": "n"})
        stats["loss_ratio"] = grouped["is_loss"].mean() * 100
        stats["std"] = stats["std"].fillna(0.0)
        return stats.reset_index()

    @staticmethod
    def seasonality_heatmap(long_df: pd.DataFrame, metric: str = "mean_profit", granularity: str = "M") -> Optional[pd.DataFrame]:
        """
        Year x month (granularity "M") or year x ISO week ("W") matrix of the average
        period profit or the share of losing tests (%).
        """
        per_test = TimeSeriesAnalyzer.per_test_periods(long_df, granularity)
        if per_test.empty:
            return None

        if granularity == "W":
            iso = per_test["period_start"].dt.isocalendar()
            per_test["Year"], per_test["Slot"] = iso["year"].astype(int), iso["week"].astype(int)
            slots, labels = range(1, 54), [f"W{w:02d}" for w in range(1, 54)]
        else:
            per_test["Year"], per_test["Slot"] = per_test["period_start"].dt.year, per_test["period_start"].dt.month
            slots, labels = range(1, 13), MONTH_LABELS

        per_test["is_loss"] = per_test["net"] < 0
        grouped = per_test.groupby(["Year", "Slot"])
        values = grouped["net"].mean() if metric == "mean_profit" else grouped["is_loss"].mean() * 100

        matrix = values.unstack("Slot").reindex(columns=slots)
        matrix.columns = labels
        if granularity == "W" and matrix["W53"].isna().all():
            matrix = matrix.drop(columns="W53")
        matrix.index = matrix.index.astype(str)
        return matrix

    @staticmethod
    def month_heatmap(long_df: pd.DataFrame, metric: str = "mean_profit") -> Optional[pd.DataFrame]:
        """Year x month matrix of the average monthly profit or the share of losing tests (%)."""
        return TimeSeriesAnalyzer.seasonality_heatmap(long_df, metric, "M")

    # ==========================================
    # BALANCE CURVES, DRAWDOWNS & TREND METRICS
    # ==========================================
    @staticmethod
    def weekly_matrices(long_df: pd.DataFrame) -> dict:
        """
        Week x test matrices on a common weekly grid: "net" (weekly P/L, 0 for idle weeks,
        NaN outside a test's own testing period), "min_eq" (lowest equity of the week, NaN
        if unknown) and "caps" (starting capital per test).
        """
        per_test = TimeSeriesAnalyzer.per_test_periods(long_df, "W")
        if per_test.empty:
            return {"net": pd.DataFrame(), "min_eq": pd.DataFrame(), "caps": pd.Series(dtype=float)}

        net = per_test.pivot(index="period_start", columns="Test_ID", values="net").sort_index()
        if long_df.empty:
            min_eq = pd.DataFrame(np.nan, index=net.index, columns=net.columns)
        else:
            weeks = long_df["date"].dt.to_period(PERIOD_FREQ["W"]).dt.start_time
            min_eq = (
                long_df.assign(period_start=weeks)
                .groupby(["period_start", "Test_ID"])["min_eq"].min()
                .unstack("Test_ID")
                .reindex(index=net.index, columns=net.columns)
            )
        caps = pd.Series(long_df.attrs.get("caps", {}), dtype=float).reindex(net.columns)
        # Balance curves, ROI and drawdowns need the real starting capital; tests without one are left out.
        known = caps.notna().to_numpy()
        return {"net": net.loc[:, known], "min_eq": min_eq.loc[:, known], "caps": caps[known]}

    @staticmethod
    def equity_curves(mats: dict, roi: bool = False) -> pd.DataFrame:
        """Weekly closing balance per test (absolute, or ROI % relative to the starting capital)."""
        net = mats["net"]
        if net.empty:
            return net
        balance = net.cumsum().where(net.notna()) + mats["caps"]
        return (balance / mats["caps"] - 1.0) * 100.0 if roi else balance

    @staticmethod
    def drawdown_curves(mats: dict, use_equity: bool = True, absolute: bool = False) -> pd.DataFrame:
        """
        Weekly drawdown (%, <= 0; or the amount in account currency with absolute) from the
        running balance peak. With use_equity the week's lowest equity is used as well, so
        intraweek floating losses are captured.
        """
        net = mats["net"]
        if net.empty:
            return net
        balance = TimeSeriesAnalyzer.equity_curves(mats)
        running_peak = balance.cummax().clip(lower=mats["caps"], axis=1)
        drawdown = balance - running_peak
        if not absolute:
            drawdown = drawdown / running_peak * 100.0
        if use_equity:
            # The week's equity low is measured against the peak reached before that week.
            previous_peak = running_peak.shift(1).fillna(running_peak).clip(lower=mats["caps"], axis=1)
            intraweek = mats["min_eq"] - previous_peak
            if not absolute:
                intraweek = intraweek / previous_peak * 100.0
            drawdown = np.fmin(drawdown, intraweek)
        drawdown = drawdown.clip(upper=0.0)
        return drawdown.where(net.notna())

    @staticmethod
    def trend_metrics(mats: dict, use_equity: bool = True) -> pd.DataFrame:
        """
        Per test: Trend R² (signed by the slope of the balance curve), Ulcer Index,
        Max Underwater (w), Weekly Vol % and Weekly Sharpe.
        """
        columns = ["Test_ID", "Trend R²", "Ulcer Index", "Max Underwater (w)", "Weekly Vol %", "Weekly Sharpe"]
        net = mats["net"]
        if net.empty:
            return pd.DataFrame(columns=columns)

        balance = TimeSeriesAnalyzer.equity_curves(mats).to_numpy()
        valid = ~np.isnan(balance)
        x = np.where(valid, np.arange(len(balance))[:, None], np.nan)

        x_mean = np.nanmean(x, axis=0)
        y_mean = np.nanmean(balance, axis=0)
        cov = np.nansum((x - x_mean) * (balance - y_mean), axis=0)
        var_x = np.nansum((x - x_mean) ** 2, axis=0)
        var_y = np.nansum((balance - y_mean) ** 2, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r2 = np.where((var_x > 0) & (var_y > 0), cov ** 2 / (var_x * var_y), 0.0)
        trend_r2 = r2 * np.sign(cov)

        drawdown = TimeSeriesAnalyzer.drawdown_curves(mats, use_equity).to_numpy()
        ulcer = np.sqrt(np.nanmean(drawdown ** 2, axis=0))

        # Longest run of weeks spent below the previous balance peak.
        caps = mats["caps"].to_numpy()
        running_peak = np.fmax(np.fmax.accumulate(np.nan_to_num(balance, nan=-np.inf), axis=0), caps)
        previous_peak = np.vstack([caps, running_peak[:-1]])
        underwater = valid & (balance < previous_peak - 1e-9)
        run = np.zeros(underwater.shape[1])
        longest = np.zeros(underwater.shape[1])
        for row in underwater:
            run = np.where(row, run + 1, 0)
            longest = np.maximum(longest, run)

        weekly_roi = (net / mats["caps"] * 100.0).to_numpy()
        vol = np.nanstd(weekly_roi, axis=0, ddof=1) if len(weekly_roi) > 1 else np.zeros(net.shape[1])
        with np.errstate(divide="ignore", invalid="ignore"):
            sharpe = np.where(vol > 1e-9, np.nanmean(weekly_roi, axis=0) / vol, 0.0)

        return pd.DataFrame(
            {
                "Test_ID": net.columns.to_numpy(),
                "Trend R²": np.round(trend_r2, 4),
                "Ulcer Index": np.round(ulcer, 4),
                "Max Underwater (w)": longest.astype(int),
                "Weekly Vol %": np.round(np.nan_to_num(vol), 4),
                "Weekly Sharpe": np.round(sharpe, 4),
            }
        )

    @staticmethod
    def band_stats(matrix: pd.DataFrame) -> pd.DataFrame:
        """Per week mean, std, min, max and n across the test columns of a curve matrix."""
        if matrix.empty:
            return pd.DataFrame(columns=["period_start", "mean", "std", "min", "max", "n"])
        stats = pd.DataFrame(
            {
                "mean": matrix.mean(axis=1),
                "std": matrix.std(axis=1).fillna(0.0),
                "min": matrix.min(axis=1),
                "max": matrix.max(axis=1),
                "n": matrix.notna().sum(axis=1),
            }
        )
        stats.index.name = "period_start"
        return stats[stats["n"] > 0].reset_index()

    # ==========================================
    # SINGLE TEST BREAKDOWN
    # ==========================================
    @staticmethod
    def test_periods(raw: str, capital: Optional[float], mode: str = "all", freq: str = "M", cumulative: bool = False) -> pd.DataFrame:
        """
        One test's P/L, trade count, win rate and drawdown per period (W / M / Y) from its
        PnL_Series. Drawdown is measured from the running balance peak (with the intraday
        equity low in "all" mode) and needs the starting capital; without it the column is NaN.
        With cumulative, P/L and counts are running totals; the drawdown always stays per period.
        """
        columns = ["period_start", "net", "trades", "wins", "win_rate", "max_dd", "max_dd_amount", "balance"]
        if not isinstance(raw, str) or not raw:
            return pd.DataFrame(columns=columns)
        payload = json.loads(raw)
        start = pd.Timestamp(payload["start"])
        days = pd.date_range(start, pd.Timestamp(payload.get("end", payload["start"])), freq="D")
        daily = pd.DataFrame(0.0, index=days, columns=["net", "trades", "wins"])
        daily["min_eq"] = np.nan

        value_col = MODE_COLUMN.get(mode, 1)
        for r in payload.get("d") or []:
            day = start + pd.Timedelta(days=int(r[0]))
            if day not in daily.index:
                continue
            has_counts = len(r) >= 9
            trades_all, trades_long = r[4], (r[6] if has_counts else np.nan)
            wins_all, wins_long = (r[7], r[8]) if has_counts else (np.nan, np.nan)
            trades = {"all": trades_all, "long": trades_long, "short": trades_all - trades_long}[mode]
            wins = {"all": wins_all, "long": wins_long, "short": wins_all - wins_long}[mode]
            daily.loc[day, ["net", "trades", "wins"]] = [r[value_col], trades, wins]
            if r[5] is not None:
                daily.loc[day, "min_eq"] = r[5]

        balance = pd.Series(np.nan, index=days)
        drawdown = pd.Series(np.nan, index=days)
        drawdown_amount = pd.Series(np.nan, index=days)
        if capital is not None and pd.notna(capital):
            balance = capital + daily["net"].cumsum()
            peak = balance.cummax().clip(lower=capital)
            drawdown_amount = balance - peak
            drawdown = drawdown_amount / peak * 100.0
            if mode == "all":
                previous_peak = peak.shift(1).fillna(capital)
                intraday = daily["min_eq"] - previous_peak
                drawdown_amount = np.fmin(drawdown_amount, intraday)
                drawdown = np.fmin(drawdown, intraday / previous_peak * 100.0)
            drawdown = drawdown.clip(upper=0.0)
            drawdown_amount = drawdown_amount.clip(upper=0.0)

        pfreq = PERIOD_FREQ.get(freq, "M")
        periods = days.to_period(pfreq)
        grouped = pd.DataFrame(
            {"net": daily["net"], "trades": daily["trades"], "wins": daily["wins"], "dd": drawdown,
             "dd_amount": drawdown_amount, "balance": balance}
        ).groupby(periods)
        out = pd.DataFrame({
            "net": grouped["net"].sum(),
            "trades": grouped["trades"].sum(min_count=1),
            "wins": grouped["wins"].sum(min_count=1),
            "max_dd": grouped["dd"].min(),
            "max_dd_amount": grouped["dd_amount"].min(),
            "balance": grouped["balance"].last(),
        })
        if cumulative:
            out["net"] = out["net"].cumsum()
            out["trades"] = out["trades"].cumsum()
            out["wins"] = out["wins"].cumsum()
        with np.errstate(divide="ignore", invalid="ignore"):
            out["win_rate"] = np.where(out["trades"] > 0, out["wins"] / out["trades"] * 100.0, np.nan)
        out["period_start"] = out.index.start_time
        return out.reset_index(drop=True)[columns]
