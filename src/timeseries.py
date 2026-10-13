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
        in the frame's attrs ("ranges": {Test_ID: (start, end)}) for zero-filling later.
        """
        columns = ["Test_ID", "date", "net", "trades", "min_eq"]
        empty = pd.DataFrame(columns=columns)
        empty.attrs["ranges"] = {}
        if df is None or df.empty or "PnL_Series" not in df.columns:
            return empty

        value_col = MODE_COLUMN.get(mode, 1)
        ranges = {}
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
    def month_heatmap(long_df: pd.DataFrame, metric: str = "mean_profit") -> Optional[pd.DataFrame]:
        """Year x month matrix of the average monthly profit or the share of losing tests (%)."""
        per_test = TimeSeriesAnalyzer.per_test_periods(long_df, "M")
        if per_test.empty:
            return None

        per_test["Year"] = per_test["period_start"].dt.year
        per_test["Month"] = per_test["period_start"].dt.month
        per_test["is_loss"] = per_test["net"] < 0
        grouped = per_test.groupby(["Year", "Month"])
        values = grouped["net"].mean() if metric == "mean_profit" else grouped["is_loss"].mean() * 100

        matrix = values.unstack("Month").reindex(columns=range(1, 13))
        matrix.columns = MONTH_LABELS
        matrix.index = matrix.index.astype(str)
        return matrix
