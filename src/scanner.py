import os
import json
import re
import zipfile
import numpy as np
import pandas as pd
import gc
import streamlit as st
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Union

# Columns a cache file must contain to be reused; older caches trigger a rescan.
REQUIRED_CACHE_COLUMNS = ("Total Trades_all", "PnL_Series", "Avg Bars Held_all", "Hold_Hist", "Equity Drawdown Amount")
TIMESERIES_FORMAT_VERSION = 2
DAY_MS = 24 * 3600 * 1000

# Per-trade holding time histogram: log10(hours) bins of HOLD_HIST_WIDTH decades from HOLD_HIST_START
# (0.0001 h = 0.36 s) to 10 000 h; shorter / longer positions fall into the first / last bin.
HOLD_HIST_START = -4.0
HOLD_HIST_WIDTH = 0.2
HOLD_HIST_BINS = 40

DURATION_METRICS = [
    "Avg Bars Held", "Median Bars Held", "P90 Bars Held", "Max Bars Held",
    "Outlier Trades %", "Skew Ratio", "Avg Hold Time (h)",
]


class CTraderScanner:
    """
    Dual-mode (Local SSD & In-Memory Cloud) cTrader optimization scanner.
    Lightning-fast JSON parameter and HTML report extraction with strict memory management.
    """

    def __init__(self, root_directory: str = ".", zip_file=None):
        self.env_mode = os.environ.get("ENV_MODE", "local").lower()
        self.zip_file = zip_file

        clean_path = str(root_directory).strip().strip('"').strip("'").replace("\\ ", " ")
        self.root_dir = Path(clean_path).expanduser().resolve()
        self.cache_parquet = self.root_dir / "optimization_cache.parquet"
        self._optres_capital_cache: Dict[str, Optional[float]] = {}
        self.cache_csv = self.root_dir / "optimization_cache.csv"

    # ==========================================
    # COMMON INTERFACE (ROUTER)
    # ==========================================
    @staticmethod
    def is_cache_compatible(df: Optional[pd.DataFrame]) -> bool:
        """True if a cached DataFrame has every column the current version needs."""
        if df is None or df.empty or not all(c in df.columns for c in REQUIRED_CACHE_COLUMNS):
            return False
        sample = df["PnL_Series"].dropna()
        if sample.empty:
            return True
        try:
            return json.loads(sample.iloc[0]).get("v", 1) >= TIMESERIES_FORMAT_VERSION
        except (ValueError, TypeError):
            return False

    def get_data(self, force_rescan: bool = False, progress_callback=None) -> Optional[pd.DataFrame]:
        """Manages caching in local mode and starts scanning."""
        if self.env_mode == "cloud":
            return self.scan(progress_callback)

        if not force_rescan:
            if self.cache_parquet.exists():
                try:
                    df = pd.read_parquet(self.cache_parquet)
                    if self.is_cache_compatible(df):
                        return df
                except Exception:
                    pass
            if self.cache_csv.exists():
                try:
                    df = pd.read_csv(self.cache_csv)
                    if self.is_cache_compatible(df):
                        return df
                except Exception:
                    pass

        if force_rescan:
            if self.cache_parquet.exists():
                self.cache_parquet.unlink()
            if self.cache_csv.exists():
                self.cache_csv.unlink()

        return self.scan(progress_callback)
    
    def scan(self, progress_callback=None) -> Optional[pd.DataFrame]:
        """The main Facade method. It decides which data processing engine to start."""
        if self.env_mode == "cloud":
            if not self.zip_file:
                return None
            return self._scan_zip_in_memory(progress_callback)
        else:
            return self._scan_local(progress_callback)
        
    # ==========================================
    # 1. CORE PARSERS
    # ==========================================
    @staticmethod
    def _optres_capital(raw: Union[str, bytes]) -> Optional[float]:
        """Starting capital from a cTrader optimization results file (.optres, JSON)."""
        try:
            text = raw.decode("utf-8-sig", errors="ignore") if isinstance(raw, bytes) else raw.lstrip("\ufeff")
            value = (json.loads(text).get("backtestingSettings") or {}).get("startingCapital")
            return float(value) if value is not None else None
        except (ValueError, TypeError, AttributeError):
            return None

    def _local_optres_capital(self, folder: Path) -> Optional[float]:
        """
        Looks for "<folder name>.optres" next to each parent of the test folder (cTrader saves
        the optimization as <Name>.optres beside the <Name>/ results folder), up to the root's parent.
        """
        current = folder.parent
        while True:
            candidate = current.parent / f"{current.name}.optres"
            key = str(candidate)
            if key not in self._optres_capital_cache:
                self._optres_capital_cache[key] = (
                    self._optres_capital(candidate.read_bytes()) if candidate.is_file() else None
                )
            if self._optres_capital_cache[key] is not None:
                return self._optres_capital_cache[key]
            if current == self.root_dir or current.parent == current:
                return None
            current = current.parent

    def _zip_optres_capital(self, z: zipfile.ZipFile, names: set, folder_path: str) -> Optional[float]:
        """ZIP variant of the .optres lookup (folder paths use forward slashes)."""
        current = os.path.dirname(folder_path)
        while current:
            candidate = f"{current}.optres"
            if candidate not in self._optres_capital_cache:
                self._optres_capital_cache[candidate] = (
                    self._optres_capital(z.read(candidate)) if candidate in names else None
                )
            if self._optres_capital_cache[candidate] is not None:
                return self._optres_capital_cache[candidate]
            current = os.path.dirname(current)
        return None

    @staticmethod
    def _apply_starting_capital(row_data: Dict[str, Any], optres_capital: Optional[float]):
        """The optimization's own setting wins; the report's value is the fallback; unknown stays NaN."""
        report_capital = row_data.pop("_report_capital", None)
        capital = optres_capital if optres_capital is not None else report_capital
        row_data["Starting Capital"] = float(capital) if capital is not None else np.nan

    def _extract_cbotset_logic(self, data: dict) -> Dict[str, Any]:
        """Extracts parameters from an already loaded JSON dictionary."""
        params = {}
        raw_params = data.get("Parameters", {})
        for k, v in raw_params.items():
            params[k] = v 
        return params

    def _extract_timeseries_logic(self, data: dict) -> Optional[str]:
        """
        Builds a compact daily P/L series from the closed trades (history.items) and the
        equity points of an already parsed report. Days are UTC calendar days.
        Format: {"v", "start", "end", "cap", "d": [[day_idx, net_all, net_long, net_short, trades, min_eq,
        trades_long, wins_all, wins_long]]} (short counts = all - long).
        Only active days are stored; weekly / monthly / yearly periods are aggregated from them
        and the inactive periods are zero-filled when the series is expanded.
        """
        trades = (data.get("history") or {}).get("items") or []
        points = (data.get("equity") or {}).get("points") or []
        main = data.get("main") or {}
        period = main.get("testingPeriod") or {}

        stamps = [t.get("closeTime") for t in trades if t.get("closeTime")]
        start_ms = period.get("startDate") or (min(stamps) if stamps else None)
        end_ms = period.get("endDate") or (max(stamps) if stamps else None)
        if start_ms is None or end_ms is None:
            return None

        start_day = int(start_ms // DAY_MS)
        days: Dict[int, list] = {}

        def bucket(ts_ms: float) -> list:
            idx = max(0, int(ts_ms // DAY_MS) - start_day)
            return days.setdefault(idx, [idx, 0.0, 0.0, 0.0, 0, None, 0, 0, 0])

        for t in trades:
            close_time = t.get("closeTime")
            if not close_time:
                continue
            net = float(t.get("net") or 0.0)
            row = bucket(close_time)
            row[1] += net
            is_long = str(t.get("direction", "")).lower() == "buy"
            row[2 if is_long else 3] += net
            row[4] += 1
            row[6] += int(is_long)
            row[7] += int(net > 0)
            row[8] += int(is_long and net > 0)

        for p in points:
            ts, min_eq = p.get("timestamp"), p.get("minEquity")
            if ts is None or min_eq is None:
                continue
            row = bucket(ts)
            row[5] = float(min_eq) if row[5] is None else min(row[5], float(min_eq))

        def iso_day(day: int) -> str:
            return datetime.fromtimestamp(day * DAY_MS / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

        payload = {
            "v": TIMESERIES_FORMAT_VERSION,
            "start": iso_day(start_day),
            "end": iso_day(int(end_ms // DAY_MS)),
            "cap": main.get("startingCapital"),
            "d": [
                [r[0], round(r[1], 2), round(r[2], 2), round(r[3], 2), r[4],
                 None if r[5] is None else round(r[5], 2), r[6], r[7], r[8]]
                for r in (days[i] for i in sorted(days))
            ],
        }
        return json.dumps(payload, separators=(",", ":"))

    @staticmethod
    def _parse_period_minutes(period: Optional[str]) -> Optional[float]:
        """Candle length in minutes for time based timeframes (m5, h1, d1, w1); None for tick/Renko/Range/monthly."""
        match = re.fullmatch(r"([mhdw])(\d+)", str(period or "").strip().lower())
        if not match:
            return None
        unit_minutes = {"m": 1, "h": 60, "d": 1440, "w": 10080}[match.group(1)]
        return float(unit_minutes * int(match.group(2)))

    @staticmethod
    def _hold_histogram(hold_hours: Dict[str, list]) -> Optional[str]:
        """
        Compact log-scale histogram of the per-trade holding times (hours), so a single test's
        trade distribution can be shown later without re-reading the report.
        Format: {"start", "width", "bins": [[bin_idx, count_all, count_long, count_short]]} (non-empty bins only).
        """
        if not hold_hours["all"]:
            return None
        counts = {}
        for col, mode in enumerate(["all", "long", "short"], start=1):
            if not hold_hours[mode]:
                continue
            log_hours = np.log10(np.maximum(np.asarray(hold_hours[mode]), 1e-9))
            idx = np.clip(((log_hours - HOLD_HIST_START) // HOLD_HIST_WIDTH).astype(int), 0, HOLD_HIST_BINS - 1)
            for b, n in zip(*np.unique(idx, return_counts=True)):
                counts.setdefault(int(b), [int(b), 0, 0, 0])[col] = int(n)
        payload = {"start": HOLD_HIST_START, "width": HOLD_HIST_WIDTH, "bins": [counts[b] for b in sorted(counts)]}
        return json.dumps(payload, separators=(",", ":"))

    def _extract_duration_logic(self, data: dict) -> Dict[str, Any]:
        """
        Position holding time per test and direction, in candles of the report's timeframe
        (and in hours): average, median, P90, max, share of outlier positions and skew ratio.
        """
        period = (data.get("main") or {}).get("period")
        period_minutes = self._parse_period_minutes(period)
        metrics: Dict[str, Any] = {"Timeframe": period}

        hold_hours = {"all": [], "long": [], "short": []}
        for t in (data.get("history") or {}).get("items") or []:
            entry, close = t.get("entryTime"), t.get("closeTime")
            if entry is None or close is None:
                continue
            hours = max(0.0, (close - entry) / 3_600_000)
            hold_hours["all"].append(hours)
            hold_hours["long" if str(t.get("direction", "")).lower() == "buy" else "short"].append(hours)

        metrics["Hold_Hist"] = self._hold_histogram(hold_hours)

        for mode, values in hold_hours.items():
            for name in DURATION_METRICS:
                metrics[f"{name}_{mode}"] = np.nan
            if not values:
                continue
            hours = np.asarray(values)
            metrics[f"Avg Hold Time (h)_{mode}"] = round(float(hours.mean()), 4)
            if period_minutes is None:
                continue

            bars = hours * 60.0 / period_minutes
            median, p90 = np.percentile(bars, [50, 90])
            mean = float(bars.mean())
            # Holding times are heavily right-skewed (seconds next to days), so the outlier
            # fence (Q3 + 3 x IQR) is taken on the log scale.
            log_bars = np.log10(np.maximum(bars, 1e-4))
            q1, q3 = np.percentile(log_bars, [25, 75])
            metrics.update({
                f"Avg Bars Held_{mode}": round(mean, 4),
                f"Median Bars Held_{mode}": round(float(median), 4),
                f"P90 Bars Held_{mode}": round(float(p90), 4),
                f"Max Bars Held_{mode}": round(float(bars.max()), 4),
                f"Outlier Trades %_{mode}": round(float((log_bars > q3 + 3 * (q3 - q1)).mean() * 100), 2),
                f"Skew Ratio_{mode}": round(mean / max(float(median), 0.01), 2),
            })
        return metrics

    def _extract_report_logic(self, html: str) -> Dict[str, Any]:
        """Extracts and parses the pure JSON block from the HTML text."""
        row_metrics = {"Equity Drawdown": 0.0, "Equity Drawdown Amount": 0.0, "PnL_Series": None, "Timeframe": None, "Hold_Hist": None, "_report_capital": None}
        for mode in ["all", "long", "short"]:
            for name in DURATION_METRICS:
                row_metrics[f"{name}_{mode}"] = np.nan

        for mode in ["all", "long", "short"]:
            for col in [
                "Net Profit", "Profit Factor", "Swaps", "Total Trades", 
                "Winning Trades", "Losing Trades", "Max Cons. Losses", 
                "Largest Win", "Largest Loss", "Average Trade", "Win Rate"
            ]:
                row_metrics[f"{col}_{mode}"] = 0.0

        try:
            match = re.search(
                r'<script[^>]*id=["\']backtesting-report["\'][^>]*>(.*?)</script>', 
                html, re.DOTALL | re.IGNORECASE
            )
            if not match:
                return row_metrics

            data = json.loads(match.group(1).strip())
            eq = data.get("equity", {})
            ts = data.get("tradeStatistics", {})

            row_metrics["Equity Drawdown"] = float(
                eq.get("maxEquityDrawdownPercent", eq.get("maxBalanceDrawdownPercent", 0.0))
            )
            row_metrics["Equity Drawdown Amount"] = float(
                eq.get("maxEquityDrawdownAbsolute", eq.get("maxBalanceDrawdownAbsolute", 0.0))
            )

            mapping = {
                "netProfit": "Net Profit",
                "profitFactor": "Profit Factor",
                "swaps": "Swaps",
                "totalTrades": "Total Trades",
                "winningTrades": "Winning Trades",
                "losingTrades": "Losing Trades",
                "maxConsecutiveLosingTrades": "Max Cons. Losses",
                "largestWinningTrade": "Largest Win",
                "largestLosingTrade": "Largest Loss",
                "averageTrade": "Average Trade",
            }

            def find_val_recursive(obj, target_key):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if k.lower() == target_key.lower():
                            return v
                        if isinstance(v, dict):
                            res = find_val_recursive(v, target_key)
                            if res is not None:
                                return res
                return None

            for json_key, col_prefix in mapping.items():
                val = ts.get(json_key)
                if val is None:
                    val = find_val_recursive(data, json_key)

                if isinstance(val, dict):
                    for mode in ["all", "long", "short"]:
                        try:
                            row_metrics[f"{col_prefix}_{mode}"] = float(val.get(mode, 0.0))
                        except (ValueError, TypeError):
                            row_metrics[f"{col_prefix}_{mode}"] = 0.0
                elif isinstance(val, (int, float)):
                    for mode in ["all", "long", "short"]:
                        row_metrics[f"{col_prefix}_{mode}"] = float(val)
                elif isinstance(val, str):
                    for mode in ["all", "long", "short"]:
                        try:
                            row_metrics[f"{col_prefix}_{mode}"] = float(re.sub(r"[^0-9.-]", "", val))
                        except ValueError:
                            row_metrics[f"{col_prefix}_{mode}"] = 0.0

            for mode in ["all", "long", "short"]:
                tot = row_metrics[f"Total Trades_{mode}"]
                win = row_metrics[f"Winning Trades_{mode}"]
                row_metrics[f"Win Rate_{mode}"] = round((win / tot) * 100, 2) if tot > 0 else 0.0

            row_metrics["_report_capital"] = (data.get("main") or {}).get("startingCapital")

            try:
                row_metrics["PnL_Series"] = self._extract_timeseries_logic(data)
            except Exception:
                row_metrics["PnL_Series"] = None

            try:
                row_metrics.update(self._extract_duration_logic(data))
            except Exception:
                pass

        except Exception:
            pass
        return row_metrics

    # ==========================================
    # 2. IN-MEMORY ZIP PROCESSOR (Cloud Mode)
    # ==========================================
    def _scan_zip_in_memory(self, progress_callback=None) -> Optional[pd.DataFrame]:
        """Streams files directly from the ZIP archive with strict garbage collection."""
        if self.zip_file is None:
            return None
        
        results = []
        
        try:
            with zipfile.ZipFile(self.zip_file, 'r') as z:
                folders = set()
                for info in z.infolist():
                    if info.filename.lower().endswith(('.cbotset', '.html', '.htm')):
                        folders.add(os.path.dirname(info.filename))
                
                valid_folders = [f for f in folders if f]
                zip_names = set(z.namelist())
                total_dirs = len(valid_folders)
                
                if total_dirs == 0:
                    return None

                for idx, folder_path in enumerate(valid_folders):
                    id_match = re.search(r"\d+", os.path.basename(folder_path))
                    test_id = int(id_match.group()) if id_match else idx
                    row_data = {"Test_ID": test_id}
                    
                    # 1. Stream cbotset (JSON)
                    cbotset_file = next((f for f in z.namelist() if f.startswith(folder_path) and f.lower().endswith('.cbotset')), None)
                    if cbotset_file:
                        with z.open(cbotset_file) as f:
                            try:
                                data = json.loads(f.read().decode('utf-8', errors='ignore'))
                                row_data.update(self._extract_cbotset_logic(data))
                                # MEMORY OPTIMIZATION: Clear the loaded JSON dictionary ONLY if successfully created
                                del data 
                            except Exception:
                                pass

                    # 2. Stream and Parse Report (HTML)
                    report_file = next((f for f in z.namelist() if f.startswith(folder_path) and f.lower().endswith(('.html', '.htm'))), None)
                    if report_file:
                        with z.open(report_file) as f:
                            try:
                                html_content = f.read().decode('utf-8', errors='ignore')
                                row_data.update(self._extract_report_logic(html_content))
                                # MEMORY OPTIMIZATION: Immediately delete the large HTML string from RAM
                                del html_content
                            except Exception:
                                pass

                    self._apply_starting_capital(row_data, self._zip_optres_capital(z, zip_names, folder_path))
                    results.append(row_data)

                    # MEMORY OPTIMIZATION: Force garbage collection periodically
                    if idx > 0 and idx % 500 == 0:
                        gc.collect()

                    if progress_callback and (idx % 25 == 0 or idx == total_dirs - 1):
                        progress_callback((idx + 1) / total_dirs)

        except MemoryError:
            st.error("🚨 Out of Memory Error: The selected dataset exceeds the free cloud limits.")
            st.warning("💡 Tip: Try uploading a ZIP file containing fewer optimization passes, or run the Dashboard locally.")
            return None
        except zipfile.BadZipFile:
            st.error("❌ Error: The uploaded file is not a valid ZIP archive.")
            return None
        except Exception as e:
            st.error(f"❌ Error processing ZIP stream: {str(e)}")
            return None

        # Final cleanup before returning DataFrame
        gc.collect()
        return self._finalize_dataframe(results)

    # ==========================================
    # 3. LOCAL SSD PROCESSOR (CLI mode)
    # ==========================================
    def _scan_local(self, progress_callback=None) -> Optional[pd.DataFrame]:
        """Read folders from the local drive with memory limits handled."""
        if not self.root_dir.exists() or not self.root_dir.is_dir():
            return None

        valid_folders = set()
        for root, _, files in os.walk(self.root_dir, followlinks=True):
            if any(f.lower().endswith((".cbotset", ".html", ".htm")) for f in files):
                valid_folders.add(Path(root))

        valid_folders = list(valid_folders)
        total_dirs = len(valid_folders)
        if total_dirs == 0:
            return None

        results = []
        try:
            for idx, folder in enumerate(valid_folders):
                id_match = re.search(r"\d+", folder.name)
                test_id = int(id_match.group()) if id_match else idx
                row_data = {"Test_ID": test_id}

                cbotset_file = next((folder / f for f in os.listdir(folder) if f.lower().endswith(".cbotset")), None)
                if cbotset_file and cbotset_file.exists():
                    try:
                        with open(cbotset_file, "r", encoding="utf-8", errors="ignore") as f:
                            data = json.load(f)
                            row_data.update(self._extract_cbotset_logic(data))
                            del data
                    except Exception:
                        pass

                report_file = next((folder / f for f in os.listdir(folder) if f.lower().endswith((".html", ".htm"))), None)
                if report_file and report_file.exists():
                    try:
                        with open(report_file, "r", encoding="utf-8", errors="ignore") as f:
                            html_content = f.read()
                            row_data.update(self._extract_report_logic(html_content))
                            del html_content
                    except Exception:
                        pass

                self._apply_starting_capital(row_data, self._local_optres_capital(folder))
                results.append(row_data)

                if idx > 0 and idx % 500 == 0:
                    gc.collect()

                if progress_callback and (idx % 25 == 0 or idx == total_dirs - 1):
                    progress_callback((idx + 1) / total_dirs)

        except MemoryError:
            print("🚨 Out of Memory Error: Processing stopped to prevent system crash.")
            return None
            
        df = self._finalize_dataframe(results)
        self._save_cache(df)
        gc.collect()
        return df

    # ==========================================
    # 4. COMMON DATA TYPE HANDLING
    # ==========================================
    def _finalize_dataframe(self, results: list) -> pd.DataFrame:
        """Converts the raw dictionaries into a strictly typed DataFrame."""
        if not results:
            return pd.DataFrame()
            
        df = pd.DataFrame(results).sort_values(by="Test_ID")
        
        # Modern Pandas dtype conversion replacing the deprecated errors='ignore'
        text_columns = {"Test_ID", "PnL_Series", "Timeframe", "Hold_Hist"}
        for col in df.columns:
            if col not in text_columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='raise')
                except (ValueError, TypeError):
                    pass
                
        for mode in ["all", "long", "short"]:
            trades_col = f"Total Trades_{mode}"
            if trades_col in df.columns:
                df[trades_col] = pd.to_numeric(df[trades_col], errors='coerce').fillna(0).astype(int)
                
        return df
    
    def _save_cache(self, df: pd.DataFrame):
        """Saves cache to Parquet (if engine available), otherwise CSV."""
        try:
            df.to_parquet(self.cache_parquet, index=False)
        except Exception:
            df.to_csv(self.cache_csv, index=False)