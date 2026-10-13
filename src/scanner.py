import os
import json
import re
import zipfile
import pandas as pd
import gc
import streamlit as st
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Union

# Columns a cache file must contain to be reused; older caches trigger a rescan.
REQUIRED_CACHE_COLUMNS = ("Total Trades_all", "PnL_Series")
TIMESERIES_FORMAT_VERSION = 1
DAY_MS = 24 * 3600 * 1000


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
        self.cache_csv = self.root_dir / "optimization_cache.csv"

    # ==========================================
    # COMMON INTERFACE (ROUTER)
    # ==========================================
    @staticmethod
    def is_cache_compatible(df: Optional[pd.DataFrame]) -> bool:
        """True if a cached DataFrame has every column the current version needs."""
        return df is not None and not df.empty and all(c in df.columns for c in REQUIRED_CACHE_COLUMNS)

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
        Format: {"v", "start", "end", "cap", "d": [[day_idx, net_all, net_long, net_short, trades, min_eq]]}
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
            return days.setdefault(idx, [idx, 0.0, 0.0, 0.0, 0, None])

        for t in trades:
            close_time = t.get("closeTime")
            if not close_time:
                continue
            net = float(t.get("net") or 0.0)
            row = bucket(close_time)
            row[1] += net
            if str(t.get("direction", "")).lower() == "buy":
                row[2] += net
            else:
                row[3] += net
            row[4] += 1

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
                [r[0], round(r[1], 2), round(r[2], 2), round(r[3], 2), r[4], None if r[5] is None else round(r[5], 2)]
                for r in (days[i] for i in sorted(days))
            ],
        }
        return json.dumps(payload, separators=(",", ":"))

    def _extract_report_logic(self, html: str) -> Dict[str, Any]:
        """Extracts and parses the pure JSON block from the HTML text."""
        row_metrics = {"Equity Drawdown": 0.0, "PnL_Series": None}

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

            try:
                row_metrics["PnL_Series"] = self._extract_timeseries_logic(data)
            except Exception:
                row_metrics["PnL_Series"] = None

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
        text_columns = {"Test_ID", "PnL_Series"}
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