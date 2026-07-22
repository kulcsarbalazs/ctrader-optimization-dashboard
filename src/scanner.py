import os
import json
import re
import zipfile
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Union

class CTraderScanner:
    """
    Dual-mode (Local SSD & In-Memory Cloud) cTrader optimization scanner.
    Lightning-fast JSON parameter and HTML report extraction.
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
    def get_data(self, force_rescan: bool = False, progress_callback=None) -> Optional[pd.DataFrame]:
        """
        Manages caching in local mode and starts scanning.
        """
        if self.env_mode == "cloud":
            return self.scan(progress_callback)

        if not force_rescan:
            if self.cache_parquet.exists():
                try:
                    df = pd.read_parquet(self.cache_parquet)
                    if not df.empty and "Total Trades_all" in df.columns:
                        return df
                except Exception:
                    pass
            if self.cache_csv.exists():
                try:
                    df = pd.read_csv(self.cache_csv)
                    if not df.empty and "Total Trades_all" in df.columns:
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
        """
        The main Facade method. It decides which data processing engine to start.
        """
        if self.env_mode == "cloud":
            if not self.zip_file:
                return None
            return self._scan_zip_in_memory(progress_callback)
        else:
            return self._scan_local(progress_callback)
        
    # ==========================================
    # 1. CORE PARSERS (They are independent of the data carrier)
    # ==========================================
    def _extract_cbotset_logic(self, data: dict) -> Dict[str, Any]:
        """Extracts parameters from an already loaded JSON dictionary."""
        params = {}
        raw_params = data.get("Parameters", {})
        for k, v in raw_params.items():
            params[k] = v 
        return params

    def _extract_report_logic(self, html: str) -> Dict[str, Any]:
        """Extracts and parses the pure JSON block from the HTML text."""
        row_metrics = {"Equity Drawdown": 0.0}

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

        except Exception:
            pass
        return row_metrics

    # ==========================================
    # 2. IN-MEMORY ZIP PROCESSOR (Cloud Mode)
    # ==========================================
    def _scan_zip_in_memory(self, progress_callback=None) -> Optional[pd.DataFrame]:
        """Streams files directly from the ZIP archive without writing to disk."""
        if self.zip_file is None:
            return None
        
        results = []
        
        try:
            with zipfile.ZipFile(self.zip_file, 'r') as z:
                # We collect the folders (to identify unique IDs)
                folders = set()
                for info in z.infolist():
                    if info.filename.lower().endswith(('.cbotset', '.html', '.htm')):
                        # Extract the file directory (e.g. "Opt_1/15/report.html" -> "Opt_1/15")
                        folders.add(os.path.dirname(info.filename))
                
                valid_folders = [f for f in folders if f]
                total_dirs = len(valid_folders)
                
                if total_dirs == 0:
                    return None

                for idx, folder_path in enumerate(valid_folders):
                    id_match = re.search(r"\d+", os.path.basename(folder_path))
                    test_id = int(id_match.group()) if id_match else idx
                    row_data = {"Test_ID": test_id}
                    
                    # 1. Search and stream cbotset
                    cbotset_file = next((f for f in z.namelist() if f.startswith(folder_path) and f.lower().endswith('.cbotset')), None)
                    if cbotset_file:
                        with z.open(cbotset_file) as f:
                            try:
                                data = json.loads(f.read().decode('utf-8', errors='ignore'))
                                row_data.update(self._extract_cbotset_logic(data))
                            except Exception:
                                pass

                    # 2. Search and stream reports
                    report_file = next((f for f in z.namelist() if f.startswith(folder_path) and f.lower().endswith(('.html', '.htm'))), None)
                    if report_file:
                        with z.open(report_file) as f:
                            html_content = f.read().decode('utf-8', errors='ignore')
                            row_data.update(self._extract_report_logic(html_content))
                    
                    results.append(row_data)

                    if progress_callback and (idx % 25 == 0 or idx == total_dirs - 1):
                        progress_callback((idx + 1) / total_dirs)

        except zipfile.BadZipFile:
            return None

        return self._finalize_dataframe(results)

    # ==========================================
    # 3. LOCAL SSD PROCESSOR (CLI mode)
    # ==========================================
    def _scan_local(self, progress_callback=None) -> Optional[pd.DataFrame]:
        """Read folders from the local drive (SSD/HDD)."""
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
        for idx, folder in enumerate(valid_folders):
            id_match = re.search(r"\d+", folder.name)
            test_id = int(id_match.group()) if id_match else idx
            row_data = {"Test_ID": test_id}

            cbotset_file = next((folder / f for f in os.listdir(folder) if f.lower().endswith(".cbotset")), None)
            if cbotset_file and cbotset_file.exists():
                try:
                    with open(cbotset_file, "r", encoding="utf-8", errors="ignore") as f:
                        row_data.update(self._extract_cbotset_logic(json.load(f)))
                except Exception:
                    pass

            report_file = next((folder / f for f in os.listdir(folder) if f.lower().endswith((".html", ".htm"))), None)
            if report_file and report_file.exists():
                try:
                    with open(report_file, "r", encoding="utf-8", errors="ignore") as f:
                        row_data.update(self._extract_report_logic(f.read()))
                except Exception:
                    pass

            results.append(row_data)

            if progress_callback and (idx % 25 == 0 or idx == total_dirs - 1):
                progress_callback((idx + 1) / total_dirs)

        df = self._finalize_dataframe(results)
        self._save_cache(df)
        return df

    # ==========================================
    # 4. COMMON DATA TYPE HANDLING (Type Casting)
    # ==========================================
    def _finalize_dataframe(self, results: list) -> pd.DataFrame:
        """It converts the read raw dictionaries into a typed DataFrame."""
        df = pd.DataFrame(results).sort_values(by="Test_ID")
        
        for col in df.columns:
            if col != "Test_ID":
                df[col] = pd.to_numeric(df[col], errors='ignore')
                
        for mode in ["all", "long", "short"]:
            trades_col = f"Total Trades_{mode}"
            if trades_col in df.columns:
                df[trades_col] = pd.to_numeric(df[trades_col], errors='coerce').fillna(0).astype(int)
                
        return df
    
    # def _parse_cbotset(self, file_path: Path) -> Dict[str, Any]:
    #     params = {}
    #     if not file_path.exists():
    #         return params
    #     try:
    #         with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
    #             data = json.load(f)
    #             raw_params = data.get("Parameters", {})
    #             for k, v in raw_params.items():
    #                 try:
    #                     params[k] = float(v)
    #                 except ValueError:
    #                     params[k] = str(v)
    #     except Exception:
    #         pass
    #     return params

    # def _parse_report_json(self, file_path: Path) -> Dict[str, Any]:
    #     """
    #     Extracts the pure JSON block from the cTrader HTML report [id="backtesting-report"]
    #     for all 3 trading modes (all, long, short).
    #     """
    #     row_metrics = {"Equity Drawdown": 0.0}

    #     # Initialize default columns for all 3 modes
    #     for mode in ["all", "long", "short"]:
    #         for col in [
    #             "Net Profit", "Profit Factor", "Swaps", "Total Trades", 
    #             "Winning Trades", "Losing Trades", "Max Cons. Losses", 
    #             "Largest Win", "Largest Loss", "Average Trade", "Win Rate"
    #         ]:
    #             row_metrics[f"{col}_{mode}"] = 0.0

    #     if not file_path.exists():
    #         return row_metrics

    #     try:
    #         with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
    #             html = f.read()

    #         match = re.search(
    #             r'<script[^>]*id=["\']backtesting-report["\'][^>]*>(.*?)</script>', 
    #             html, re.DOTALL | re.IGNORECASE
    #         )
    #         if not match:
    #             return row_metrics

    #         data = json.loads(match.group(1).strip())

    #         # 1. Read fixed JSON paths
    #         eq = data.get("equity", {})
    #         ts = data.get("tradeStatistics", {})

    #         row_metrics["Equity Drawdown"] = float(
    #             eq.get("maxEquityDrawdownPercent", eq.get("maxBalanceDrawdownPercent", 0.0))
    #         )

    #         mapping = {
    #             "netProfit": "Net Profit",
    #             "profitFactor": "Profit Factor",
    #             "swaps": "Swaps",
    #             "totalTrades": "Total Trades",
    #             "winningTrades": "Winning Trades",
    #             "losingTrades": "Losing Trades",
    #             "maxConsecutiveLosingTrades": "Max Cons. Losses",
    #             "largestWinningTrade": "Largest Win",
    #             "largestLosingTrade": "Largest Loss",
    #             "averageTrade": "Average Trade",
    #         }

    #         # 2. Recursive fallback search
    #         def find_val_recursive(obj, target_key):
    #             if isinstance(obj, dict):
    #                 for k, v in obj.items():
    #                     if k.lower() == target_key.lower():
    #                         return v
    #                     if isinstance(v, dict):
    #                         res = find_val_recursive(v, target_key)
    #                         if res is not None:
    #                             return res
    #             return None

    #         for json_key, col_prefix in mapping.items():
    #             val = ts.get(json_key)
    #             if val is None:
    #                 val = find_val_recursive(data, json_key)

    #             if isinstance(val, dict):
    #                 for mode in ["all", "long", "short"]:
    #                     try:
    #                         row_metrics[f"{col_prefix}_{mode}"] = float(val.get(mode, 0.0))
    #                     except (ValueError, TypeError):
    #                         row_metrics[f"{col_prefix}_{mode}"] = 0.0
    #             elif isinstance(val, (int, float)):
    #                 for mode in ["all", "long", "short"]:
    #                     row_metrics[f"{col_prefix}_{mode}"] = float(val)
    #             elif isinstance(val, str):
    #                 for mode in ["all", "long", "short"]:
    #                     try:
    #                         row_metrics[f"{col_prefix}_{mode}"] = float(re.sub(r"[^0-9.-]", "", val))
    #                     except ValueError:
    #                         row_metrics[f"{col_prefix}_{mode}"] = 0.0

    #         # Calculate Win Rate for all 3 modes
    #         for mode in ["all", "long", "short"]:
    #             tot = row_metrics[f"Total Trades_{mode}"]
    #             win = row_metrics[f"Winning Trades_{mode}"]
    #             row_metrics[f"Win Rate_{mode}"] = round((win / tot) * 100, 2) if tot > 0 else 0.0

    #     except Exception:
    #         pass
    #     return row_metrics

    # def scan(self, progress_callback=None) -> Optional[pd.DataFrame]:
    #     if not self.root_dir.exists() or not self.root_dir.is_dir():
    #         return None

    #     valid_folders = set()
    #     for root, _, files in os.walk(self.root_dir, followlinks=True):
    #         has_cbotset = any(f.lower().endswith(".cbotset") for f in files)
    #         has_report = any(f.lower().endswith((".html", ".htm")) for f in files)
    #         if has_cbotset or has_report:
    #             valid_folders.add(Path(root))

    #     valid_folders = list(valid_folders)
    #     if not valid_folders:
    #         return None

    #     results = []
    #     total_dirs = len(valid_folders)

    #     for idx, folder in enumerate(valid_folders):
    #         cbotset_file = next(
    #             (folder / f for f in os.listdir(folder) if f.lower().endswith(".cbotset")),
    #             folder / "parameters.cbotset",
    #         )
    #         report_file = next(
    #             (folder / f for f in os.listdir(folder) if f.lower().endswith((".html", ".htm"))),
    #             folder / "report.html",
    #         )

    #         id_match = re.search(r"\d+", folder.name)
    #         test_id = int(id_match.group()) if id_match else idx

    #         row_data = {"Test_ID": test_id}
    #         row_data.update(self._parse_cbotset(cbotset_file))
    #         row_data.update(self._parse_report_json(report_file))
    #         results.append(row_data)

    #         if progress_callback and (idx % 25 == 0 or idx == total_dirs - 1):
    #             progress_callback((idx + 1) / total_dirs)

    #     df = pd.DataFrame(results).sort_values(by="Test_ID")

    #     for col in df.columns:
    #         if col != "Test_ID": 
    #             df[col] = pd.to_numeric(df[col], errors='ignore')

    #     for mode in ["all", "long", "short"]:
    #         trades_col = f"Total Trades_{mode}"
    #         if trades_col in df.columns:
    #             df[trades_col] = pd.to_numeric(df[trades_col], errors='coerce').fillna(0).astype(int)

    #     self._save_cache(df)
    #     return df

    def _save_cache(self, df: pd.DataFrame):
        """Saves cache to Parquet (if engine available), otherwise CSV."""
        try:
            df.to_parquet(self.cache_parquet, index=False)
        except Exception:
            df.to_csv(self.cache_csv, index=False)

    # def get_data(self, force_rescan: bool = False, progress_callback=None) -> Optional[pd.DataFrame]:
    #     """Loads data from cache (Parquet/CSV) without scanning if possible."""
    #     if not force_rescan:
    #         if self.cache_parquet.exists():
    #             try:
    #                 df = pd.read_parquet(self.cache_parquet)
    #                 if not df.empty and "Total Trades_all" in df.columns:
    #                     return df
    #             except Exception:
    #                 pass
    #         if self.cache_csv.exists():
    #             try:
    #                 df = pd.read_csv(self.cache_csv)
    #                 if not df.empty and "Total Trades_all" in df.columns:
    #                     return df
    #             except Exception:
    #                 pass

    #     if force_rescan:
    #         if self.cache_parquet.exists():
    #             self.cache_parquet.unlink()
    #         if self.cache_csv.exists():
    #             self.cache_csv.unlink()

    #     return self.scan(progress_callback)
