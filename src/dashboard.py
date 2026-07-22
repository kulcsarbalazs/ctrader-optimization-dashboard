import os
import uuid
import time
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, Any

from src.scanner import CTraderScanner
from src.analyzer import StrategyAnalyzer

class OptimizationDashboard:
    """Controls the entire Streamlit user interface."""

    def __init__(self, initial_root_dir: str = "."):
        st.set_page_config(page_title="cTrader OOP Analyzer", layout="wide")
        self.initial_root_dir = initial_root_dir
        self.root_dir = initial_root_dir
        self.scanner = None
        self.analyzer = None

    def _cleanup_old_cache_files(self, max_age_hours: int = 24):
        """
        Implements a 'Lazy Garbage Collection' pattern.
        Scans the /tmp directory and deletes Parquet files older than max_age_hours.
        Crucial for preventing disk space exhaustion on Streamlit Community Cloud.
        """
        tmp_dir = Path("/tmp")
        if not tmp_dir.exists():
            return
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        # Iterate over all cached parquet files specifically created by this app
        for cache_file in tmp_dir.glob("ctrader_opt_*.parquet"):
            try:
                # Check file modification time (st_mtime)
                file_mtime = cache_file.stat().st_mtime
                if (current_time - file_mtime) > max_age_seconds:
                    cache_file.unlink() # Delete the file
            except Exception:
                # Silently ignore files that are locked or already deleted
                pass

    def _safe_range_slider(self, label: str, series: pd.Series, step: float = None) -> tuple:
        """Safe slider that prevents UI crash when min == max."""
        s_min = float(series.min()) if not series.empty else 0.0
        s_max = float(series.max()) if not series.empty else 1.0
        if s_min >= s_max:
            s_max = s_min + 1.0
        return st.sidebar.slider(label, min_value=s_min, max_value=s_max, value=(s_min, s_max), step=step)

    def _render_sidebar(self) -> tuple:
        st.sidebar.header("📁 Data Source")

        run_mode = os.environ.get("ENV_MODE", "local").lower()
        
        raw_df = None

        # ==========================================
        # CLOUD MODE: ZIP UPLOAD (No folder path)
        # ==========================================
        if run_mode == "cloud":
            st.sidebar.caption("☁️ Cloud Engine (In-Memory & Cache)")
            
            # 1. Check if the user already has an active session in the URL
            session_id = st.query_params.get("session_id")
            cache_file = Path(f"/tmp/ctrader_opt_{session_id}.parquet") if session_id else None

            # 2. RESTORE FROM CACHE (If URL has ID and file exists on server)
            if cache_file and cache_file.exists():
                st.sidebar.success("✅ Session restored successfully!")
                
                if "cached_df" not in st.session_state:
                    # Instantly load the 200KB Parquet instead of re-uploading the ZIP
                    raw_df = pd.read_parquet(cache_file)
                    st.session_state["cached_df"] = raw_df
                else:
                    raw_df = st.session_state["cached_df"]

                # Allow user to completely reset their session
                if st.sidebar.button("🗑️ Start New Session", use_container_width=True):
                    cache_file.unlink(missing_ok=True)
                    st.query_params.clear()
                    st.session_state.clear()
                    st.rerun()

            # 3. FRESH UPLOAD (No valid session found)
            else:
                uploaded_zip = st.sidebar.file_uploader(
                    "Upload cTrader optimization ZIP file:", 
                    type=["zip"]
                )
                
                if uploaded_zip is not None:
                    if "cached_df" not in st.session_state:
                        with st.spinner("📦 Processing ZIP in memory (I/O free)..."):
                            
                            # Initialize scanner with the in-memory ZIP
                            self.scanner = CTraderScanner(zip_file=uploaded_zip)
                            raw_df = self.scanner.get_data()
                            
                            if raw_df is not None and not raw_df.empty:
                                # Trigger Garbage Collection before creating a new file
                                self._cleanup_old_cache_files(max_age_hours=24)
                                
                                # Generate a unique short ID for this upload
                                new_session_id = str(uuid.uuid4())[:8]
                                new_cache_file = Path(f"/tmp/ctrader_opt_{new_session_id}.parquet")
                                
                                # Ensure /tmp directory exists (standard on Linux/macOS)
                                new_cache_file.parent.mkdir(parents=True, exist_ok=True)
                                
                                # Save the highly compressed lightweight Parquet file
                                raw_df.to_parquet(new_cache_file, index=False)
                                
                                # Inject the ID into the browser's URL and save to RAM
                                st.query_params["session_id"] = new_session_id
                                st.session_state["cached_df"] = raw_df
                                
                                # Rerun to switch the UI from "Uploader" to "Restored" mode
                                st.rerun()

        # ==========================================
        # LOCAL MODE: CLI / Folder path
        # ==========================================
        else:
            st.sidebar.caption("💻 Run locally (Folder mode)")
            raw_input_dir = st.sidebar.text_input(
                "Optimization folders location:",
                value=self.initial_root_dir,
                help="The root directory, which contains the test folders.",
            )
            self.root_dir = str(raw_input_dir).strip().strip('"').strip("'").replace("\\ ", " ")
            self.scanner = CTraderScanner(self.root_dir)
            st.sidebar.caption(f"📍 **Current route:**\n`{self.scanner.root_dir}`")

            force_rescan = st.sidebar.button("🔄 Rescan / Clear Cache", use_container_width=True)
            if force_rescan:
                st.toast("⚡ Cache cleared, fresh scan starts!")

            progress_bar = st.sidebar.empty()
            raw_df = self.scanner.get_data(
                force_rescan=force_rescan,
                progress_callback=lambda p: (
                    progress_bar.progress(p, text="Processing folders (JSON)...") if p < 1.0 else progress_bar.empty()
                ),
            )

        if raw_df is None or raw_df.empty or "Total Trades_all" not in raw_df.columns:
            return None, {}, None

        # --- 1. TRADING MODE SELECTION ---
        st.sidebar.divider()
        st.sidebar.header("⚙️ Trading Mode")
        mode_label = st.sidebar.selectbox(
            "Select direction to analyze:",
            ["📊 Both (All)", "📈 Only Long", "📉 Only Short"],
        )
        mode = "all" if "Both" in mode_label else ("long" if "Long" in mode_label else "short")

        # --- 2. ACTIVE COLUMN MAPPING ---
        active_df = raw_df.copy()
        for col in [
            "Net Profit", "Profit Factor", "Swaps", "Total Trades", 
            "Winning Trades", "Losing Trades", "Max Cons. Losses", 
            "Largest Win", "Largest Loss", "Average Trade", "Win Rate"
        ]:
            col_name = f"{col}_{mode}"
            if col_name in active_df.columns:
                active_df[col] = active_df[col_name]

        self.analyzer = StrategyAnalyzer(active_df)

        # --- 3. MAIN PERFORMANCE FILTERS ---
        st.sidebar.divider()
        st.sidebar.header("🏆 Main Performance Filters")

        max_trades = int(active_df["Total Trades"].max()) if not active_df.empty else 1
        if max_trades <= 0:
            max_trades = 1
        min_trades = st.sidebar.slider(
            "Minimum Trades", min_value=0, max_value=max_trades, value=min(10, max_trades)
        )

        profit_range = self._safe_range_slider("Net Profit Range", active_df["Net Profit"])
        winrate_range = st.sidebar.slider("Win Rate (%) Range", 0.0, 100.0, (0.0, 100.0))
        max_dd = st.sidebar.slider("Max Equity Drawdown (%)", 0.0, 100.0, 100.0)
        pf_range = self._safe_range_slider("Profit Factor Range", active_df["Profit Factor"], step=0.05)

        # --- 4. DETAILED RISK FILTERS ---
        st.sidebar.divider()
        st.sidebar.header("🔬 Detailed Risk Filters")

        max_cons_val = int(active_df["Max Cons. Losses"].max()) if not active_df.empty else 10
        if max_cons_val <= 0:
            max_cons_val = 1
        max_cons_losses = st.sidebar.slider(
            "Max Consecutive Losses", min_value=0, max_value=max_cons_val, value=max_cons_val
        )

        largest_win_range = self._safe_range_slider("Largest Winning Trade", active_df["Largest Win"])
        largest_loss_range = self._safe_range_slider("Largest Losing Trade", active_df["Largest Loss"])
        avg_trade_range = self._safe_range_slider("Average Trade Result", active_df["Average Trade"])
        swaps_range = self._safe_range_slider("Swaps Range", active_df["Swaps"])

        filters = {
            "min_trades": min_trades,
            "profit_range": profit_range,
            "winrate_range": winrate_range,
            "max_dd": max_dd,
            "pf_range": pf_range,
            "max_cons_losses": max_cons_losses,
            "largest_win_range": largest_win_range,
            "largest_loss_range": largest_loss_range,
            "avg_trade_range": avg_trade_range,
            "swaps_range": swaps_range,
        }

        return active_df, filters, mode_label

    def _render_metrics(self, total_count: int, filtered_df: pd.DataFrame, mode_label: str):
        st.caption(f"⚡ Active Trading Mode: **{mode_label}**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Scanned Tests", f"{total_count}")
        col2.metric(
            "Passed Filters", f"{len(filtered_df)}", f"{round((len(filtered_df)/max(1, total_count))*100, 1)}%"
        )
        avg_pf = round(filtered_df["Profit Factor"].mean(), 2) if not filtered_df.empty else 0.0
        col3.metric("Avg Profit Factor (Filtered)", f"{avg_pf}")
        st.divider()

    def _render_filter_diagnostics(self, df: pd.DataFrame, filters: Dict[str, Any]):
        st.warning("⚠️ **No strategies left to display with the current filter settings!**")
        st.markdown("### 🔍 Filter Diagnostics (Where did the tests fail?)")

        total = len(df)
        t_ok = len(df[df["Total Trades"] >= filters["min_trades"]])
        p_ok = len(df[(df["Net Profit"] >= filters["profit_range"][0]) & (df["Net Profit"] <= filters["profit_range"][1])])
        w_ok = len(df[(df["Win Rate"] >= filters["winrate_range"][0]) & (df["Win Rate"] <= filters["winrate_range"][1])])
        d_ok = len(df[df["Equity Drawdown"] <= filters["max_dd"]])
        pf_ok = len(df[(df["Profit Factor"] >= filters["pf_range"][0]) & (df["Profit Factor"] <= filters["pf_range"][1])])

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("1. Passed Trades Count:", f"{t_ok}", f"{round(t_ok/max(1,total)*100, 1)}%")
        col2.metric("2. Passed Net Profit:", f"{p_ok}", f"{round(p_ok/max(1,total)*100, 1)}%")
        col3.metric("3. Passed Win Rate:", f"{w_ok}", f"{round(w_ok/max(1,total)*100, 1)}%")
        col4.metric("4. Passed Drawdown:", f"{d_ok}", f"{round(d_ok/max(1,total)*100, 1)}%")
        col5.metric("5. Passed Profit Factor:", f"{pf_ok}", f"{round(pf_ok/max(1,total)*100, 1)}%")

        st.info("💡 **Tip:** Adjust sliders that result in `0` passing tests to broaden your search space.")

        with st.expander("👀 View Raw, Unfiltered Data (Top 25)"):
            st.dataframe(
                df.head(25)[
                    ["Test_ID", "Total Trades", "Net Profit", "Win Rate", "Profit Factor", "Swaps", "Max Cons. Losses"]
                ]
            )

    def _render_charts(self, filtered_df: pd.DataFrame):
        st.subheader("1. 🎯 Identifying the 'Sweet Spot' (Win Rate vs. Profit)")
        st.write("Bubble size represents trade count; color indicates Equity Drawdown. Hover for parameter details!")

        param_cols = self.analyzer.get_parameter_columns(filtered_df)
        base_hover = ["Test_ID", "Profit Factor", "Swaps", "Max Cons. Losses"]
        dynamic_params = param_cols[:6]
        
        valid_hover_data = [col for col in (base_hover + dynamic_params) if col in filtered_df.columns]

        fig_scatter = px.scatter(
            filtered_df,
            x="Win Rate", y="Net Profit", color="Equity Drawdown", size="Total Trades",
            hover_data=valid_hover_data,
            color_continuous_scale=px.colors.diverging.RdYlGn[::-1],
            labels={
                "Win Rate": "Win Rate (%)", "Net Profit": "Net Profit", "Equity Drawdown": "Drawdown (%)",
            },
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("2. 🕸️ Parameter Pathways (Parallel Coordinates)")
        st.write("Visualizes how winning strategies flow through specific parameter combinations.")

        selected_params = param_cols[:7] + ["Win Rate", "Net Profit"]

        if len(selected_params) > 2 and not filtered_df.empty:
            fig_par = px.parallel_coordinates(
                filtered_df[selected_params],
                color="Net Profit",
                color_continuous_scale=px.colors.sequential.Viridis,
                labels={col: col.replace("Pips", " (P)").replace("Trigger", "Trig") for col in selected_params},
            )
            st.plotly_chart(fig_par, use_container_width=True)

        st.subheader("3. 🤖 Machine Learning: Feature Importance")
        st.write("A Random Forest model analyzes the filtered data to rank which parameters had the highest impact on Net Profit.")

        ml_df = self.analyzer.calculate_feature_importance(filtered_df, param_cols, target="Net Profit")
        if ml_df is not None:
            fig_bar = px.bar(
                ml_df.tail(12),
                x="Relevance (Weight %)", y="Parameter", orientation="h",
                text="Relevance (Weight %)", color="Relevance (Weight %)",
                color_continuous_scale=px.colors.sequential.Plasma,
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("⚠️ Not enough tests remain for ML training (minimum 10 required).")

    def _render_search_space_pruning(self, filtered_df: pd.DataFrame):
        st.divider()
        st.subheader("5. 🔬 Search Space Pruning (Optimization Re-targeting)")
        st.write("Use the calculated **Recommended Min and Max ranges** in cTrader to drastically reduce CPU time in future runs!")

        param_cols = self.analyzer.get_parameter_columns(filtered_df)
        if not param_cols or len(filtered_df) < 4:
            st.info("⚠️ At least 4 filtered strategies are required for this analysis.")
            return

        col_cfg1, col_cfg2 = st.columns(2)
        target_metric = col_cfg1.selectbox(
            "Define 'Best Strategies' by:", ["Net Profit", "Win Rate", "Profit Factor"]
        )
        elite_pct = col_cfg2.slider("Elite Cluster Size (Top %):", min_value=10, max_value=50, value=25, step=5) / 100.0

        pruning_df = self.analyzer.generate_pruning_report(
            filtered_df, param_cols, target_metric=target_metric, elite_percentile=elite_pct
        )

        st.markdown(f"#### 📋 Recommended cTrader Parameter Ranges (Based on Top {int(elite_pct*100)}%)")
        st.dataframe(pruning_df, use_container_width=True)

        csv_prune = pruning_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Pruned Ranges (CSV)", data=csv_prune, 
            file_name="cTrader_pruned_ranges.csv", mime="text/csv"
        )

        st.markdown("#### 📊 Visualization: Parameter Tightness within Elite Cluster")
        selected_param = st.selectbox("Select parameter to visualize:", param_cols)

        threshold = filtered_df[target_metric].quantile(1.0 - elite_pct)
        plot_df = filtered_df.copy()
        plot_df["Cluster"] = plot_df[target_metric].apply(
            lambda x: f"🔥 Elite Top {int(elite_pct*100)}%" if x >= threshold else "❄️ The Rest"
        )

        fig_box = px.box(
            plot_df, x="Cluster", y=selected_param, color="Cluster", points="all",
            color_discrete_map={f"🔥 Elite Top {int(elite_pct*100)}%": "#00CC96", "❄️ The Rest": "#636EFA"},
            labels={"Cluster": "Performance Cluster", selected_param: selected_param},
            title=f"Distribution and Tightness of {selected_param} in Elite Group",
        )
        st.plotly_chart(fig_box, use_container_width=True)

    def _render_data_table(self, filtered_df: pd.DataFrame):
        st.subheader("4. 📋 Top Strategy Cluster (Data Table)")
        st.dataframe(filtered_df.sort_values(by="Net Profit", ascending=False), use_container_width=True)

        csv_export = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Filtered List (CSV)", data=csv_export, 
            file_name="sweet_spot_strategies.csv", mime="text/csv"
        )

    def run(self):
        st.title("🚀 cTrader Multi-Folder Optimization Analyzer")
        st.write("Automated parsing, filtering, and Machine Learning parameter research for cTrader backtest optimizations.")

        df, filters, mode_label = self._render_sidebar()

        if df is None or df.empty:
            st.error("❌ No valid test folders found at the specified path!")
            return

        filtered_df = self.analyzer.filter_strategies(df, filters)
        self._render_metrics(len(df), filtered_df, mode_label)

        if not filtered_df.empty:
            self._render_charts(filtered_df)
            self._render_data_table(filtered_df)
            self._render_search_space_pruning(filtered_df)
        else:
            self._render_filter_diagnostics(df, filters)
