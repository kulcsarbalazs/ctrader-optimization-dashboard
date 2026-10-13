import os
import json
import uuid
import time
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any

from src.scanner import CTraderScanner
from src.analyzer import StrategyAnalyzer
from src.timeseries import TimeSeriesAnalyzer


@st.cache_data(max_entries=16, show_spinner=False)
def _cached_time_series(series_df: pd.DataFrame, mode: str, freq: str, cumulative: bool) -> pd.DataFrame:
    return TimeSeriesAnalyzer.aggregate(TimeSeriesAnalyzer.expand(series_df, mode), freq, cumulative)


@st.cache_data(max_entries=16, show_spinner=False)
def _cached_month_heatmap(series_df: pd.DataFrame, mode: str, metric: str):
    return TimeSeriesAnalyzer.month_heatmap(TimeSeriesAnalyzer.expand(series_df, mode), metric)


@st.cache_data(max_entries=8, show_spinner=False)
def _cached_trend_metrics(series_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    mats = TimeSeriesAnalyzer.weekly_matrices(TimeSeriesAnalyzer.expand(series_df, mode))
    # The equity low is account-wide, so it only applies when both directions are analyzed.
    return TimeSeriesAnalyzer.trend_metrics(mats, use_equity=(mode == "all"))


@st.cache_data(max_entries=24, show_spinner=False)
def _cached_set_curves(series_df: pd.DataFrame, mode: str, roi: bool) -> Dict[str, pd.DataFrame]:
    mats = TimeSeriesAnalyzer.weekly_matrices(TimeSeriesAnalyzer.expand(series_df, mode))
    equity = TimeSeriesAnalyzer.equity_curves(mats, roi=roi)
    drawdown = TimeSeriesAnalyzer.drawdown_curves(mats, use_equity=(mode == "all"))
    return {
        "equity": equity,
        "equity_stats": TimeSeriesAnalyzer.band_stats(equity),
        "drawdown_stats": TimeSeriesAnalyzer.band_stats(drawdown),
    }


TREND_COLUMNS = ["Trend R²", "Ulcer Index", "Max Underwater (w)", "Weekly Vol %", "Weekly Sharpe"]
MAX_SAVED_SETS = 4
SET_COLORS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]


class OptimizationDashboard:
    """Controls the entire Streamlit user interface."""

    def __init__(self, initial_root_dir: str = "."):
        st.set_page_config(page_title="cTrader Optimization Analyzer", layout="wide")
        self.initial_root_dir = initial_root_dir
        self.root_dir = initial_root_dir
        self.scanner = None
        self.analyzer = None
        self.run_mode = os.environ.get("ENV_MODE", "local").lower()
        self.mode = "all"

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
        for cache_file in [*tmp_dir.glob("ctrader_opt_*.parquet"), *tmp_dir.glob("ctrader_opt_*_sets.json")]:
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

        raw_df = None

        # ==========================================
        # CLOUD MODE: ZIP UPLOAD (No folder path)
        # ==========================================
        if self.run_mode == "cloud":
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

                if not CTraderScanner.is_cache_compatible(raw_df):
                    st.sidebar.warning(
                        "This session was created by an older version. "
                        "Start a new session and re-upload the ZIP to enable the Time-Series Breakdown."
                    )

                # Allow user to completely reset their session
                if st.sidebar.button("🗑️ Start New Session", use_container_width=True):
                    cache_file.unlink(missing_ok=True)
                    Path(f"/tmp/ctrader_opt_{session_id}_sets.json").unlink(missing_ok=True)
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
        self.mode = mode

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

        if "PnL_Series" in active_df.columns and active_df["PnL_Series"].notna().any():
            trend_df = _cached_trend_metrics(active_df[["Test_ID", "PnL_Series"]], mode)
            active_df = active_df.drop(columns=TREND_COLUMNS, errors="ignore").merge(trend_df, on="Test_ID", how="left")

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

        # --- 5. TREND & STABILITY FILTERS ---
        trend_filters = {}
        if "Trend R²" in active_df.columns:
            st.sidebar.divider()
            st.sidebar.header("📈 Trend & Stability Filters")
            min_r2 = st.sidebar.slider(
                "Min Trend Stability (R²)", -1.0, 1.0, -1.0, step=0.01,
                help="Linear fit of the weekly balance curve: 1 = perfectly steady growth, negative = steady decline.",
            )
            ulcer_max = float(np.ceil(active_df["Ulcer Index"].max() or 1.0))
            max_ulcer = st.sidebar.slider(
                "Max Ulcer Index", 0.0, max(ulcer_max, 1.0), max(ulcer_max, 1.0), step=0.05,
                help="Depth and duration of drawdowns combined (root mean square of the weekly drawdown %).",
            )
            uw_max = int(active_df["Max Underwater (w)"].max() or 1)
            max_underwater = st.sidebar.slider(
                "Max Time Underwater (weeks)", 0, max(uw_max, 1), max(uw_max, 1),
                help="Longest stretch the balance stayed below its previous peak.",
            )
            trend_filters = {
                "Trend R²": (">=", min_r2, min_r2 > -1.0),
                "Ulcer Index": ("<=", max_ulcer, max_ulcer < max(ulcer_max, 1.0)),
                "Max Underwater (w)": ("<=", max_underwater, max_underwater < max(uw_max, 1)),
            }

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
            "trend_filters": trend_filters,
        }

        st.sidebar.divider()
        st.sidebar.markdown("### 👨‍💻 About the Author")
        st.sidebar.markdown("Hi! I build high-performance cTrader tools. If this tool saved you time or CPU costs, consider supporting the project!")
        
        # Donatr.ee / Ko-fi gomb
        st.sidebar.markdown(
            """
            <a href='https://donatr.ee/kulcsarbalazs' target='_blank'>
                <img src='https://img.shields.io/badge/Support_me-Donatr.ee-FF5E5B?style=for-the-badge' alt='Support me' />
            </a>
            """,
            unsafe_allow_html=True
        )

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

        active_trend = {k: v for k, v in filters.get("trend_filters", {}).items() if v[2] and k in df.columns}
        if active_trend:
            trend_ok = int(self.analyzer.trend_mask(df, active_trend).sum())
            st.metric("6. Passed Trend & Stability:", f"{trend_ok}", f"{round(trend_ok/max(1,total)*100, 1)}%")

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
            "Define 'Best Strategies' by:",
            ["Net Profit", "Win Rate", "Profit Factor"] + (["Trend R²"] if "Trend R²" in filtered_df.columns else [])
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

    @st.fragment
    def _render_time_series(self, filtered_df: pd.DataFrame):
        st.divider()
        st.subheader("6. 📅 Time-Series Breakdown")
        st.write(
            "How the filtered strategies performed period by period. "
            "Wide bands or deep red cells reveal seasonal or regime-specific weaknesses."
        )

        if "PnL_Series" not in filtered_df.columns or filtered_df["PnL_Series"].isna().all():
            st.info("ℹ️ No time-series data in the loaded dataset. Use **Rescan / Clear Cache** (or start a new session) to rebuild it.")
            return
        if len(filtered_df) < 2:
            st.info("⚠️ At least 2 filtered strategies are required for this analysis.")
            return

        series_df = filtered_df[["Test_ID", "PnL_Series"]]
        freq_labels = {"Weekly": "W", "Monthly": "M", "Yearly": "Y"}

        with st.container(horizontal=True, vertical_alignment="bottom"):
            freq_label = st.segmented_control(
                "Breakdown", list(freq_labels), default="Monthly", key="ts_freq"
            ) or "Monthly"
            cumulative = st.toggle("Cumulative", key="ts_cumulative", help="Show the running total instead of each period's own P/L.")

        stats = _cached_time_series(series_df, self.mode, freq_labels[freq_label], cumulative)
        if stats.empty:
            st.info("ℹ️ The filtered strategies have no closed trades to break down.")
            return

        x = stats["period_start"]
        upper = stats["mean"] + stats["std"]
        lower = stats["mean"] - stats["std"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=upper, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            x=x, y=lower, mode="lines", line=dict(width=0), fill="tonexty",
            fillcolor="rgba(99, 110, 250, 0.25)", name="± 1 std", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(x=x, y=stats["max"], mode="lines", name="Max", line=dict(dash="dot", width=1, color="#00CC96")))
        fig.add_trace(go.Scatter(x=x, y=stats["min"], mode="lines", name="Min", line=dict(dash="dot", width=1, color="#EF553B")))
        fig.add_trace(go.Scatter(
            x=x, y=stats["mean"], mode="lines", name="Mean", line=dict(width=2, color="#636EFA"),
            customdata=stats[["std", "n", "loss_ratio", "median"]].to_numpy(),
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>Mean: %{y:.2f}<br>Std: %{customdata[0]:.2f}<br>Median: %{customdata[3]:.2f}"
                "<br>Losing tests: %{customdata[2]:.1f}%<br>Tests: %{customdata[1]}<extra></extra>"
            ),
        ))
        fig.add_hline(y=0, line_width=1, line_color="gray")
        fig.update_layout(
            title=f"{'Cumulative' if cumulative else freq_label} P/L across {len(filtered_df)} strategies (mean ± std, min/max)",
            xaxis_title="Period", yaxis_title="Net Profit", hovermode="x unified",
        )
        st.plotly_chart(fig, width="stretch")

        st.markdown("#### 🗓️ Seasonality heatmap (year × month)")
        heat_metric = st.segmented_control(
            "Heatmap metric", ["Average profit", "Losing tests (%)"], default="Average profit", key="ts_heat_metric"
        ) or "Average profit"
        is_profit = heat_metric == "Average profit"
        matrix = _cached_month_heatmap(series_df, self.mode, "mean_profit" if is_profit else "loss_ratio")
        if matrix is not None and not matrix.empty:
            fig_heat = px.imshow(
                matrix, text_auto=".0f" if not is_profit else ".1f", aspect="auto",
                color_continuous_scale="RdYlGn" if is_profit else "Reds",
                color_continuous_midpoint=0 if is_profit else None,
                labels=dict(x="Month", y="Year", color="Avg profit" if is_profit else "Losing tests %"),
            )
            st.plotly_chart(fig_heat, width="stretch")

        if not cumulative:
            st.markdown(f"#### ⚠️ Weakest {freq_label.lower()} periods")
            st.caption("Periods with the highest share of losing strategies, then the lowest average P/L.")
            period_format = {"Weekly": "%Y-%m-%d", "Monthly": "%Y-%m", "Yearly": "%Y"}[freq_label]
            worst = stats.sort_values(["loss_ratio", "mean"], ascending=[False, True]).head(10).copy()
            worst["period_start"] = worst["period_start"].dt.strftime(period_format)
            worst = worst[["period_start", "loss_ratio", "mean", "median", "std", "min", "max", "n"]].rename(columns={
                "period_start": "Period start", "mean": "Mean", "std": "Std", "min": "Min",
                "max": "Max", "median": "Median", "loss_ratio": "Losing tests (%)", "n": "Tests",
            }).round(2)
            st.dataframe(worst, hide_index=True, width="stretch")
            st.download_button(
                "📥 Download Period Statistics (CSV)",
                data=stats.round(4).to_csv(index=False).encode("utf-8"),
                file_name=f"cTrader_time_series_{freq_label.lower()}.csv", mime="text/csv",
            )

    # ==========================================
    # SAVED FILTER SETS
    # ==========================================
    def _saved_sets_file(self):
        """Cloud sessions keep their saved sets next to the session's Parquet cache."""
        session_id = st.query_params.get("session_id")
        if self.run_mode == "cloud" and session_id:
            return Path(f"/tmp/ctrader_opt_{session_id}_sets.json")
        return None

    def _get_saved_sets(self) -> Dict[str, Any]:
        if "saved_sets" not in st.session_state:
            sets = {}
            sets_file = self._saved_sets_file()
            if sets_file and sets_file.exists():
                try:
                    sets = json.loads(sets_file.read_text())
                except Exception:
                    sets = {}
            st.session_state["saved_sets"] = sets
        return st.session_state["saved_sets"]

    def _persist_saved_sets(self):
        sets_file = self._saved_sets_file()
        if sets_file:
            try:
                sets_file.write_text(json.dumps(st.session_state["saved_sets"]))
            except Exception:
                pass

    @staticmethod
    def _rgba(hex_color: str, alpha: float) -> str:
        h = hex_color.lstrip("#")
        return f"rgba({int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}, {alpha})"

    @st.fragment
    def _render_trend_volatility(self, filtered_df: pd.DataFrame, full_df: pd.DataFrame):
        st.divider()
        st.subheader("7. 📈 Trend & Volatility Comparison")
        st.write(
            "Average balance curve of the filtered strategies with a ±1 std band and the min/max envelope, "
            "the drawdown (underwater) curve below it, and a side-by-side comparison with your saved filter sets."
        )

        if "Trend R²" not in filtered_df.columns:
            st.info("ℹ️ No time-series data in the loaded dataset. Use **Rescan / Clear Cache** (or start a new session) to rebuild it.")
            return

        mode_names = {"all": "All", "long": "Long", "short": "Short"}
        sets = self._get_saved_sets()

        # --- Save / delete sets ---
        with st.container(horizontal=True, vertical_alignment="bottom"):
            set_name = st.text_input("Set name", placeholder=f"Set {len(sets) + 1}", key="tv_set_name")
            save_clicked = st.button("💾 Save current filtered set", disabled=len(sets) >= MAX_SAVED_SETS)
        if save_clicked:
            name = set_name.strip() or f"Set {len(sets) + 1}"
            sets[name] = {
                "test_ids": [int(t) for t in filtered_df["Test_ID"]],
                "mode": self.mode,
                "created": time.strftime("%Y-%m-%d %H:%M"),
            }
            self._persist_saved_sets()
            st.session_state["tv_compare"] = [*st.session_state.get("tv_compare", []), name]
            st.toast(f"Saved '{name}' with {len(filtered_df)} strategies.")
        if len(sets) >= MAX_SAVED_SETS:
            st.caption(f"⚠️ Maximum {MAX_SAVED_SETS} saved sets. Delete one to save a new set.")

        compare = []
        if sets:
            st.session_state["tv_compare"] = [n for n in st.session_state.get("tv_compare", list(sets)) if n in sets]
            compare = st.pills(
                "Compare with saved sets", list(sets), selection_mode="multi", key="tv_compare",
                format_func=lambda n: f"{n} ({len(sets[n]['test_ids'])}, saved in {mode_names.get(sets[n]['mode'], '?')})",
            ) or []
            with st.container(horizontal=True, vertical_alignment="bottom"):
                to_delete = st.selectbox("Saved set", list(sets), key="tv_delete_name", label_visibility="collapsed")
                if st.button("🗑️ Delete set"):
                    sets.pop(to_delete, None)
                    self._persist_saved_sets()
                    st.rerun(scope="fragment")

        # --- Chart options ---
        with st.container(horizontal=True, vertical_alignment="bottom"):
            scale = st.segmented_control("Scale", ["Absolute", "ROI %"], default="Absolute", key="tv_scale") or "Absolute"
            spaghetti = st.toggle("Show individual curves", key="tv_spaghetti", help="Top strategies of the current filter as faint lines.")
            if spaghetti:
                top_n = st.number_input("Top N", min_value=1, max_value=50, value=20, key="tv_top_n")
                rank_by = st.selectbox("Rank by", ["Net Profit", "Trend R²"], key="tv_rank_by")
        roi = scale == "ROI %"

        groups = [("Current filter", filtered_df)]
        for name in compare:
            groups.append((name, full_df[full_df["Test_ID"].isin(sets[name]["test_ids"])]))

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.06,
            subplot_titles=("Balance curve" + (" (ROI %)" if roi else ""), "Drawdown from peak (%)"),
        )
        summary = []
        for idx, (name, group_df) in enumerate(groups):
            if group_df.empty or group_df["PnL_Series"].isna().all():
                continue
            color = SET_COLORS[idx % len(SET_COLORS)]
            curves = _cached_set_curves(group_df[["Test_ID", "PnL_Series"]], self.mode, roi)
            eq, dd = curves["equity_stats"], curves["drawdown_stats"]
            if eq.empty:
                continue
            x = eq["period_start"]
            many = len(group_df) > 1

            if idx == 0 and spaghetti:
                top_ids = group_df.nlargest(int(top_n), rank_by)["Test_ID"]
                for test_id in [t for t in top_ids if t in curves["equity"].columns]:
                    line = curves["equity"][test_id].dropna()
                    fig.add_trace(go.Scatter(
                        x=line.index, y=line, mode="lines", line=dict(width=1, color=self._rgba("#9AA0A6", 0.35)),
                        name=f"Test {test_id}", legendgroup="spaghetti", showlegend=False,
                        hovertemplate=f"Test {test_id}<br>%{{x|%Y-%m-%d}}: %{{y:.2f}}<extra></extra>",
                    ), row=1, col=1)

            if many:
                fig.add_trace(go.Scatter(x=x, y=eq["mean"] + eq["std"], mode="lines", line=dict(width=0),
                                         legendgroup=name, showlegend=False, hoverinfo="skip"), row=1, col=1)
                fig.add_trace(go.Scatter(x=x, y=eq["mean"] - eq["std"], mode="lines", line=dict(width=0),
                                         fill="tonexty", fillcolor=self._rgba(color, 0.18),
                                         legendgroup=name, showlegend=False, hoverinfo="skip"), row=1, col=1)
                for col, dash in (("max", "dot"), ("min", "dot")):
                    fig.add_trace(go.Scatter(x=x, y=eq[col], mode="lines", line=dict(width=1, dash=dash, color=color),
                                             legendgroup=name, showlegend=False, name=f"{name} {col}",
                                             hovertemplate=f"{name} {col}: %{{y:.2f}}<extra></extra>"), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=x, y=eq["mean"], mode="lines", line=dict(width=2.5, color=color), name=f"{name} ({len(group_df)})",
                legendgroup=name, customdata=eq[["std", "n"]].to_numpy(),
                hovertemplate=f"{name}<br>%{{x|%Y-%m-%d}}<br>Mean: %{{y:.2f}}<br>Std: %{{customdata[0]:.2f}}<br>Tests: %{{customdata[1]}}<extra></extra>",
            ), row=1, col=1)
            fig.add_trace(go.Scatter(x=dd["period_start"], y=dd["mean"], mode="lines", line=dict(width=2, color=color),
                                     legendgroup=name, showlegend=False,
                                     hovertemplate=f"{name} avg DD: %{{y:.2f}}%<extra></extra>"), row=2, col=1)
            if many:
                fig.add_trace(go.Scatter(x=dd["period_start"], y=dd["min"], mode="lines",
                                         line=dict(width=1, dash="dot", color=color), legendgroup=name, showlegend=False,
                                         hovertemplate=f"{name} worst DD: %{{y:.2f}}%<extra></extra>"), row=2, col=1)

            summary.append({
                "Set": name,
                "Tests": len(group_df),
                "Avg Net Profit": round(group_df["Net Profit"].mean(), 2),
                "Avg Trend R²": round(group_df["Trend R²"].mean(), 3),
                "Median Trend R²": round(group_df["Trend R²"].median(), 3),
                "Avg Ulcer Index": round(group_df["Ulcer Index"].mean(), 3),
                "Worst DD %": round(dd["min"].min(), 2),
                "Avg Max Underwater (w)": round(group_df["Max Underwater (w)"].mean(), 1),
                "Avg Weekly Vol %": round(group_df["Weekly Vol %"].mean(), 3),
                "Avg Weekly Sharpe": round(group_df["Weekly Sharpe"].mean(), 3),
            })

        if not summary:
            st.info("ℹ️ The selected strategies have no time-series data to plot.")
            return

        fig.update_layout(height=720, hovermode="x unified", legend=dict(orientation="h", y=1.08))
        fig.update_yaxes(title_text="ROI %" if roi else "Balance", row=1, col=1)
        fig.update_yaxes(title_text="DD %", row=2, col=1)
        st.plotly_chart(fig, width="stretch")

        st.markdown("#### 📊 Set comparison")
        summary_df = pd.DataFrame(summary)
        st.dataframe(summary_df, hide_index=True, width="stretch")
        st.download_button(
            "📥 Download Set Comparison (CSV)", data=summary_df.to_csv(index=False).encode("utf-8"),
            file_name="cTrader_set_comparison.csv", mime="text/csv",
        )

    def _render_data_table(self, filtered_df: pd.DataFrame):
        st.subheader("4. 📋 Top Strategy Cluster (Data Table)")
        table_df = filtered_df.drop(columns=["PnL_Series"], errors="ignore")
        st.dataframe(table_df.sort_values(by="Net Profit", ascending=False), use_container_width=True)

        csv_export = table_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Filtered List (CSV)", data=csv_export, 
            file_name="sweet_spot_strategies.csv", mime="text/csv"
        )

    def _render_empty_state(self):
        """Displays instructions if no data has been loaded yet."""

        if self.run_mode == "cloud":
            st.info("👈 **Get Started:** Please upload your cTrader Optimization `.zip` file in the sidebar to begin.")
            st.markdown("""
            ### ☁️ How to use the Cloud Analyzer:
            1. **Save Results:** In cTrader, save your optimization results once the backtesting process is complete.
            2. **Compress to ZIP:** Locate the saved optimization folder on your computer and compress the entire directory into a single `.zip` archive.
            3. **Upload & Analyze:** Upload this `.zip` file using the uploader in the left sidebar. The engine will automatically parse and evaluate the data in-memory!
            """)
        else:
            st.info("👈 **Get Started:** Please enter the path to your optimization folders in the sidebar, or place them in the root directory.")
            st.markdown("""
            ### 💻 How to use the Local Analyzer:
            1. **Save Results:** In cTrader, save your optimization results once the backtesting process is complete.
            2. **Set Data Path:** Enter the absolute path of the saved optimization folder in the left sidebar configuration panel.
            3. **Scan & Analyze:** The engine will bypass memory limits and stream the results directly from your SSD/HDD!
            """)

    def _render_disclaimer(self):
        """Legal notice placed at the very bottom of the page."""
        st.divider()
        st.caption("""
        ⚠️ **Disclaimer:** This software is provided "AS IS" for educational and analytical purposes only. 
        It does not constitute financial advice. The author is not responsible for any financial losses incurred through the use of this software or strategies derived from its output. 
        Past performance in backtests is not indicative of future results.
        """)

    def run(self):
        st.title("🚀 cTrader Multi-Folder Optimization Analyzer")
        st.write("Automated parsing, filtering, and Machine Learning parameter research for cTrader backtest optimizations.")

        df, filters, mode_label = self._render_sidebar()

        if df is None or df.empty:
            st.error("❌ No valid test folders found at the specified path!")
            self._render_empty_state()
            return

        filtered_df = self.analyzer.filter_strategies(df, filters)
        self._render_metrics(len(df), filtered_df, mode_label)

        if not filtered_df.empty:
            self._render_charts(filtered_df)
            self._render_data_table(filtered_df)
            self._render_search_space_pruning(filtered_df)
            self._render_time_series(filtered_df)
            self._render_trend_volatility(filtered_df, df)
        else:
            self._render_filter_diagnostics(df, filters)

        self._render_disclaimer()
