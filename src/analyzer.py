import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from typing import Optional, List, Dict, Any

class StrategyAnalyzer:
    """
    Responsible for data filtering, statistics, machine learning, 
    and parameter space pruning.
    """

    def __init__(self, df: pd.DataFrame):
        self.raw_df = df

    def filter_strategies(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Applies strict filtering conditions on active columns."""
        if df is None or df.empty:
            return pd.DataFrame()

        cond = (
            (df["Total Trades"] >= filters["min_trades"]) &
            (df["Net Profit"] >= filters["profit_range"][0]) &
            (df["Net Profit"] <= filters["profit_range"][1]) &
            (df["Win Rate"] >= filters["winrate_range"][0]) &
            (df["Win Rate"] <= filters["winrate_range"][1]) &
            (df["Equity Drawdown"] <= filters["max_dd"]) &
            (df["Profit Factor"] >= filters["pf_range"][0]) &
            (df["Profit Factor"] <= filters["pf_range"][1]) &
            (df["Swaps"] >= filters["swaps_range"][0]) &
            (df["Swaps"] <= filters["swaps_range"][1]) &
            (df["Max Cons. Losses"] <= filters["max_cons_losses"]) &
            (df["Largest Win"] >= filters["largest_win_range"][0]) &
            (df["Largest Win"] <= filters["largest_win_range"][1]) &
            (df["Largest Loss"] >= filters["largest_loss_range"][0]) &
            (df["Largest Loss"] <= filters["largest_loss_range"][1]) &
            (df["Average Trade"] >= filters["avg_trade_range"][0]) &
            (df["Average Trade"] <= filters["avg_trade_range"][1])
        )
        return df[cond]

    def get_parameter_columns(self, df: pd.DataFrame) -> List[str]:
        ignore_prefixes = [
            "Test_ID", "Total Trades", "Winning Trades", "Losing Trades", 
            "Profit Factor", "Equity Drawdown", "Net Profit", "Win Rate", 
            "Swaps", "Max Cons. Losses", "Largest Win", "Largest Loss", "Average Trade"
        ]
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        valid_cols = []
        for c in num_cols:
            if any(c.startswith(prefix) for prefix in ignore_prefixes):
                continue
            if df[c].nunique() > 1:
                valid_cols.append(c)
        return valid_cols

    def calculate_feature_importance(
        self, df: pd.DataFrame, param_cols: List[str], target: str = "Net Profit"
    ) -> Optional[pd.DataFrame]:
        if len(df) < 10 or not param_cols:
            return None

        X = df[param_cols].fillna(0)
        y = df[target].fillna(0)

        rf = RandomForestRegressor(n_estimators=150, random_state=42)
        rf.fit(X, y)

        importance_df = pd.DataFrame(
            {"Parameter": X.columns, "Relevance (Weight %)": np.round(rf.feature_importances_ * 100, 2)}
        ).sort_values(by="Relevance (Weight %)", ascending=True)

        return importance_df

    def generate_pruning_report(
        self, df: pd.DataFrame, param_cols: List[str], target_metric: str = "Net Profit", elite_percentile: float = 0.25
    ) -> pd.DataFrame:
        if df.empty or not param_cols:
            return pd.DataFrame()

        threshold = df[target_metric].quantile(1.0 - elite_percentile)
        elite_df = df[df[target_metric] >= threshold]

        stats = []
        for col in param_cols:
            s_min = elite_df[col].min()
            s_q25 = elite_df[col].quantile(0.25)
            s_med = elite_df[col].median()
            s_mean = elite_df[col].mean()
            s_q75 = elite_df[col].quantile(0.75)
            s_max = elite_df[col].max()
            s_std = elite_df[col].std()

            overall_range = max(1e-5, df[col].max() - df[col].min())
            iqr_range = s_q75 - s_q25
            concentration = round((1.0 - (iqr_range / overall_range)) * 100, 1)

            stats.append(
                {
                    "Parameter": col,
                    "Recommended Min (Q25)": round(s_q25, 2),
                    "Average": round(s_mean, 2),
                    "Median": round(s_med, 2),
                    "Recommended Max (Q75)": round(s_q75, 2),
                    "Standard Dev (±)": round(s_std, 2),
                    "Absolute Min": round(s_min, 2),
                    "Absolute Max": round(s_max, 2),
                    "Concentration (%)": f"{max(0.0, concentration)}% tightness",
                }
            )

        return pd.DataFrame(stats)
