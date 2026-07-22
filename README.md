# 🚀 cTrader Optimization Analyzer

A robust, object-oriented Python Streamlit dashboard designed to scan, parse, and analyze cTrader optimization folders. This tool allows algorithmic traders to find the **Sweet Spot** in their automated trading strategies using Machine Learning (Random Forest) and interactive visualizations.

## 🎯 Use Case & Industry Context
Finding robust parameters for an algorithmic trading strategy is computationally expensive and prone to data-mining bias (overfitting). When you run an optimization process in cTrader across thousands of parameter combinations, finding the true "sweet spots" (areas where parameters are robust, not just isolated lucky peaks) requires deep analysis.

This tool solves this by:
1. **Rapid JSON Parsing:** Bypasses heavy HTML processing by extracting pure JSON payloads directly from cTrader's backtest output.
2. **Search Space Pruning:** Uses a Random Forest Regressor to identify which parameters actually move the needle on Net Profit.
3. **Elite Cluster Analysis:** Recommends precise parameter ranges based on the top 25% of performing models, allowing you to narrow your next optimization run and save hours of CPU time.

## 🏗️ Architecture

The project has been refactored into a scalable Object-Oriented structure, preparing it for future Internationalization (i18n) and cloud deployment:

- `app.py` - The main entry point.
- `src/scanner.py` - Handles I/O, regex, JSON parsing, and Parquet/CSV caching.
- `src/analyzer.py` - Contains the business logic, Machine Learning (scikit-learn), and statistical pruning.
- `src/dashboard.py` - The Streamlit UI layer handling states, charts (Plotly), and rendering.
- `src/utils.py` - CLI argument parsing.

## 🛠️ Installation & Usage

1. **Clone the repository**
2. **Init virtual environment**
   ```bash
   python -m venv .venv
   ```
2. **Activate venv**
- Unix systems:
   ```bash
   source .venv/bin/activate
   ```
- Windows:
   ```bash
   source .venv\Scripts\activate
   ```
2. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. **Run the dashboard:**
   ```bash
   streamlit run app.py -- -d "/path/to/your/ctrader/optimization/folders"
   ```

