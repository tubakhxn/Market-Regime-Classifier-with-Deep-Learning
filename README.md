# Market Regime Classifier with Deep Learning

## Overview
This project is an advanced Streamlit application for classifying market regimes using deep learning. It analyzes multi-asset price data, computes financial features, and applies a deep sequence model (LSTM) to detect and visualize market regimes interactively.

### Features
- Upload CSV or fetch live data from Yahoo Finance
- Computes log returns, rolling volatility, and rolling correlation matrices
- Deep learning model (LSTM) for regime classification
- Interactive, animated 3D regime surface, regime timeline, correlation heatmaps, and price charts (Plotly, dark institutional theme)
- Sidebar for model/data parameters
- Quant-style metric panels

## How to Run
1. Clone or fork this repository:
   ```sh
   git clone https://github.com/tubakhxn/market-regime-classifier.git
   ```
   Or fork using the GitHub UI (top right corner).
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the app:
   ```sh
   streamlit run market_regime_classifier_app.py
   ```

## Creator/Developer
**tubakhxn**

## Forking Instructions
- Click the "Fork" button on the top right of the GitHub repository page.
- Clone your fork locally and follow the run instructions above.

---
For questions or contributions, contact tubakhxn.
