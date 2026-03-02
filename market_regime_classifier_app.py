import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import time
import io

st.set_page_config(
    page_title="Market Regime Classifier with Deep Learning",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📈"
)

# ==============================
# Sidebar - Parameters
# ==============================
st.sidebar.title("Settings")
lookback = st.sidebar.slider("Lookback Window (days)", 30, 252, 90, 10)
num_regimes = st.sidebar.slider("Number of Regimes", 2, 6, 3, 1)
lstm_units = st.sidebar.slider("LSTM Units", 16, 128, 64, 8)
dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)
learning_rate = st.sidebar.select_slider("Learning Rate", options=[0.001, 0.005, 0.01, 0.02], value=0.005)
epochs = st.sidebar.slider("Epochs", 10, 100, 30, 5)

# ==============================
# Data Input
# ==============================
st.title("Market Regime Classifier with Deep Learning")
st.markdown("""
#### Upload multi-asset price CSV or fetch live data
""")


st.markdown("**If no CSV is uploaded, demo live data will be loaded automatically.**")
uploaded_file = st.file_uploader("Upload CSV (Date, Asset1, Asset2, ...)", type=["csv"])
default_tickers = "AAPL,MSFT,GOOG,AMZN,TSLA"
live_tickers = st.text_input("Or enter comma-separated tickers for live data (Yahoo Finance)", default_tickers)
live_period = st.selectbox("Live Data Period", ["1y", "2y", "5y", "max"], index=0)
live_interval = st.selectbox("Live Data Interval", ["1d", "1wk", "1mo"], index=0)

@st.cache_data(show_spinner=False)
def load_data(uploaded_file, live_tickers, live_period, live_interval):
    df = None
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, parse_dates=[0])
        df = df.set_index(df.columns[0])
    else:
        tickers = [t.strip() for t in live_tickers.split(",") if t.strip()]
        df_raw = yf.download(tickers, period=live_period, interval=live_interval)
        # Try to get 'Adj Close', fallback to 'Close' or main df
        if not df_raw.empty:
            if "Adj Close" in df_raw.columns or (isinstance(df_raw.columns, pd.MultiIndex) and "Adj Close" in df_raw.columns.get_level_values(0)):
                try:
                    df = df_raw["Adj Close"]
                except Exception:
                    df = df_raw
            elif "Close" in df_raw.columns or (isinstance(df_raw.columns, pd.MultiIndex) and "Close" in df_raw.columns.get_level_values(0)):
                try:
                    df = df_raw["Close"]
                except Exception:
                    df = df_raw
            else:
                df = df_raw
            df = df.dropna(how="all")
    if df is None or df.empty:
        st.error("No data loaded. Please check tickers or try a different period/interval.")
        return pd.DataFrame()
    return df

data = load_data(uploaded_file, live_tickers, live_period, live_interval)
assets = data.columns.tolist()

# ==============================
# Feature Engineering
# ==============================
def compute_features(df, lookback):
    log_ret = np.log(df / df.shift(1)).dropna()
    vol = log_ret.rolling(lookback).std()
    corr = log_ret.rolling(lookback).corr().dropna()
    return log_ret, vol, corr

log_ret, vol, corr = compute_features(data, lookback)

# ==============================
# Prepare Model Data
# ==============================
def prepare_model_data(log_ret, vol, corr, lookback):
    # Flatten rolling correlation matrices
    corr_flat = []
    for t in range(lookback, len(log_ret)):
        cmat = corr.iloc[t * len(assets):(t+1) * len(assets)]
        vals = cmat.values.flatten()
        if vals.size == len(assets) * len(assets):
            cmat = vals.reshape(len(assets), len(assets))
            corr_flat.append(cmat[np.triu_indices(len(assets), 1)])
        else:
            # Not enough data, skip
            corr_flat.append(np.zeros(int(len(assets)*(len(assets)-1)/2)))
    corr_flat = np.array(corr_flat)
    # Features: log_ret, vol, corr_flat
    X = []
    for t in range(lookback, len(log_ret)):
        if t-lookback < len(corr_flat):
            x = np.concatenate([
                log_ret.iloc[t].values,
                vol.iloc[t].values,
                corr_flat[t-lookback]
            ])
            X.append(x)
    X = np.array(X)
    # Rolling windows
    X_seq = []
    for t in range(len(X) - lookback):
        X_seq.append(X[t:t+lookback])
    X_seq = np.array(X_seq)
    return X_seq

X_seq = prepare_model_data(log_ret, vol, corr, lookback)

# ==============================
# Deep Learning Model
# ==============================
def build_lstm(input_shape, num_regimes, lstm_units, dropout_rate, learning_rate):
    model = Sequential([
        LSTM(lstm_units, input_shape=input_shape, return_sequences=False),
        Dropout(dropout_rate),
        Dense(num_regimes, activation="softmax")
    ])
    model.compile(optimizer=Adam(learning_rate), loss="categorical_crossentropy")
    return model

# Dummy regime labels for unsupervised training (simulate with KMeans)
from sklearn.cluster import KMeans
if len(X_seq) > 0:
    kmeans = KMeans(n_clusters=num_regimes, random_state=42).fit(X_seq.reshape(len(X_seq), -1))
    y_seq = tf.keras.utils.to_categorical(kmeans.labels_, num_classes=num_regimes)
else:
    y_seq = np.zeros((0, num_regimes))

# Train/Test split
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# Model Training
if len(X_train) > 0:
    model = build_lstm(X_train.shape[1:], num_regimes, lstm_units, dropout_rate, learning_rate)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=16, verbose=0)
    regime_probs = model.predict(X_seq)
else:
    regime_probs = np.zeros((0, num_regimes))

# Smooth regime probabilities
def smooth_probs(probs, window=5):
    if len(probs) == 0:
        return probs
    return np.apply_along_axis(lambda m: np.convolve(m, np.ones(window)/window, mode='same'), axis=0, arr=probs)
regime_probs_smooth = smooth_probs(regime_probs, window=5)

# ==============================
# Visualization
# ==============================
st.markdown("---")
st.subheader("Regime Structure Visualization")

# 1) 3D Regime Surface
if len(regime_probs_smooth) > 0:
    time_axis = np.arange(len(regime_probs_smooth))
    regime_axis = np.arange(num_regimes)
    surface = go.Surface(
        z=regime_probs_smooth.T,
        x=time_axis,
        y=regime_axis,
        colorscale="Viridis",
        showscale=True,
        lighting=dict(ambient=0.7, diffuse=0.8, specular=0.5),
    )
    layout = go.Layout(
        title="3D Regime Probability Surface",
        scene=dict(
            xaxis_title="Time",
            yaxis_title="Regime",
            zaxis_title="Probability",
            bgcolor="#181818",
        ),
        paper_bgcolor="#181818",
        font=dict(color="#e0e0e0"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig_surface = go.Figure(data=[surface], layout=layout)
    st.plotly_chart(fig_surface, use_container_width=True)

# 2) Animated regime probability timeline
if len(regime_probs_smooth) > 0:
    df_probs = pd.DataFrame(regime_probs_smooth, columns=[f"Regime {i+1}" for i in range(num_regimes)])
    df_probs["Time"] = np.arange(len(df_probs))
    fig_anim = px.line(df_probs, x="Time", y=df_probs.columns[:-1],
                      title="Animated Regime Probability Timeline",
                      template="plotly_dark")
    fig_anim.update_layout(
        paper_bgcolor="#181818",
        font=dict(color="#e0e0e0"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_anim, use_container_width=True)

# 3) Correlation heatmap per regime
if len(regime_probs_smooth) > 0:
    st.subheader("Correlation Heatmap per Regime")
    for r in range(num_regimes):
        idx = np.argmax(regime_probs_smooth[:, r])
        corr_matrix = corr.iloc[idx * len(assets):(idx+1) * len(assets)]
        if isinstance(corr_matrix, pd.DataFrame):
            corr_matrix = corr_matrix.values.reshape(len(assets), len(assets))
        fig_heat = px.imshow(corr_matrix, x=assets, y=assets, color_continuous_scale="Viridis",
                            title=f"Regime {r+1} Correlation Heatmap")
        fig_heat.update_layout(
            paper_bgcolor="#181818",
            font=dict(color="#e0e0e0"),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# 4) Price chart colored by detected regime
if len(regime_probs_smooth) > 0:
    st.subheader("Price Chart Colored by Regime")
    detected_regime = np.argmax(regime_probs_smooth, axis=1)
    df_price = data.iloc[-len(detected_regime):]
    for asset in assets:
        fig_price = go.Figure()
        for r in range(num_regimes):
            mask = detected_regime == r
            fig_price.add_trace(go.Scatter(
                x=df_price.index[mask],
                y=df_price[asset][mask],
                mode="lines",
                name=f"Regime {r+1}",
                line=dict(width=2),
            ))
        fig_price.update_layout(
            title=f"{asset} Price by Regime",
            paper_bgcolor="#181818",
            font=dict(color="#e0e0e0"),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_price, use_container_width=True)

# ==============================
# Quant-style Metric Panels
# ==============================
st.markdown("---")
st.subheader("Regime Metrics")
if len(regime_probs_smooth) > 0:
    avg_probs = regime_probs_smooth.mean(axis=0)
    for r in range(num_regimes):
        st.metric(label=f"Regime {r+1} Avg Probability", value=f"{avg_probs[r]:.2f}")

st.markdown("""
---
*Institutional dark theme. All visualizations are interactive and animated. Backend: Numpy, TensorFlow, Plotly. Single-file Streamlit app.*
""")
