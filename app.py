import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="TimesFM Forecasting Studio", layout="wide")
st.title("TimesFM Forecasting Studio")
st.caption("Upload a regular time series, select a target and optional numerical covariates, then forecast using TimesFM 3.0.")

@st.cache_resource(show_spinner="Loading TimesFM 3.0 (first run downloads the weights)…")
def load_model():
    import torch
    from timesfm3 import TimesFM3Evaluator, ModelConfig
    return TimesFM3Evaluator(ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch", device="cuda" if torch.cuda.is_available() else "cpu"))

def forecast(model, y, horizon, past=None, known=None, positive=False):
    kw = {"contexts": [np.asarray(y, np.float32)], "horizon": horizon,
          "return_quantiles": True, "use_symmetric_averaging": False, "make_positive": positive}
    if past is not None: kw["past_only_covariates"] = [np.asarray(past, np.float32)]
    if known is not None: kw["past_future_covariates"] = [np.asarray(known, np.float32)]
    return next(iter(model.predict_batch(**kw)))

def plot(history_dates, y, future_dates, o, target, actual=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_dates, y=y, mode="lines", name="History"))
    fig.add_trace(go.Scatter(x=future_dates, y=o.quantiles[:, 8], mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=future_dates, y=o.quantiles[:, 0], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(60,120,210,0.20)", name="P10–P90"))
    fig.add_trace(go.Scatter(x=future_dates, y=o.forecast, mode="lines+markers", name="Median forecast"))
    if actual is not None: fig.add_trace(go.Scatter(x=future_dates, y=actual, mode="lines+markers", name="Observed (held-out)"))
    fig.update_layout(xaxis_title="Time", yaxis_title=target, hovermode="x unified", legend=dict(orientation="h"), margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

f = st.file_uploader("Upload CSV", type="csv")
if f is None:
    st.info("Start with the included sample.csv. Future target values may be blank; future covariates need values when selected.")
    st.stop()
try:
    df = pd.read_csv(f)
    if df.empty or len(df.columns) < 2: raise ValueError("CSV needs at least a time column and a target column.")
    st.dataframe(df.head(8), use_container_width=True)
    time_col = st.selectbox("Date/time column", df.columns)
    target = st.selectbox("Target column", [c for c in df.columns if c != time_col])
    dates = pd.to_datetime(df[time_col], errors="raise", dayfirst=False)
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing: raise ValueError("Dates must be valid, unique and in increasing order. Use ISO dates: YYYY-MM-DD.")
    freq_options = {"Daily": "D", "Hourly": "h", "Weekly": "7D", "Monthly (start)": "MS"}
    inferred = pd.infer_freq(dates) if len(dates) >= 3 else None
    frequency = st.selectbox("Sampling interval", list(freq_options), index=next((i for i, v in enumerate(freq_options.values()) if v == inferred), 0))
    freq = freq_options[frequency]
    expected = pd.date_range(dates.iloc[0], periods=len(dates), freq=freq)
    if not np.array_equal(expected.to_numpy(), dates.to_numpy()): raise ValueError("Dates are not regularly spaced at this selected frequency. Correct the CSV or select a different interval.")
    y_full = pd.to_numeric(df[target], errors="coerce")
    n = int(y_full.notna().sum())
    if n < 14 or not y_full.iloc[:n].notna().all() or y_full.iloc[n:].notna().any():
        raise ValueError("Provide at least 14 consecutive measured target values, optionally followed by blank future target rows. No gaps inside history.")
    trailing = len(df) - n
    horizon = st.number_input("Forecast horizon (time steps)", min_value=1, max_value=256, value=min(trailing or 7, 256))
    horizon = int(horizon)
    columns = [c for c in df.columns if c not in (time_col, target)]
    past_cols = st.multiselect("Past-only numerical features (optional)", columns)
    known_cols = st.multiselect("Known future numerical features (optional)", [c for c in columns if c not in past_cols])
    if set(past_cols) & set(known_cols): raise ValueError("A feature cannot belong to both groups.")
    if known_cols and trailing < horizon: raise ValueError(f"Future covariates require {horizon} future rows with blank target values. This CSV contains {trailing}.")
    if trailing and trailing != horizon: st.info("The forecast uses the selected horizon. Extra blank future rows are ignored, or predicted dates are extended if necessary.")
    validation = st.checkbox("Backtest on the last historical observations", value=False)
    positive = st.checkbox("Restrict forecast to non-negative values (for non-negative history)", value=False)
    numeric = df[columns].apply(pd.to_numeric, errors="coerce") if columns else pd.DataFrame(index=df.index)
    for c in past_cols:
        if numeric[c].iloc[:n].isna().any(): raise ValueError(f"'{c}' contains missing/non-numerical historical values.")
    for c in known_cols:
        if numeric[c].iloc[:n+horizon].isna().any(): raise ValueError(f"'{c}' must have numeric values throughout history and the next {horizon} steps.")
    if st.button("Run forecast", type="primary"):
        with st.spinner("Generating forecast…"):
            model = load_model()
            y = y_full.iloc[:n].to_numpy(np.float32)
            po = numeric[past_cols].iloc[:n].to_numpy(np.float32).T if past_cols else None
            pf = numeric[known_cols].iloc[:n+horizon].to_numpy(np.float32).T if known_cols else None
            o = forecast(model, y, horizon, po, pf, positive)
            future_dates = pd.date_range(dates.iloc[n-1], periods=horizon+1, freq=freq)[1:]
            result = pd.DataFrame({"date": future_dates, "forecast": o.forecast, "p10": o.quantiles[:, 0], "p90": o.quantiles[:, 8]})
            plot(dates.iloc[:n], y, future_dates, o, target)
            st.download_button("Download forecast CSV", result.to_csv(index=False), "forecast.csv", "text/csv")
            st.dataframe(result, use_container_width=True)
            if validation:
                if n - horizon < 14: raise ValueError("Backtesting requires at least 14 historical observations before the held-out horizon. Reduce the horizon or provide more history.")
                k = n - horizon
                po_test = numeric[past_cols].iloc[:k].to_numpy(np.float32).T if past_cols else None
                pf_test = numeric[known_cols].iloc[:n].to_numpy(np.float32).T if known_cols else None
                bt = forecast(model, y[:k], horizon, po_test, pf_test, positive)
                actual = y[k:]
                mae = float(np.mean(np.abs(actual-bt.forecast)))
                rmse = float(np.sqrt(np.mean((actual-bt.forecast)**2)))
                coverage = float(np.mean((actual >= bt.quantiles[:, 0]) & (actual <= bt.quantiles[:, 8]))*100)
                st.subheader("Historical backtest")
                st.caption("Train on earlier observations; predict the last measured steps. Known-future covariates use values from the held-out period.")
                a, b, c = st.columns(3)
                a.metric("MAE", f"{mae:.3g}"); b.metric("RMSE", f"{rmse:.3g}"); c.metric("P10–P90 coverage", f"{coverage:.1f}%")
                plot(dates.iloc[:k], y[:k], dates.iloc[k:n], bt, target, actual)
except Exception as e:
    st.error(f"Cannot process this dataset: {e}")

st.caption("Research demonstration only. Quantile ranges are model estimates, not guaranteed calibrated prediction intervals. TimesFM 3.0 weights have non-commercial, non-production restrictions.")
