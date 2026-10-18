import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Digester H₂S Forecast", page_icon="📈")
st.title("Digester H₂S Forecast")
st.write("Upload daily measurements to forecast H₂S with TimesFM 3.0. Future airflow and temperature must be supplied.")
st.caption("For research and demonstration, not operational safety decisions.")

@st.cache_resource(show_spinner="Loading TimesFM model (first run may take some time)…")
def load_model():
    from timesfm3 import TimesFM3Evaluator, ModelConfig
    return TimesFM3Evaluator(ModelConfig(checkpoint_path="google/timesfm-3.0-pytorch", per_core_batch_size=1, device="cpu"))

file = st.file_uploader("Choose a CSV file", type="csv")
st.download_button("Download example CSV", data=open("sample.csv", "rb").read(), file_name="sample.csv", mime="text/csv")
if file:
    try:
        df = pd.read_csv(file)
        needed = ["date", "airflow", "temperature", "h2s"]
        if not set(needed).issubset(df.columns): raise ValueError("CSV must have columns: date, airflow, temperature, h2s")
        df = df[needed].copy(); df["date"] = pd.to_datetime(df["date"], errors="raise")
        if df["date"].isna().any() or df["date"].duplicated().any(): raise ValueError("Dates must be unique and valid.")
        df = df.sort_values("date").reset_index(drop=True)
        if not df["date"].diff().iloc[1:].eq(pd.Timedelta(days=1)).all(): raise ValueError("Dates must be consecutive daily observations without gaps.")
        for col in ["airflow", "temperature", "h2s"]: df[col] = pd.to_numeric(df[col], errors="coerce")
        n = df.h2s.notna().sum(); h = len(df)-n
        if n < 14 or h < 1 or h > 30 or df.h2s.iloc[:n].isna().any() or df.h2s.iloc[n:].notna().any():
            raise ValueError("Provide at least 14 measured days, followed by 1–30 future days with blank h2s.")
        if not np.isfinite(df[["airflow","temperature"]].to_numpy()).all() or not np.isfinite(df.h2s.iloc[:n]).all() or (df.h2s.iloc[:n] < 0).any():
            raise ValueError("Airflow and temperature must be available for every day; measured H₂S must be finite and non-negative.")
        st.write(f"**{n} measured days** | **{h} forecast days**")
        st.dataframe(df.tail(min(len(df), 12)), hide_index=True)
        if st.button("Predict H₂S", type="primary"):
            with st.spinner("Forecasting…"):
                y = df.h2s.iloc[:n].to_numpy(dtype=np.float32)
                x = df[["airflow", "temperature"]].to_numpy(dtype=np.float32).T
                o = next(load_model().predict_batch(contexts=[y], horizon=h, past_future_covariates=[x], return_quantiles=True, use_symmetric_averaging=False, make_positive=True, sort_quantiles=True))
                q = np.asarray(o.quantiles); median = np.maximum(np.asarray(o.forecast), 0)
                low, high = np.maximum(q[:, 0], 0), np.maximum(q[:, -1], 0)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df.date.iloc[:n], y, label="Measured H₂S", color="teal")
            ax.plot(df.date.iloc[n:], median, "o-", label="Median forecast", color="royalblue")
            ax.fill_between(df.date.iloc[n:], low, high, color="royalblue", alpha=.2, label="10th–90th percentiles")
            ax.set(xlabel="Date", ylabel="H₂S (same units as CSV)"); ax.legend(); fig.autofmt_xdate(); st.pyplot(fig)
            result = pd.DataFrame({"date": df.date.iloc[n:].dt.strftime("%Y-%m-%d"), "airflow": df.airflow.iloc[n:], "temperature": df.temperature.iloc[n:], "forecast_h2s": median, "p10": low, "p90": high})
            st.dataframe(result, hide_index=True)
            st.download_button("Download predictions CSV", result.to_csv(index=False).encode(), "h2s_predictions.csv", "text/csv")
            st.caption("The 10th–90th percentile band is a model quantile range, not a validated 80% coverage guarantee. Validate against held-out digester data before use.")
    except Exception as e:
        st.error(f"Could not process the file or generate the forecast: {e}")
