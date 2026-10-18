# TimesFM Forecasting Studio

Streamlit webpage for forecasting **any regularly spaced numeric time series** using TimesFM 3.0. No HTML needed.

## Files
- `app.py`: complete website
- `requirements.txt`: Python packages
- `sample.csv`: example daily measurements (last 7 days are future rows with blank target values)
- `README.md`: these instructions

## Update your existing GitHub/Streamlit website
1. Download and extract the ZIP. Keep a copy of the old project as a backup.
2. Open your existing GitHub repository. Replace the four root-level files with these four files (GitHub's **Add file → Upload files**, or edit/delete individually). Do not upload the ZIP alone.
3. **Commit changes**. Your existing Streamlit Community Cloud deployment should use the updated `app.py` and `requirements.txt` once redeployed.
4. If dependencies do not update, use **Reboot app** (and optionally clear cache) from Streamlit management. Use Python 3.11 or 3.12 if you need to set Python version in advanced settings.

## CSV rules
- A date/time column and at least one target column. Select both on the webpage. Use ISO dates (`YYYY-MM-DD` or ISO datetimes) and chronological order.
- Choose daily, hourly, weekly (7-day spacing) or monthly-start sampling. Times must be regularly spaced with no duplicate/missing timestamps.
- At least **14 consecutive historical target values**. To supply future known covariates, append future rows with **blank target cells**.
- **Past-only features:** numeric values for every historical row; future values not required.
- **Known future features:** numeric values covering both history and the next forecast-horizon rows. They must be genuinely known or assumed, not future observations unavailable at prediction time.
- You can leave both feature lists blank for ordinary univariate forecasting; no future rows are then required.
- The website forecasts one chosen target at a time. Other target columns can be treated as features only when their time availability permits it.

## Forecasting and validation
- Model: `google/timesfm-3.0-pytorch`, loaded once and cached, using CUDA if available, otherwise CPU.
- The shaded region shows the model's 10th to 90th predicted percentiles, not a validated confidence interval.
- The optional backtest withholds the last `horizon` historical target values and reports MAE, RMSE and empirical percentile coverage. When known-future features are selected, the backtest uses their actually recorded values for the held-out period (a conditional evaluation).
- The uploaded data are processed by the Streamlit host; do not use confidential datasets unless the host and access controls are appropriate.

## Deployment limitations
- TimesFM 3.0 (~330M parameters) may exceed free cloud memory or run slowly on CPU. If that happens, use a host with more memory or GPU.
- Google distributes the 3.0 pretrained weights under a **non-commercial, non-production** licence. Check the official licence before exposing the app publicly or using it commercially: https://github.com/google-research/timesfm
- Forecasts may be inaccurate on unfamiliar processes. Backtest before relying on them.
