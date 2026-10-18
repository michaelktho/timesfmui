# Digester H₂S forecasting webpage

Upload a daily CSV containing `date,airflow,temperature,h2s`. The historical rows must contain all four fields. At the end of the file add 1–30 consecutive future daily rows where `airflow` and `temperature` are planned/expected values, and `h2s` is empty. The page forecasts H₂S for these future rows using TimesFM 3.0 and displays a median prediction with its 10th–90th percentile band.

## Deploy without writing HTML

1. Sign in to GitHub, make a new repository named `digester-h2s-forecast`, and choose Public (or Private if your Streamlit account supports it).
2. Click **Add file → Upload files** and upload `app.py`, `requirements.txt`, `sample.csv`, and `README.md`. Click **Commit changes**.
3. Go to https://share.streamlit.io and log in with GitHub. Choose **Create app → Yup, I have an app**.
4. Choose the repository, `main` branch, and `app.py` file. In advanced settings, use Python 3.11 or 3.12. Click **Deploy**.
5. Open the public Streamlit URL and upload `sample.csv`, then click **Predict H₂S**.

**Important:** TimesFM 3.0 is a large pretrained model. Free Streamlit hosting may run out of RAM or be too slow on CPU. This project is a starting point, not a guarantee of deployability on a free plan. If you see memory or resource errors, deploy the same Python app to hosting with more memory; avoid repeated restarts.

**Model licence:** Downloaded/self-hosted TimesFM 3.0 weights have non-commercial/non-production restrictions. Review Google's licence before publishing or using the application: https://github.com/google-research/timesfm

**Data privacy:** Do not put real research measurements in a public GitHub repository. Visitors upload CSVs to the hosting service, so review its privacy and data processing arrangements for sensitive data.

**Research caveat:** Forecasts under planned airflow/temperature are observational predictions, not causal evidence. The quantile band is not a validated coverage guarantee.
