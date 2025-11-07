from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import yfinance as yf
import numpy as np
import warnings
from yfinance.shared import _ERRORS
import os
import joblib

# Suppress annoying FutureWarnings from yfinance
warnings.filterwarnings("ignore", message="YF.download")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
fema_df = pd.read_csv(os.path.join(BASE_DIR, "fema_disasters.csv"), low_memory=False)


app = FastAPI()


BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "fema_ng_model.pkl")
model = joblib.load(model_path)


# Allow local frontend to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------
# Request body schema
# -------------------------------
class Input(BaseModel):
    eventType: str
    duration: float
    counties: int
    state: str


# -------------------------------
# /analogs endpoint
# -------------------------------
@app.post("/analogs")
def get_analogs(data: Input):
    df = fema_df.copy()

    # 1️⃣ Match by incident type (case-insensitive)
    df = df[df["incidentType"].str.contains(data.eventType, case=False, na=False)]

    # 2️⃣ Filter by state if available
    if data.state:
        df = df[df["state"].str.upper() == data.state.upper()]

    # 3️⃣ Compute duration of each disaster using incidentBeginDate / incidentEndDate
    # Group by incidentId for multi-county disasters
    group_col = "incidentId"

    # Convert date columns properly
    df["incidentBeginDate"] = pd.to_datetime(df["incidentBeginDate"], errors="coerce")
    df["incidentEndDate"] = pd.to_datetime(df["incidentEndDate"], errors="coerce")

    # Drop rows missing both start and end dates
    df = df.dropna(subset=["incidentBeginDate", "incidentEndDate"], how="all")

    # Compute start/end per incident
    durations = (
        df.groupby(group_col)
        .agg(
            start_date=("incidentBeginDate", "min"),
            end_date=("incidentEndDate", "max"),
            incidentType=("incidentType", "first"),
            state=("state", lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
            designatedArea=("designatedArea", lambda x: ", ".join(sorted(set(x.dropna())))),
            counties_impacted=("fipsCountyCode", pd.Series.nunique),
        )
        .reset_index()
    )

    # Calculate event duration in days (ensure minimum of 1)
    durations["duration_days"] = (
        (durations["end_date"] - durations["start_date"]).dt.days.fillna(0).clip(lower=1)
    )

    # 4️⃣ Apply similarity filters
    durations["duration_diff"] = abs(durations["duration_days"] - data.duration)
    durations["counties_diff"] = abs(durations["counties_impacted"] - data.counties)

    similar = durations[
        (durations["duration_diff"] <= 10) & (durations["counties_diff"] <= 5)
    ].copy()

    # 5️⃣ Similarity scoring
    similar["similarity"] = 1 / (
        1 + similar["duration_diff"] + similar["counties_diff"]
    )

    # 6️⃣ Add price change data for each historical analog
    symbol = (
        "NG=F"
        if "Gas" in data.eventType or "Hurricane" in data.eventType
        else "CL=F"
    )

    results = []
    for _, row in similar.sort_values("similarity", ascending=False).head(5).iterrows():
        start, end = row["start_date"], row["end_date"]
        if pd.isna(start) or pd.isna(end):
            continue
        price_change = None
        try:
            # Expand the date window ±7 days for missing market data
            fetch_start = start - pd.Timedelta(days=7)
            fetch_end = end + pd.Timedelta(days=7)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                price_df = yf.download(
                    symbol,
                    start=fetch_start,
                    end=fetch_end,
                    interval="1d",
                    progress=False,
                    auto_adjust=True,
                )

            if not price_df.empty:
                # Pick the nearest trading day to event start and end
                first_close = price_df.loc[
                    price_df.index >= start, "Close"
                ].iloc[0] if (price_df.index >= start).any() else price_df["Close"].iloc[0]

                last_close = price_df.loc[
                    price_df.index <= end, "Close"
                ].iloc[-1] if (price_df.index <= end).any() else price_df["Close"].iloc[-1]

                price_change = ((last_close - first_close) / first_close) * 100

        except Exception as e:
            print(f"[WARN] Could not fetch prices for {symbol} ({start.date()}–{end.date()}): {e}")
            price_change = None


        results.append(
            {
                "incidentType": row["incidentType"],
                "state": row["state"],
                "Start Date": str(row["start_date"].date()),
                "End Date": str(row["end_date"].date()),
                "Area": row["designatedArea"],
                "# of Counties": int(row["counties_impacted"]),
                "duration_days": int(row["duration_days"]),
                "price_change": round(price_change, 2)
                if price_change is not None
                else None,
            }
        )

    return results


# -------------------------------
# /predict endpoint
# -------------------------------
@app.post("/predict")
def predict(data: Input):
    # --- Determine commodity symbol ---
    if "gas" in data.eventType.lower():
        symbol = "NG=F"
    elif "oil" in data.eventType.lower():
        symbol = "CL=F"
    elif "hurricane" in data.eventType.lower():
        # hurricanes tend to impact natural gas supply first
        symbol = "NG=F"
    else:
        symbol = "CL=F"

    # --- Robust price fetch: auto-adjust, handle empty Yahoo responses ---
    try:
        df = yf.download(
            symbol,
            period="60d",           # grab a bit more context
            interval="1d",
            progress=False,
            auto_adjust=True,
        )

        if df.empty or "Close" not in df.columns:
            raise ValueError(f"No valid data returned for {symbol}")

    except Exception as e:
        print(f"[WARN] Could not fetch Yahoo data for {symbol}: {e}")
        # fallback: small synthetic dataframe to avoid model crash
        df = pd.DataFrame(
            {"Close": np.linspace(2, 3, 45)},
            index=pd.date_range(end=pd.Timestamp.today(), periods=45),
        )

    # --- Calculate returns & rolling volatility ---
    df["return"] = df["Close"].pct_change()
    df["volatility"] = df["return"].rolling(7).std()

    NG_Pre7_Return = df["return"].tail(7).sum() * 100
    NG_Pre7_Vol = df["volatility"].tail(7).mean() * 100
    NG_Pre30_Return = df["return"].tail(30).sum() * 100
    NG_Pre30_Vol = df["volatility"].tail(30).mean() * 100

    # --- Feature engineering ---
    sev_type = 1.0 if "Hurricane" in data.eventType else 0.8
    type_relative_severity = (data.duration / 5 + data.counties / 50) / 2

    X = pd.DataFrame([{
        "incidentType": data.eventType,
        "type_relative_severity": type_relative_severity,
        "duration_days": data.duration,
        "counties_impacted": data.counties,
        "NG_Pre7_Return": NG_Pre7_Return,
        "NG_Pre7_Vol": NG_Pre7_Vol,
        "NG_Pre30_Return": NG_Pre30_Return,
        "NG_Pre30_Vol": NG_Pre30_Vol,
        "Severity_Type_Interaction": type_relative_severity * sev_type,
    }])

    # --- Align columns with model ---
    expected_cols = getattr(model, "feature_names_in_", X.columns)
    for col in expected_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[expected_cols]

    # --- Predict & convert to probability ---
    prediction = float(model.predict(X)[0])
    prob_up = float(1 / (1 + np.exp(-prediction * 10)))  # smooth sigmoid

    return {
        "predicted_return": prediction,
        "prob_up": prob_up,
        "prob_down": 1 - prob_up,
        "symbol": symbol,
        "live_features": {
            "NG_Pre7_Return": NG_Pre7_Return,
            "NG_Pre7_Vol": NG_Pre7_Vol,
            "NG_Pre30_Return": NG_Pre30_Return,
            "NG_Pre30_Vol": NG_Pre30_Vol,
        },
    }
