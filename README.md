# FEMA Events API (FastAPI Backend)

This microservice powers the CommodiSense dashboard.
It predicts how natural disasters (e.g., hurricanes, floods, wildfires)
impact commodity prices using a Random Forest Regressor trained on FEMA and Yahoo Finance data.

## Run Locally
```bash
pip install -r requirements.txt
uvicorn fema_api:app --reload --port 8000
```

## Endpoints
- `/predict` → returns likelihoods for price movement  
- `/analogs` → returns historical event matches
