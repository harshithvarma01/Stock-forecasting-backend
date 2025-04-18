import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DateRangeRequest(BaseModel):
    start_date: str
    end_date: str

def clean_data(df):
    for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna().reset_index(drop=True)
    return df

def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df[~df['Close'].astype(str).str.contains("AMZN", na=False)]
    df = clean_data(df)
    return df

def preprocess_data(df):
    X = df[['Close', 'High', 'Low', 'Open', 'Volume']].values
    y = df['Close'].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

def train_model(df):
    X, y, scaler = preprocess_data(df)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    with open("random_forest_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    return model, scaler

def load_model_from_disk():
    global model, scaler
    with open("random_forest_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

def forecast_stock(model, scaler, df, days):
    last_row = df.iloc[-1]
    last_data = np.array([[last_row['Close'], last_row['High'], last_row['Low'], last_row['Open'], last_row['Volume']]])
    scaled_data = scaler.transform(last_data)
    
    forecast = []
    # Calculate historical stats from the last 10 days
    price_changes = df['Close'].iloc[-10:].pct_change().dropna()
    avg_daily_change = price_changes.mean()  # Average percentage change
    std_daily_change = price_changes.std()   # Standard deviation for volatility
    
    current_price = last_row['Close']
    min_price = current_price  # Set minimum price to the last known close
    
    for day in range(days):
        next_day_pred = model.predict(scaled_data)
        base_price = next_day_pred[0]
        
        # Apply trend with volatility (random walk)
        daily_change = avg_daily_change + np.random.normal(0, std_daily_change)
        new_close = current_price * (1 + daily_change)
        
        # Ensure price doesn't drop below the starting value
        new_close = max(new_close, min_price)
        
        # Apply a slight upward correction every 5 days for long-term growth
        if day % 5 == 0 and day > 0:
            new_close = max(new_close, current_price * 1.001)
        
        new_high = new_close * (1 + 0.02 * np.random.normal(0, 0.5))  # 2% variation
        new_low = new_close * (1 - 0.02 * np.random.normal(0, 0.5))    # 2% variation
        new_open = (new_high + new_low) / 2
        new_volume = max(0, last_data[0][4] * (0.95 + np.random.normal(0, 0.05)))  # Slight volume fluctuation
        
        last_data[0] = [new_close, new_high, new_low, new_open, new_volume]
        scaled_data = scaler.transform(last_data)
        forecast.append(new_close)
        current_price = new_close
    
    return forecast
@app.post("/forecast")
async def forecast(data: DateRangeRequest):
    if model is None or scaler is None or df is None:
        return {"error": "Model or data not loaded."}
    try:
        start_date = datetime.strptime(data.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(data.end_date, "%Y-%m-%d")
        days = (end_date - start_date).days + 1
        if days <= 0:
            return {"error": "End date must be after start date."}
        forecasted_prices = forecast_stock(model, scaler, df, days)
        forecast_dates = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]
        return {"forecasted_prices": forecasted_prices, "forecast_dates": forecast_dates}
    except Exception as e:
        return {"error": str(e)}

file_path = 'Amazon.csv'
try:
    df = load_data(file_path)
    if not (os.path.exists("random_forest_model.pkl") and os.path.exists("scaler.pkl")):
        model, scaler = train_model(df)
    else:
        load_model_from_disk()
except Exception as e:
    df, model, scaler = None, None, None
    print(f"Startup error: {e}")
