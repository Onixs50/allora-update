# allora-update


 
 ```bash
    cd $HOME && cd basic-coin-prediction-node
    docker compose down
 ```
 ```bash
   nano model.py
  ```
# change it
```bash
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from zipfile import ZipFile
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from updater import download_binance_monthly_data, download_binance_daily_data
from config import data_base_path, model_file_path, predictions_file_path

binance_data_path = os.path.join(data_base_path, "binance/futures-klines")
training_price_data_path = os.path.join(data_base_path, "eth_price_data.csv")

def download_data():
    cm_or_um = "um"
    symbols = ["ETHUSDT"]
    intervals = ["1d"]
    years = ["2020", "2021", "2022", "2023", "2024"]
    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    download_path = binance_data_path
    
    download_binance_monthly_data(cm_or_um, symbols, intervals, years, months, download_path)
    print(f"Downloaded monthly data to {download_path}.")
    
    current_datetime = datetime.now()
    current_year = current_datetime.year
    current_month = current_datetime.month
    download_binance_daily_data(cm_or_um, symbols, intervals, current_year, current_month, download_path)
    print(f"Downloaded daily data to {download_path}.")

def format_data():
    files = sorted([x for x in os.listdir(binance_data_path)])

    if len(files) == 0:
        print("No files found to process.")
        return

    price_df = pd.DataFrame()
    for file in files:
        zip_file_path = os.path.join(binance_data_path, file)

        if not zip_file_path.endswith(".zip"):
            continue

        with ZipFile(zip_file_path) as myzip:
            with myzip.open(myzip.filelist[0]) as f:
                line = f.readline()
                header = 0 if line.decode("utf-8").startswith("open_time") else None
            df = pd.read_csv(myzip.open(myzip.filelist[0]), header=header).iloc[:, :11]
            df.columns = [
                "start_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "end_time",
                "volume_usd",
                "n_trades",
                "taker_volume",
                "taker_volume_usd",
            ]
            df.index = [pd.Timestamp(x + 1, unit="ms") for x in df["end_time"]]
            df.index.name = "date"
            price_df = pd.concat([price_df, df])

    price_df.sort_index().to_csv(training_price_data_path)
    print(f"Formatted data saved to {training_price_data_path}.")

def train_model():
    price_data = pd.read_csv(training_price_data_path)
    df = pd.DataFrame()
    
    df["date"] = pd.to_datetime(price_data["date"])
    df["date"] = df["date"].map(pd.Timestamp.timestamp)

    df["price"] = price_data[["open", "close", "high", "low"]].mean(axis=1)

    x = df["date"].values.reshape(-1, 1)
    y = df["price"].values.reshape(-1, 1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "SVR": SVR(kernel='rbf'),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=0),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=0),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=100, random_state=0)
    }

    best_model = None
    best_score = -float("inf")
    for name, model in models.items():
        model.fit(x_train_scaled, y_train)
        y_pred = model.predict(x_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{name}: Mean Squared Error = {mse}, R^2 Score = {r2}")
        
        if r2 > best_score:
            best_score = r2
            best_model = model

    os.makedirs(os.path.dirname(model_file_path), exist_ok=True)

    with open(model_file_path, "wb") as f:
        pickle.dump(best_model, f)

    print(f"Trained model saved to {model_file_path}")

    y_best_pred = best_model.predict(x_test_scaled)
    predictions_df = pd.DataFrame({
        'Date': pd.to_datetime(df.loc[x_test.flatten().astype(int).astype(np.int64)].index, unit='ms'),
        'Actual Price': y_test.flatten(),
        'Predicted Price': y_best_pred.flatten()
    })
    predictions_df.to_csv(predictions_file_path, index=False)
    print(f"Predictions saved to {predictions_file_path}")

    plt.figure(figsize=(14, 7))
    plt.plot(predictions_df['Date'], predictions_df['Actual Price'], color='blue', label='Actual Price')
    plt.plot(predictions_df['Date'], predictions_df['Predicted Price'], color='red', linestyle='--', label='Predicted Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Actual vs Predicted Prices')
    plt.legend()
    plt.savefig("price_predictions.png")
    plt.show()

download_data()
format_data()
train_model()
 ```
# one by one
 ```bash
pip install -r requirements.txt
docker compose build
docker compose up -d
curl http://localhost:8000/inference/ETH
docker compose logs -f
```
# if see some errors 
```bash
nano Dockerfile
```
```bash
# Stage 1: Build environment
FROM python:3.9-slim as builder

# Set the working directory in the container
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y gcc && \
    pip install --upgrade pip setuptools wheel

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the installed dependencies from the builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the application files from the current directory
COPY . .

# Install any additional system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Set the entrypoint command
CMD ["gunicorn", "--conf", "gunicorn_conf.py", "app:app"]
```


