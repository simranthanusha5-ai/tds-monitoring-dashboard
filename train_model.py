import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import joblib


# 1. Load the dataset
data = pd.read_excel(
    "data/PrimeOxy TDS Dataset.xlsx"
)


# 2. Prepare the data
data["Timestamp"] = pd.to_datetime(
    data["Timestamp"]
)

data["TDS (ppm)"] = pd.to_numeric(
    data["TDS (ppm)"],
    errors="coerce"
)

data = data.dropna(
    subset=["Timestamp", "TDS (ppm)"]
)

data = data.sort_values(
    "Timestamp"
)


# 3. Create time features
data["hour"] = data["Timestamp"].dt.hour

data["day_of_week"] = (
    data["Timestamp"].dt.dayofweek
)


# 4. Create previous TDS features
data["tds_lag_1"] = (
    data["TDS (ppm)"].shift(1)
)

data["tds_lag_2"] = (
    data["TDS (ppm)"].shift(2)
)

data["tds_lag_3"] = (
    data["TDS (ppm)"].shift(3)
)

data["tds_lag_24"] = (
    data["TDS (ppm)"].shift(24)
)


# Remove rows where lag values don't exist
data = data.dropna()


# 5. Select input features
features = [
    "hour",
    "day_of_week",
    "tds_lag_1",
    "tds_lag_2",
    "tds_lag_3",
    "tds_lag_24"
]

X = data[features]

y = data["TDS (ppm)"]


# 6. Split chronologically
split = int(len(data) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]


# 7. Create the model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)


# 8. Train
model.fit(
    X_train,
    y_train
)


# 9. Predict the test data
predictions = model.predict(
    X_test
)


# 10. Evaluate the model
mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

print("Model Results")
print("--------------------")

print(
    f"MAE: {mae:.2f} ppm"
)

print(
    f"RMSE: {rmse:.2f} ppm"
)


# 11. Save the trained model
joblib.dump(
    model,
    "tds_model.pkl"
)

print("--------------------")
print("Model saved as tds_model.pkl")