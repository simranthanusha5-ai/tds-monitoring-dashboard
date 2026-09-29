import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Load data
data = pd.read_excel("data/PrimeOxy TDS Dataset.xlsx")

data["Timestamp"] = pd.to_datetime(data["Timestamp"])
data["TDS (ppm)"] = pd.to_numeric(
    data["TDS (ppm)"],
    errors="coerce"
)

data = data.dropna(
    subset=["Timestamp", "TDS (ppm)"]
)

data = data.sort_values(
    "Timestamp"
).reset_index(drop=True)

data["hour"] = data["Timestamp"].dt.hour


# Use last 48 hours for testing
train = data.iloc[:-48].copy()
test = data.iloc[-48:].copy()


# --------------------------------------------------
# SIMPLE HOURLY AVERAGE
# --------------------------------------------------

hour_average = (
    train
    .groupby("hour")["TDS (ppm)"]
    .mean()
)

simple_predictions = []

for _, row in test.iterrows():

    prediction = hour_average.loc[row["hour"]]

    simple_predictions.append(prediction)


# --------------------------------------------------
# RECENT-WEIGHTED HOURLY AVERAGE
# --------------------------------------------------

weighted_predictions = []

for _, row in test.iterrows():

    hour = row["hour"]

    historical = train[
        train["hour"] == hour
    ].copy()

    historical["weight"] = np.arange(
        1,
        len(historical) + 1
    )

    prediction = np.average(
        historical["TDS (ppm)"],
        weights=historical["weight"]
    )

    weighted_predictions.append(prediction)


# --------------------------------------------------
# EVALUATE
# --------------------------------------------------

actual = test["TDS (ppm)"].values


simple_mae = mean_absolute_error(
    actual,
    simple_predictions
)

simple_rmse = np.sqrt(
    mean_squared_error(
        actual,
        simple_predictions
    )
)


weighted_mae = mean_absolute_error(
    actual,
    weighted_predictions
)

weighted_rmse = np.sqrt(
    mean_squared_error(
        actual,
        weighted_predictions
    )
)


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print()
print("=" * 65)
print("HOURLY FORECAST COMPARISON")
print("=" * 65)

print()
print("Simple Hourly Average")

print(
    f"MAE:  {simple_mae:.2f} ppm"
)

print(
    f"RMSE: {simple_rmse:.2f} ppm"
)

print()
print("Recent-Weighted Hourly Average")

print(
    f"MAE:  {weighted_mae:.2f} ppm"
)

print(
    f"RMSE: {weighted_rmse:.2f} ppm"
)

print("=" * 65)


# --------------------------------------------------
# ACTUAL VS PREDICTED
# --------------------------------------------------

comparison = pd.DataFrame({

    "Timestamp": test["Timestamp"].values,

    "Actual_TDS": actual,

    "Simple_Forecast": simple_predictions,

    "Weighted_Forecast": weighted_predictions

})

print()
print(comparison.to_string(index=False))