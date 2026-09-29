import pandas as pd
import numpy as np

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

data = pd.read_excel("data/PrimeOxy TDS Dataset.xlsx")

data["Timestamp"] = pd.to_datetime(data["Timestamp"])
data["TDS (ppm)"] = pd.to_numeric(
    data["TDS (ppm)"],
    errors="coerce"
)

data = data.dropna(
    subset=["Timestamp", "TDS (ppm)"]
)

data = data.sort_values("Timestamp").reset_index(drop=True)


# --------------------------------------------------
# 2. CREATE FEATURES
# --------------------------------------------------

data["hour"] = data["Timestamp"].dt.hour
data["day_of_week"] = data["Timestamp"].dt.dayofweek

data["lag_1"] = data["TDS (ppm)"].shift(1)
data["lag_2"] = data["TDS (ppm)"].shift(2)
data["lag_3"] = data["TDS (ppm)"].shift(3)
data["lag_6"] = data["TDS (ppm)"].shift(6)
data["lag_12"] = data["TDS (ppm)"].shift(12)
data["lag_24"] = data["TDS (ppm)"].shift(24)

data["rolling_3"] = (
    data["TDS (ppm)"]
    .shift(1)
    .rolling(3)
    .mean()
)

data["rolling_6"] = (
    data["TDS (ppm)"]
    .shift(1)
    .rolling(6)
    .mean()
)

data["rolling_24"] = (
    data["TDS (ppm)"]
    .shift(1)
    .rolling(24)
    .mean()
)

data = data.dropna().reset_index(drop=True)


# --------------------------------------------------
# 3. FEATURES
# --------------------------------------------------

features = [
    "hour",
    "day_of_week",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "lag_24",
    "rolling_3",
    "rolling_6",
    "rolling_24"
]

X = data[features]
y = data["TDS (ppm)"]


# --------------------------------------------------
# 4. TIME SERIES VALIDATION
# --------------------------------------------------

tscv = TimeSeriesSplit(n_splits=4)

results = {
    "Previous Hour": [],
    "Previous Day": [],
    "Gradient Boosting": []
}


for fold, (train_index, test_index) in enumerate(
    tscv.split(X),
    start=1
):

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]


    # --------------------------------------------------
    # BASELINE 1: PREVIOUS HOUR
    # --------------------------------------------------

    previous_hour_predictions = X_test["lag_1"].values

    mae = mean_absolute_error(
        y_test,
        previous_hour_predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            previous_hour_predictions
        )
    )

    results["Previous Hour"].append(
        (mae, rmse)
    )


    # --------------------------------------------------
    # BASELINE 2: PREVIOUS DAY
    # --------------------------------------------------

    previous_day_predictions = X_test["lag_24"].values

    mae = mean_absolute_error(
        y_test,
        previous_day_predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            previous_day_predictions
        )
    )

    results["Previous Day"].append(
        (mae, rmse)
    )


    # --------------------------------------------------
    # MODEL: GRADIENT BOOSTING
    # --------------------------------------------------

    model = HistGradientBoostingRegressor(
        max_iter=300,
        max_leaf_nodes=15,
        l2_regularization=1,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)


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

    results["Gradient Boosting"].append(
        (mae, rmse)
    )


    print(f"\nFold {fold}")
    print("-" * 40)

    print(
        f"Previous Hour   MAE: "
        f"{results['Previous Hour'][-1][0]:.2f} "
        f"RMSE: "
        f"{results['Previous Hour'][-1][1]:.2f}"
    )

    print(
        f"Previous Day    MAE: "
        f"{results['Previous Day'][-1][0]:.2f} "
        f"RMSE: "
        f"{results['Previous Day'][-1][1]:.2f}"
    )

    print(
        f"Gradient Boost  MAE: "
        f"{results['Gradient Boosting'][-1][0]:.2f} "
        f"RMSE: "
        f"{results['Gradient Boosting'][-1][1]:.2f}"
    )


# --------------------------------------------------
# 5. FINAL COMPARISON
# --------------------------------------------------

print("\n")
print("=" * 65)
print("FINAL MODEL COMPARISON")
print("=" * 65)

summary = []

for model_name, scores in results.items():

    mae_values = [
        score[0]
        for score in scores
    ]

    rmse_values = [
        score[1]
        for score in scores
    ]

    summary.append({
        "Model": model_name,
        "Average MAE": np.mean(mae_values),
        "Average RMSE": np.mean(rmse_values)
    })


summary_df = pd.DataFrame(summary)

summary_df = summary_df.sort_values(
    "Average MAE"
)

print(
    summary_df.to_string(
        index=False,
        formatters={
            "Average MAE": "{:.2f}".format,
            "Average RMSE": "{:.2f}".format
        }
    )
)

print("=" * 65)