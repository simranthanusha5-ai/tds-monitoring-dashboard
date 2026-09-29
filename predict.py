import pandas as pd

DATA_FILE = "data/PrimeOxy TDS Dataset.xlsx"
THRESHOLD = 70

data = pd.read_excel(DATA_FILE)

data["Timestamp"] = pd.to_datetime(data["Timestamp"])
data["TDS (ppm)"] = pd.to_numeric(data["TDS (ppm)"], errors="coerce")

data = data.dropna(subset=["Timestamp", "TDS (ppm)"])
data = data.sort_values("Timestamp")

data["Hour"] = data["Timestamp"].dt.hour
data["Spike"] = data["TDS (ppm)"] > THRESHOLD

hourly_average = data.groupby("Hour")["TDS (ppm)"].mean()
hourly_max = data.groupby("Hour")["TDS (ppm)"].max()

spike_rate = data.groupby("Hour")["Spike"].mean()

last_time = data["Timestamp"].iloc[-1]

future_times = pd.date_range(
    start=last_time + pd.Timedelta(hours=1),
    periods=168,
    freq="h"
)

rows = []

for timestamp in future_times:

    hour = timestamp.hour

    predicted = hourly_average.loc[hour]
    maximum = hourly_max.loc[hour]
    rate = spike_rate.loc[hour]

    if rate >= 0.25:
        risk = "HIGH"
    elif rate > 0:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    rows.append({
        "Timestamp": timestamp,
        "Predicted_TDS": round(predicted, 2),
        "Historical_Max": round(maximum, 2),
        "Spike_Rate": round(rate, 3),
        "Risk": risk,
        "Spike": predicted > THRESHOLD
    })

forecast = pd.DataFrame(rows)

forecast.to_csv("forecast.csv", index=False)

print()
print("NEXT 7 DAYS TDS FORECAST")
print("=" * 60)
print(forecast.head(24).to_string(index=False))

print()
print("SUMMARY")
print("=" * 60)
print("Minimum:", round(forecast["Predicted_TDS"].min(), 2), "ppm")
print("Maximum:", round(forecast["Predicted_TDS"].max(), 2), "ppm")
print("Average:", round(forecast["Predicted_TDS"].mean(), 2), "ppm")
print("Threshold:", THRESHOLD, "ppm")
print("Predicted breaches:", int(forecast["Spike"].sum()))

print()
print("RISK")
print("=" * 60)
print(forecast["Risk"].value_counts())

print()
print("Saved forecast.csv")
