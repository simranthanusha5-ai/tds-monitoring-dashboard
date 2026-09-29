from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd

app = Flask(__name__)

app.secret_key = "tds-project-secret"

DATA_FILE = "data/PrimeOxy TDS Dataset.xlsx"
FORECAST_FILE = "forecast.csv"

THRESHOLD = 70


# =========================================================
# GENERATE FORECAST
# =========================================================

def generate_forecast():

    data = pd.read_excel(DATA_FILE)

    data["Timestamp"] = pd.to_datetime(data["Timestamp"])
    data["TDS (ppm)"] = pd.to_numeric(
        data["TDS (ppm)"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["Timestamp", "TDS (ppm)"]
    )

    data = data.sort_values("Timestamp")

    data["Hour"] = data["Timestamp"].dt.hour

    data["Spike"] = data["TDS (ppm)"] > THRESHOLD

    hourly_average = (
        data.groupby("Hour")["TDS (ppm)"]
        .mean()
    )

    hourly_max = (
        data.groupby("Hour")["TDS (ppm)"]
        .max()
    )

    spike_rate = (
        data.groupby("Hour")["Spike"]
        .mean()
    )

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

    forecast.to_csv(
        FORECAST_FILE,
        index=False
    )


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/")
def dashboard():

    # Load historical data
    data = pd.read_excel(DATA_FILE)

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

    data = data.sort_values("Timestamp")


    # Current reading
    current_tds = float(
        data["TDS (ppm)"].iloc[-1]
    )

    current_time = data["Timestamp"].iloc[-1]


    # Historical spikes
    historical_spikes = int(
        (
            data["TDS (ppm)"] > THRESHOLD
        ).sum()
    )


    # -----------------------------------------------------
    # ACTUAL DATA FOR COMPARISON CHART
    # -----------------------------------------------------

    actual_data = data.tail(48).copy()

    actual_labels = [
        timestamp.strftime("%d %b %H:%M")
        for timestamp in actual_data["Timestamp"]
    ]

    actual_values = [
        float(value)
        for value in actual_data["TDS (ppm)"]
    ]


    # -----------------------------------------------------
    # FORECAST
    # -----------------------------------------------------

    forecast = pd.read_csv(
        FORECAST_FILE
    )

    forecast["Timestamp"] = pd.to_datetime(
        forecast["Timestamp"]
    )


    highest_prediction = float(
        forecast["Predicted_TDS"].max()
    )

    average_prediction = float(
        forecast["Predicted_TDS"].mean()
    )

    predicted_breaches = int(
        forecast["Spike"].sum()
    )


    # -----------------------------------------------------
    # RISK COUNTS
    # -----------------------------------------------------

    high_risk = int(
        (
            forecast["Risk"] == "HIGH"
        ).sum()
    )

    medium_risk = int(
        (
            forecast["Risk"] == "MEDIUM"
        ).sum()
    )

    low_risk = int(
        (
            forecast["Risk"] == "LOW"
        ).sum()
    )


    # -----------------------------------------------------
    # SYSTEM STATUS
    # -----------------------------------------------------

    if current_tds > THRESHOLD:
        system_status = "ALERT"

    elif high_risk > 0:
        system_status = "MONITOR"

    else:
        system_status = "NORMAL"


    # -----------------------------------------------------
    # ALERTS
    # -----------------------------------------------------

    alerts = forecast[
        forecast["Risk"].isin(
            ["HIGH", "MEDIUM"]
        )
    ].head(20)

    alert_rows = []

    for _, row in alerts.iterrows():

        if row["Risk"] == "HIGH":

            message = (
                "High historical spike risk "
                "for this time period."
            )

        else:

            message = (
                "Medium historical spike risk "
                "for this time period."
            )

        alert_rows.append({
            "Timestamp":
                row["Timestamp"].strftime(
                    "%d %b %Y %H:%M"
                ),

            "Predicted_TDS":
                round(
                    row["Predicted_TDS"],
                    2
                ),

            "Spike_Rate":
                round(
                    row["Spike_Rate"] * 100,
                    1
                ),

            "Risk":
                row["Risk"],

            "Message":
                message
        })


    # -----------------------------------------------------
    # COMBINED CHART DATA
    # -----------------------------------------------------

    forecast_labels = [
        timestamp.strftime("%d %b %H:%M")
        for timestamp in forecast["Timestamp"]
    ]

    forecast_values = [
        float(value)
        for value in forecast["Predicted_TDS"]
    ]

    # Labels = actual period + forecast period
    chart_labels = actual_labels + forecast_labels

    # Actual line ends when forecast begins
    chart_actual_values = (
        actual_values +
        [None] * len(forecast_values)
    )

    # Forecast line begins after actual data
    chart_forecast_values = (
        [None] * len(actual_values) +
        forecast_values
    )

    chart_threshold_values = (
        [THRESHOLD] *
        len(chart_labels)
    )


    # -----------------------------------------------------
    # FORECAST TABLE
    # -----------------------------------------------------

    forecast_table = forecast.copy()

    forecast_table["Timestamp"] = (
        forecast_table["Timestamp"]
        .dt.strftime("%d %b %Y %H:%M")
    )

    forecast_table = forecast_table.head(48)

    forecast_rows = forecast_table.to_dict(
        orient="records"
    )


    # -----------------------------------------------------
    # RENDER
    # -----------------------------------------------------

    return render_template(

        "index.html",

        current_tds=round(
            current_tds,
            2
        ),

        current_time=
            current_time.strftime(
                "%d %b %Y %H:%M"
            ),

        threshold=THRESHOLD,

        historical_spikes=
            historical_spikes,

        highest_prediction=
            round(
                highest_prediction,
                2
            ),

        average_prediction=
            round(
                average_prediction,
                2
            ),

        predicted_breaches=
            predicted_breaches,

        high_risk=high_risk,

        medium_risk=medium_risk,

        low_risk=low_risk,

        system_status=
            system_status,

        alerts=
            alert_rows,

        chart_labels=
            chart_labels,

        chart_actual_values=
            chart_actual_values,

        chart_forecast_values=
            chart_forecast_values,

        chart_threshold_values=
            chart_threshold_values,

        forecast_rows=
            forecast_rows
    )


# =========================================================
# UPLOAD
# =========================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload():

    file = request.files.get("file")

    if not file or not file.filename:

        flash(
            "Please select an Excel file."
        )

        return redirect(
            url_for("dashboard")
        )

    filename = file.filename.lower()

    if not filename.endswith(".xlsx"):

        flash(
            "Please upload an Excel (.xlsx) file."
        )

        return redirect(
            url_for("dashboard")
        )

    try:

        file.save(DATA_FILE)

        test_data = pd.read_excel(
            DATA_FILE
        )

        required_columns = [
            "Timestamp",
            "TDS (ppm)"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in test_data.columns
        ]

        if missing_columns:

            flash(
                "Invalid file. Required columns are "
                "Timestamp and TDS (ppm)."
            )

            return redirect(
                url_for("dashboard")
            )

        generate_forecast()

        flash(
            "Dataset uploaded successfully. "
            "A new 7-day forecast has been generated."
        )

    except Exception as error:

        flash(
            f"Upload failed: {error}"
        )

    return redirect(
        url_for("dashboard")
    )


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=8000,
        debug=True
    )