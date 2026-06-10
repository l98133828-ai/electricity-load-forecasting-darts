import matplotlib.pyplot as plt
import pandas as pd

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import NaiveSeasonal, RandomForestModel, ExponentialSmoothing
from darts.metrics import mae, rmse, mape
from darts.utils.timeseries_generation import datetime_attribute_timeseries

DATA_PATH = "data/household_data_60min_singleindex.csv"
TARGET_COLUMN = "DE_KN_residential5_grid_import"
VAL_HOURS = 24 * 30


def load_hourly_consumption_series():
    df = pd.read_csv(DATA_PATH)

    df = df[["utc_timestamp", TARGET_COLUMN]].copy()
    df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"])
    df = df.dropna(subset=[TARGET_COLUMN])

    df = df.rename(columns={
        "utc_timestamp": "time",
        TARGET_COLUMN: "cumulative_import",
    })

    df = df.sort_values("time")
    df["hourly_consumption"] = df["cumulative_import"].diff()
    df = df.dropna(subset=["hourly_consumption"])
    df = df[df["hourly_consumption"] >= 0]

    series = TimeSeries.from_dataframe(
        df,
        time_col="time",
        value_cols="hourly_consumption",
        fill_missing_dates=True,
        freq="h",
    )

    return series


def evaluate_model(name, model, train, val):
    print(f"Training {name}...")
    model.fit(train)
    prediction = model.predict(len(val))

    return {
        "model": name,
        "mae": mae(val, prediction),
        "rmse": rmse(val, prediction),
        "mape": mape(val, prediction),
        "prediction": prediction,
    }


def evaluate_model_with_covariates(name, model, train, val, future_covariates):
    print(f"Training {name}...")
    model.fit(train, future_covariates=future_covariates)
    prediction = model.predict(len(val), future_covariates=future_covariates)

    return {
        "model": name,
        "mae": mae(val, prediction),
        "rmse": rmse(val, prediction),
        "mape": mape(val, prediction),
        "prediction": prediction,
    }


def build_time_covariates(series):
    hour_covariate = datetime_attribute_timeseries(
        series,
        attribute="hour",
        one_hot=False,
    )

    dayofweek_covariate = datetime_attribute_timeseries(
        series,
        attribute="dayofweek",
        one_hot=False,
    )

    month_covariate = datetime_attribute_timeseries(
        series,
        attribute="month",
        one_hot=False,
    )

    future_covariates = hour_covariate.stack(dayofweek_covariate).stack(month_covariate)

    scaler = Scaler()
    return scaler.fit_transform(future_covariates)


def save_forecast_plot(val, prediction, model_name):
    plt.figure()
    val.plot(label="actual")
    prediction.plot(label="forecast")

    plt.title(f"Electricity Load Forecast with {model_name}")
    plt.xlabel("Time")
    plt.ylabel("Hourly Consumption")
    plt.legend()
    plt.savefig("outputs/electricity_best_forecast.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    series = load_hourly_consumption_series()
    train, val = series[:-VAL_HOURS], series[-VAL_HOURS:]
    future_covariates = build_time_covariates(series)

    models = {
        "NaiveSeasonal(K=24)": NaiveSeasonal(K=24),
        "NaiveSeasonal(K=168)": NaiveSeasonal(K=168),
        "RandomForestModel": RandomForestModel(
            lags=[-1, -2, -3, -24, -48, -168],
            n_estimators=300,
            random_state=42,
        ),
        "ExponentialSmoothing(seasonal=24)": ExponentialSmoothing(
            seasonal_periods=24,
        ),
    }

    covariate_models = {
        "RandomForestModel + time covariates": RandomForestModel(
            lags=[-1, -2, -3, -24, -48, -168],
            lags_future_covariates=[0],
            n_estimators=300,
            random_state=42,
        ),
    }

    results = []

    for name, model in models.items():
        result = evaluate_model(name, model, train, val)
        results.append(result)

    for name, model in covariate_models.items():
        result = evaluate_model_with_covariates(
            name,
            model,
            train,
            val,
            future_covariates,
        )
        results.append(result)

    results_df = pd.DataFrame([
        {
            "model": row["model"],
            "mae": row["mae"],
            "rmse": row["rmse"],
            "mape": row["mape"],
        }
        for row in results
    ])

    results_df = results_df.sort_values("mae")
    results_df.to_csv("outputs/electricity_model_comparison.csv", index=False)

    print("\nElectricity Model Comparison")
    print(results_df)

    best_model_name = results_df.iloc[0]["model"]
    best_prediction = next(
        row["prediction"]
        for row in results
        if row["model"] == best_model_name
    )

    save_forecast_plot(val, best_prediction, best_model_name)

    print(f"\nBest model by MAE: {best_model_name}")
    print("Saved results to outputs/electricity_model_comparison.csv")
    print("Saved plot to outputs/electricity_best_forecast.png")


if __name__ == "__main__":
    main()
