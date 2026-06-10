import matplotlib.pyplot as plt
import pandas as pd

from darts.datasets import AirPassengersDataset
from darts.models import NaiveSeasonal, ExponentialSmoothing, ARIMA, RandomForestModel
from darts.metrics import mae, rmse, mape


def evaluate_model(name, model, train, val):
    model.fit(train)
    prediction = model.predict(len(val))

    result = {
        "model": name,
        "mae": mae(val, prediction),
        "rmse": rmse(val, prediction),
        "mape": mape(val, prediction),
        "prediction": prediction,
    }

    return result


def save_best_forecast_plot(series, prediction, model_name):
    series.plot(label="actual")
    prediction.plot(label="forecast")

    plt.title(f"AirPassengers Forecast with {model_name}")
    plt.xlabel("Month")
    plt.ylabel("Passengers")
    plt.legend()
    plt.savefig("outputs/best_forecast.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    series = AirPassengersDataset().load()
    train, val = series[:-36], series[-36:]

    models = {
        "NaiveSeasonal(K=12)": NaiveSeasonal(K=12),
        "ExponentialSmoothing": ExponentialSmoothing(),
        "ARIMA": ARIMA(),
        "RandomForestModel": RandomForestModel(
            lags=12,
            n_estimators=200,
            random_state=42,
        ),
    }

    results = []

    for name, model in models.items():
        print(f"Training {name}...")
        result = evaluate_model(name, model, train, val)
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

    results_df = results_df.sort_values("mape")
    results_df.to_csv("outputs/model_comparison.csv", index=False)

    print("\nModel Comparison")
    print(results_df)

    best_model_name = results_df.iloc[0]["model"]
    best_prediction = next(
        row["prediction"]
        for row in results
        if row["model"] == best_model_name
    )

    save_best_forecast_plot(series, best_prediction, best_model_name)

    print(f"\nBest model: {best_model_name}")
    print("Saved results to outputs/model_comparison.csv")
    print("Saved plot to outputs/best_forecast.png")


if __name__ == "__main__":
    main()