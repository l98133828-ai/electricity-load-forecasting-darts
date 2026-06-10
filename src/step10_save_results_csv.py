import pandas as pd

from darts.datasets import AirPassengersDataset
from darts.models import NaiveSeasonal, ExponentialSmoothing, ARIMA, RandomForestModel
from darts.metrics import mae, rmse, mape

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
    model.fit(train)
    prediction = model.predict(len(val))

    results.append({
        "model": name,
        "mae": mae(val, prediction),
        "rmse": rmse(val, prediction),
        "mape": mape(val, prediction),
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("mape")

print(results_df)

results_df.to_csv("outputs/model_comparison.csv", index=False)
print("Saved to outputs/model_comparison.csv")