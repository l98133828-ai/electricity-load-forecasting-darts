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
        "MAE": mae(val, prediction),
        "RMSE": rmse(val, prediction),
        "MAPE": mape(val, prediction),
    })

print("Model Comparison")
print("-" * 70)
print(f"{'Model':25s} {'MAE':>10s} {'RMSE':>10s} {'MAPE':>10s}")

for row in results:
    print(
        f"{row['model']:25s} "
        f"{row['MAE']:10.2f} "
        f"{row['RMSE']:10.2f} "
        f"{row['MAPE']:10.2f}"
    )