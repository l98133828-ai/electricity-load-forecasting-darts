import matplotlib.pyplot as plt

from darts.datasets import AirPassengersDataset
from darts.models import ExponentialSmoothing
from darts.metrics import mae, rmse, mape

series = AirPassengersDataset().load()
train, val = series[:-36], series[-36:]

model = ExponentialSmoothing()
model.fit(train)

prediction = model.predict(len(val))

print("Best Model: ExponentialSmoothing")
print("MAE:", mae(val, prediction))
print("RMSE:", rmse(val, prediction))
print("MAPE:", mape(val, prediction))

series.plot(label="actual")
prediction.plot(label="forecast")

plt.title("AirPassengers Forecast with ExponentialSmoothing")
plt.xlabel("Month")
plt.ylabel("Passengers")
plt.legend()
plt.savefig("outputs/best_forecast.png", dpi=300, bbox_inches="tight")
plt.show()