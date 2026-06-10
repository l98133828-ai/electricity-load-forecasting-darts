import matplotlib.pyplot as plt

from darts.datasets import AirPassengersDataset
from darts.models import ExponentialSmoothing
from darts.metrics import mae, rmse, mape

series = AirPassengersDataset().load()

train, val = series[:-36], series[-36:]

model = ExponentialSmoothing()
model.fit(train)

prediction = model.predict(len(val))

print("MAE:", mae(val, prediction))
print("RMSE:", rmse(val, prediction))
print("MAPE:", mape(val, prediction))

series.plot(label="actual")
prediction.plot(label="ExponentialSmoothing forecast")

plt.title("Exponential Smoothing Forecast")
plt.xlabel("Month")
plt.ylabel("Passengers")
plt.legend()
plt.show()