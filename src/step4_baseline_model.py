import matplotlib.pyplot as plt

from darts.datasets import AirPassengersDataset
from darts.models import NaiveSeasonal
from darts.metrics import mae, rmse, mape

series = AirPassengersDataset().load()

train, val = series[:-36], series[-36:]

model = NaiveSeasonal(K=12)
model.fit(train)

prediction = model.predict(len(val))

print("MAE:", mae(val, prediction))
print("RMSE:", rmse(val, prediction))
print("MAPE:", mape(val, prediction))

series.plot(label="actual")
prediction.plot(label="NaiveSeasonal forecast")

plt.title("NaiveSeasonal Forecast")
plt.xlabel("Month")
plt.ylabel("Passengers")
plt.legend()
plt.show()