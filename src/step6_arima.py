import matplotlib.pyplot as plt

from darts.datasets import AirPassengersDataset
from darts.models import ARIMA
from darts.metrics import mae, rmse, mape

series = AirPassengersDataset().load()

train, val = series[:-36], series[-36:]

model = ARIMA()
model.fit(train)

prediction = model.predict(len(val))

print("MAE:", mae(val, prediction))
print("RMSE:", rmse(val, prediction))
print("MAPE:", mape(val, prediction))

series.plot(label="actual")
prediction.plot(label="ARIMA forecast")

plt.title("ARIMA Forecast")
plt.xlabel("Month")
plt.ylabel("Passengers")
plt.legend()
plt.show()