import matplotlib.pyplot as plt

from darts.datasets import AirPassengersDataset
from darts.models import ExponentialSmoothing
from darts.metrics import mape

series = AirPassengersDataset().load()

train, val = series[:-12], series[-12:]

model = ExponentialSmoothing()
model.fit(train)

prediction = model.predict(len(val))

print("MAPE:", mape(val, prediction))

series.plot(label="actual")
prediction.plot(label="forecast")
plt.legend()
plt.show()