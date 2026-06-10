import matplotlib.pyplot as plt

from darts.datasets import AirPassengersDataset

series = AirPassengersDataset().load()

series.plot(label="Monthly passengers")
plt.title("AirPassengers Time Series")
plt.xlabel("Month")
plt.ylabel("Passengers")
plt.legend()
plt.show()