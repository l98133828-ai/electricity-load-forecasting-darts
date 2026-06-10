import matplotlib.pyplot as plt

from darts.datasets import AirPassengersDataset

series = AirPassengersDataset().load()

train, val = series[:-36], series[-36:]

train.plot(label="train")
val.plot(label="validation")

plt.title("Train / Validation Split")
plt.xlabel("Month")
plt.ylabel("Passengers")
plt.legend()
plt.show()

print("Train start:", train.start_time())
print("Train end:", train.end_time())
print("Train length:", len(train))

print("Validation start:", val.start_time())
print("Validation end:", val.end_time())
print("Validation length:", len(val))