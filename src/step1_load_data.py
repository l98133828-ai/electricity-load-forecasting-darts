from darts.datasets import AirPassengersDataset

series = AirPassengersDataset().load()

print(series)
print("Start:", series.start_time())
print("End:", series.end_time())
print("Length:", len(series))
print("Frequency:", series.freq)