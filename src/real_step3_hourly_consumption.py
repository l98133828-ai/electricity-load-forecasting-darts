import matplotlib.pyplot as plt
import pandas as pd

from darts import TimeSeries

DATA_PATH = "data/household_data_60min_singleindex.csv"
TARGET_COLUMN = "DE_KN_residential5_grid_import"

df = pd.read_csv(DATA_PATH)

df = df[["utc_timestamp", TARGET_COLUMN]].copy()
df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"])
df = df.dropna(subset=[TARGET_COLUMN])

df = df.rename(columns={
    "utc_timestamp": "time",
    TARGET_COLUMN: "cumulative_import",
})

df = df.sort_values("time")

df["hourly_consumption"] = df["cumulative_import"].diff()

df = df.dropna(subset=["hourly_consumption"])

df = df[df["hourly_consumption"] >= 0]

print("Hourly consumption summary:")
print(df["hourly_consumption"].describe())

series = TimeSeries.from_dataframe(
    df,
    time_col="time",
    value_cols="hourly_consumption",
    fill_missing_dates=True,
    freq="h",
)

print(series)
print("Start:", series.start_time())
print("End:", series.end_time())
print("Length:", len(series))
print("Frequency:", series.freq)

series.plot(label="hourly consumption")
plt.title("Residential Hourly Electricity Consumption")
plt.xlabel("Time")
plt.ylabel("Hourly Consumption")
plt.legend()
plt.savefig("outputs/electricity_hourly_consumption.png", dpi=300, bbox_inches="tight")
plt.show()