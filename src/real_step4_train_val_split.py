import matplotlib.pyplot as plt
import pandas as pd

from darts import TimeSeries

DATA_PATH = "data/household_data_60min_singleindex.csv"
TARGET_COLUMN = "DE_KN_residential5_grid_import"
VAL_HOURS = 24 * 30

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

series = TimeSeries.from_dataframe(
    df,
    time_col="time",
    value_cols="hourly_consumption",
    fill_missing_dates=True,
    freq="h",
)

train, val = series[:-VAL_HOURS], series[-VAL_HOURS:]

print("Train start:", train.start_time())
print("Train end:", train.end_time())
print("Train length:", len(train))

print("Validation start:", val.start_time())
print("Validation end:", val.end_time())
print("Validation length:", len(val))

train[-24 * 14:].plot(label="train last 14 days")
val.plot(label="validation 30 days")

plt.title("Electricity Load Train / Validation Split")
plt.xlabel("Time")
plt.ylabel("Hourly Consumption")
plt.legend()
plt.savefig("outputs/electricity_train_val_split.png", dpi=300, bbox_inches="tight")
plt.show()