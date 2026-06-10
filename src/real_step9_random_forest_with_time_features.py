import matplotlib.pyplot as plt
import pandas as pd

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae, rmse, mape
from darts.models import RandomForestModel
from darts.utils.timeseries_generation import datetime_attribute_timeseries

DATA_PATH = "data/household_data_60min_singleindex.csv"
TARGET_COLUMN = "DE_KN_residential5_grid_import"
VAL_HOURS = 24 * 30


def load_hourly_consumption_series():
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

    return series


series = load_hourly_consumption_series()
train, val = series[:-VAL_HOURS], series[-VAL_HOURS:]

hour_covariate = datetime_attribute_timeseries(
    series,
    attribute="hour",
    one_hot=False,
)

dayofweek_covariate = datetime_attribute_timeseries(
    series,
    attribute="dayofweek",
    one_hot=False,
)

month_covariate = datetime_attribute_timeseries(
    series,
    attribute="month",
    one_hot=False,
)

future_covariates = hour_covariate.stack(dayofweek_covariate).stack(month_covariate)

scaler = Scaler()
future_covariates = scaler.fit_transform(future_covariates)

model = RandomForestModel(
    lags=[-1, -2, -3, -24, -48, -168],
    lags_future_covariates=[0],
    n_estimators=300,
    random_state=42,
)

model.fit(
    train,
    future_covariates=future_covariates,
)

prediction = model.predict(
    len(val),
    future_covariates=future_covariates,
)

print("RandomForestModel with time covariates")
print("MAE:", mae(val, prediction))
print("RMSE:", rmse(val, prediction))
print("MAPE:", mape(val, prediction))

val.plot(label="actual")
prediction.plot(label="forecast")

plt.title("Electricity Load Forecast with RandomForest + Time Features")
plt.xlabel("Time")
plt.ylabel("Hourly Consumption")
plt.legend()
plt.savefig("outputs/electricity_random_forest_time_features.png", dpi=300, bbox_inches="tight")
plt.show()