import pandas as pd

DATA_PATH = "data/household_data_60min_singleindex.csv"
TARGET_COLUMN = "DE_KN_residential5_grid_import"

df = pd.read_csv(DATA_PATH)

print("Data shape:", df.shape)
print("Columns:", len(df.columns))

print("\nFirst 5 rows:")
print(df[["utc_timestamp", TARGET_COLUMN]].head())

print("\nMissing values in target:")
print(df[TARGET_COLUMN].isna().sum())

print("\nTarget summary:")
print(df[TARGET_COLUMN].describe())

print("\nTime range:")
print("Start:", df["utc_timestamp"].min())
print("End:", df["utc_timestamp"].max())