import pandas as pd

DATA_PATH = "data/raw/twcs.csv"

df = pd.read_csv(DATA_PATH, nrows=5)

print("Columns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.to_string())

print("\nNumber of columns:", len(df.columns))