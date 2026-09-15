import pandas as pd

DATA_PATH = "data/raw/twcs.csv"

total_rows = 0
inbound_count = 0
outbound_count = 0
authors = set()

for chunk in pd.read_csv(DATA_PATH, chunksize=100_000):
    total_rows += len(chunk)

    inbound_count += chunk["inbound"].sum()
    outbound_count += (~chunk["inbound"]).sum()

    authors.update(chunk["author_id"].unique())

print("Total tweets:", total_rows)
print("Customer tweets (inbound):", inbound_count)
print("Brand tweets (outbound):", outbound_count)
print("Unique authors:", len(authors))