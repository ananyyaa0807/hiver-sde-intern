import pandas as pd

DATA_PATH = "data/raw/twcs.csv"

brand_counts = {}

for chunk in pd.read_csv(DATA_PATH, chunksize=100_000):
    outbound = chunk[chunk["inbound"] == False]

    counts = outbound["author_id"].value_counts()

    for author, count in counts.items():
        brand_counts[author] = brand_counts.get(author, 0) + count

top_brands = sorted(
    brand_counts.items(),
    key=lambda item: item[1],
    reverse=True
)

print("Top 30 outbound accounts:\n")

for author, count in top_brands[:30]:
    print(f"{author:<30} {count:>10,}")