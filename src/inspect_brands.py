import pandas as pd

DATA_PATH = "data/raw/twcs.csv"

BRANDS = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "Delta",
]

SAMPLES_PER_BRAND = 5

samples = {brand: [] for brand in BRANDS}

for chunk in pd.read_csv(DATA_PATH, chunksize=100_000):
    for brand in BRANDS:
        brand_tweets = chunk[
            (chunk["author_id"] == brand)
        ]

        for _, row in brand_tweets.iterrows():
            if len(samples[brand]) < SAMPLES_PER_BRAND:
                samples[brand].append(row)

    if all(len(samples[brand]) >= SAMPLES_PER_BRAND for brand in BRANDS):
        break

for brand in BRANDS:
    print("\n" + "=" * 80)
    print(f"BRAND: {brand}")
    print("=" * 80)

    for i, row in enumerate(samples[brand], start=1):
        print(f"\nExample {i}")
        print(f"Tweet ID: {row['tweet_id']}")
        print(f"Customer/Brand: {'Customer' if row['inbound'] else 'Brand'}")
        print(f"Text: {row['text']}")