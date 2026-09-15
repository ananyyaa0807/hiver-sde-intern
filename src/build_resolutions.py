import pandas as pd

DATA_PATH = "data/raw/twcs.csv"
OUTPUT_PATH = "data/spotify_resolutions.csv"

BRAND = "SpotifyCares"

# ---------------------------------------------------------
# PASS 1:
# Find SpotifyCares replies and the customer tweet
# they are replying to.
# ---------------------------------------------------------

pairs = []

for chunk in pd.read_csv(DATA_PATH, chunksize=100_000):

    brand_replies = chunk[
        (chunk["author_id"] == BRAND) &
        (chunk["inbound"] == False) &
        (chunk["in_response_to_tweet_id"].notna())
    ]

    for _, row in brand_replies.iterrows():
        pairs.append({
            "customer_tweet_id": int(row["in_response_to_tweet_id"]),
            "brand_tweet_id": int(row["tweet_id"]),
            "brand_reply": row["text"]
        })


print("SpotifyCares replies with parent customer tweets:", len(pairs))


# ---------------------------------------------------------
# PASS 2:
# Get the actual customer messages
# ---------------------------------------------------------

pairs_df = pd.DataFrame(pairs)

customer_ids = set(pairs_df["customer_tweet_id"])

customer_rows = []

for chunk in pd.read_csv(DATA_PATH, chunksize=100_000):

    matches = chunk[
        chunk["tweet_id"].isin(customer_ids)
    ]

    customer_rows.append(matches)


customers = pd.concat(customer_rows, ignore_index=True)

customers = customers[
    customers["inbound"] == True
]


# ---------------------------------------------------------
# Join customer message + historical brand reply
# ---------------------------------------------------------

customers = customers[
    ["tweet_id", "text"]
].rename(
    columns={
        "tweet_id": "customer_tweet_id",
        "text": "customer_message"
    }
)

resolutions = pairs_df.merge(
    customers,
    on="customer_tweet_id",
    how="inner"
)

# Remove duplicates
resolutions = resolutions.drop_duplicates(
    subset=["customer_tweet_id", "brand_tweet_id"]
)

# Save
resolutions.to_csv(
    OUTPUT_PATH,
    index=False
)

print("Historical resolution pairs:", len(resolutions))
print("Saved to:", OUTPUT_PATH)

print("\nSample historical resolutions:\n")

for _, row in resolutions.head(10).iterrows():

    print("-" * 80)
    print("CUSTOMER:")
    print(row["customer_message"])

    print("\nSPOTIFYCARES REPLY:")
    print(row["brand_reply"])