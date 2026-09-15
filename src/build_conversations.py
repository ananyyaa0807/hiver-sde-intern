import pandas as pd

DATA_PATH = "data/raw/twcs.csv"
BRAND = "SpotifyCares"

# ---------------------------------------------------------
# PASS 1:
# Find customer tweets that SpotifyCares directly replied to
# ---------------------------------------------------------

customer_ids = set()
brand_reply_count = 0

for chunk in pd.read_csv(DATA_PATH, chunksize=100_000):

    brand_replies = chunk[
        (chunk["author_id"] == BRAND) &
        (chunk["inbound"] == False)
    ]

    brand_reply_count += len(brand_replies)

    ids = brand_replies["in_response_to_tweet_id"].dropna()

    customer_ids.update(ids.astype(int).tolist())


print("Brand:", BRAND)
print("Brand replies:", brand_reply_count)
print("Customer messages with direct brand replies:", len(customer_ids))


# ---------------------------------------------------------
# PASS 2:
# Retrieve those customer messages
# ---------------------------------------------------------

customer_messages = []

for chunk in pd.read_csv(DATA_PATH, chunksize=100_000):

    matches = chunk[
        chunk["tweet_id"].isin(customer_ids)
    ]

    customer_messages.append(matches)

customers = pd.concat(customer_messages, ignore_index=True)

# Keep only actual inbound/customer messages
customers = customers[customers["inbound"] == True]

# Save the extracted customer messages
customers.to_csv(
    "data/spotify_customer_messages.csv",
    index=False
)

print("\nExtracted customer messages:", len(customers))
print("Saved to: data/spotify_customer_messages.csv")

print("\nSample customer messages:\n")

for _, row in customers.head(20).iterrows():
    print("-" * 80)
    print("Tweet ID:", row["tweet_id"])
    print("Text:", row["text"])