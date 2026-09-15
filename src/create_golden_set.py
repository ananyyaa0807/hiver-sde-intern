import pandas as pd

INPUT_PATH = "data/spotify_customer_messages.csv"
OUTPUT_PATH = "data/golden_set.csv"

df = pd.read_csv(INPUT_PATH)

# Fixed random seed makes the sample reproducible
golden = df.sample(
    n=200,
    random_state=42
).copy()

golden["gold_intent"] = ""

golden = golden[
    ["tweet_id", "text", "gold_intent"]
]

golden.to_csv(
    OUTPUT_PATH,
    index=False
)

print("Golden set created.")
print("Number of examples:", len(golden))
print("Saved to:", OUTPUT_PATH)