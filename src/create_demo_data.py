import pandas as pd

INPUT_PATH = "data/spotify_resolutions.csv"
OUTPUT_PATH = "data/demo_resolutions.csv"

df = pd.read_csv(INPUT_PATH)

# Keep a small reproducible sample for the repository
demo = df.sample(
    n=min(500, len(df)),
    random_state=42
)

demo.to_csv(
    OUTPUT_PATH,
    index=False
)

print("Demo dataset created.")
print("Rows:", len(demo))
print("Saved to:", OUTPUT_PATH)