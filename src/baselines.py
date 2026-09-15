import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


GOLDEN_PATH = "data/golden_set.csv"

df = pd.read_csv(GOLDEN_PATH)

X = df["text"].fillna("")
y = df["gold_intent"]


# ---------------------------------------------------------
# Baseline 1: Majority class
# Always predict the most common intent
# ---------------------------------------------------------

majority_class = y.value_counts().idxmax()

majority_predictions = [majority_class] * len(y)

majority_accuracy = accuracy_score(
    y,
    majority_predictions
)


# ---------------------------------------------------------
# Baseline 2: Simple keyword rules
# ---------------------------------------------------------

def keyword_baseline(text):

    text = str(text).lower()

    if any(word in text for word in [
        "premium", "subscription", "payment",
        "billing", "charged"
    ]):
        return "account_subscription"

    if any(word in text for word in [
        "shuffle", "repeat", "playback",
        "not playing", "can't play"
    ]):
        return "playback_issue"

    if any(word in text for word in [
        "crash", "crashing", "app", "iphone",
        "android", "ios", "device"
    ]):
        return "app_device_issue"

    if any(word in text for word in [
        "missing", "unavailable", "album",
        "artist", "track", "songs"
    ]):
        return "content_availability"

    if any(word in text for word in [
        "ad", "ads", "advertisement",
        "offer", "promotion", "discount"
    ]):
        return "ads_promotions"

    return "general_support"


keyword_predictions = [
    keyword_baseline(text)
    for text in X
]

keyword_accuracy = accuracy_score(
    y,
    keyword_predictions
)


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print("=" * 70)
print("BASELINE RESULTS")
print("=" * 70)

print("\nBaseline 1 — Majority class")
print("Always predicts:", majority_class)
print("Accuracy:", round(majority_accuracy, 4))

print("\nBaseline 2 — Keyword rules")
print("Accuracy:", round(keyword_accuracy, 4))