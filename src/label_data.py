import pandas as pd
import re

INPUT_PATH = "data/spotify_customer_messages.csv"
OUTPUT_PATH = "data/spotify_labeled.csv"


def classify_intent(text):
    text = str(text).lower()

    if any(word in text for word in [
        "shuffle", "repeat", "play", "playing", "playback",
        "song won't", "song doesnt", "song doesn't", "can't play"
    ]):
        return "playback_issue"

    if any(word in text for word in [
        "premium", "subscription", "subscribed", "payment",
        "billing", "charge", "account"
    ]):
        return "account_subscription"

    if any(word in text for word in [
        "app", "crash", "crashes", "bug", "update",
        "iphone", "android", "ios", "device", "desktop"
    ]):
        return "app_device_issue"

    if any(word in text for word in [
        "missing", "unavailable", "album", "artist",
        "song", "track", "tracks", "greyed"
    ]):
        return "content_availability"

    if any(word in text for word in [
        "ad", "ads", "advertisement", "offer", "promotion",
        "deal", "discount"
    ]):
        return "ads_promotions"

    return "general_support"


df = pd.read_csv(INPUT_PATH)

df["intent"] = df["text"].apply(classify_intent)

print("Intent distribution:")
print(df["intent"].value_counts())

df.to_csv(OUTPUT_PATH, index=False)

print("\nSaved:", OUTPUT_PATH)