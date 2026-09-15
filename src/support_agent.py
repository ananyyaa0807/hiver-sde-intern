import pandas as pd
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_PATH = "data/intent_model.pkl"
RESOLUTION_PATH = "data/spotify_resolutions.csv"


# ---------------------------------------------------------
# Load model and historical resolutions
# ---------------------------------------------------------

model = joblib.load(MODEL_PATH)

resolutions = pd.read_csv(RESOLUTION_PATH)

resolutions["customer_message"] = (
    resolutions["customer_message"]
    .fillna("")
)

resolutions["brand_reply"] = (
    resolutions["brand_reply"]
    .fillna("")
)


# ---------------------------------------------------------
# Build retrieval index
# ---------------------------------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)

historical_vectors = vectorizer.fit_transform(
    resolutions["customer_message"]
)


# ---------------------------------------------------------
# Support agent
# ---------------------------------------------------------

def support_agent(message):

    # 1. Predict intent
    intent = model.predict([message])[0]

    # 2. Find similar historical customer messages
    query_vector = vectorizer.transform([message])

    similarities = cosine_similarity(
        query_vector,
        historical_vectors
    ).flatten()

    top_indices = similarities.argsort()[-3:][::-1]

    top_examples = resolutions.iloc[top_indices].copy()

    # Best similarity score
    best_score = similarities[top_indices[0]]

    # 3. Choose the most similar historical reply
    best_reply = top_examples.iloc[0]["brand_reply"]

    # 4. Decide whether to auto-handle or escalate
    if best_score >= 0.45:
        decision = "AUTO-HANDLE"
        reason = (
            "A sufficiently similar historical SpotifyCares "
            "resolution was found."
        )
    else:
        decision = "ESCALATE"
        reason = (
            "No sufficiently similar historical resolution "
            "was found, so human review is recommended."
        )

    return {
        "intent": intent,
        "similarity": round(float(best_score), 3),
        "reply": best_reply,
        "decision": decision,
        "reason": reason,
        "evidence": top_examples[
            ["customer_message", "brand_reply"]
        ]
    }


# ---------------------------------------------------------
# Test messages
# ---------------------------------------------------------

test_messages = [
    "My Spotify shuffle and repeat buttons are not working",
    "The Spotify app keeps crashing on my iPhone",
    "I paid for Premium but my account is still showing Free",
    "Some of my favorite songs are missing from Spotify",
    "I keep getting annoying advertisements",
]


for message in test_messages:

    print("\n" + "=" * 80)
    print("CUSTOMER:")
    print(message)

    result = support_agent(message)

    print("\nINTENT:")
    print(result["intent"])

    print("\nSIMILARITY:")
    print(result["similarity"])

    print("\nDRAFT REPLY:")
    print(result["reply"])

    print("\nDECISION:")
    print(result["decision"])

    print("\nREASON:")
    print(result["reason"])

    print("\nHISTORICAL EVIDENCE:")

    for _, example in result["evidence"].iterrows():

        print("\nCustomer:", example["customer_message"])
        print("Brand:", example["brand_reply"])