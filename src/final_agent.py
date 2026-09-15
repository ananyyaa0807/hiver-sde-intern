import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_PATH = "data/intent_model.pkl"
RESOLUTION_PATH = "data/spotify_resolutions.csv"


model = joblib.load(MODEL_PATH)

resolutions = pd.read_csv(RESOLUTION_PATH)

resolutions["customer_message"] = (
    resolutions["customer_message"].fillna("")
)

resolutions["brand_reply"] = (
    resolutions["brand_reply"].fillna("")
)


vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)

historical_vectors = vectorizer.fit_transform(
    resolutions["customer_message"]
)


def support_agent(message):

    # Intent prediction
    intent = model.predict([message])[0]

    probabilities = model.predict_proba([message])[0]
    confidence = max(probabilities)

    # Historical retrieval
    query_vector = vectorizer.transform([message])

    similarities = cosine_similarity(
        query_vector,
        historical_vectors
    ).flatten()

    best_index = similarities.argmax()
    similarity = float(similarities[best_index])

    evidence = resolutions.iloc[best_index]

    # Escalation policy
    if confidence >= 0.70 and similarity >= 0.45:

        decision = "AUTO-HANDLE"

        reason = (
            "The intent prediction is confident and a strong "
            "historical SpotifyCares resolution was found."
        )

    else:

        decision = "ESCALATE"

        reason = (
            "The model lacks sufficient confidence or the "
            "historical evidence match is weak."
        )

    return {
        "intent": intent,
        "confidence": round(confidence, 3),
        "similarity": round(similarity, 3),
        "decision": decision,
        "reason": reason,
        "draft_reply": evidence["brand_reply"],
        "evidence_customer": evidence["customer_message"],
        "evidence_reply": evidence["brand_reply"]
    }


# ---------------------------------------------------------
# Interactive testing
# ---------------------------------------------------------

print("=" * 70)
print("SPOTIFY SUPPORT AGENT")
print("=" * 70)

while True:

    message = input(
        "\nEnter a customer message (or type 'exit'): "
    )

    if message.lower() == "exit":
        break

    result = support_agent(message)

    print("\nIntent:", result["intent"])
    print("Confidence:", result["confidence"])
    print("Evidence similarity:", result["similarity"])

    print("\nDraft reply:")
    print(result["draft_reply"])

    print("\nDecision:", result["decision"])

    print("Reason:", result["reason"])

    print("\nHistorical evidence:")
    print("Customer:", result["evidence_customer"])
    print("SpotifyCares:", result["evidence_reply"])