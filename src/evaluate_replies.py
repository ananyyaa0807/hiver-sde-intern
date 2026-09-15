import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RESOLUTIONS_PATH = "data/demo_resolutions.csv"
GOLDEN_PATH = "data/golden_set.csv"
OUTPUT_PATH = "data/reply_evaluation.json"


def find_best_evidence(customer_message, resolutions, vectorizer, matrix):
    query_vector = vectorizer.transform([customer_message])
    similarities = cosine_similarity(query_vector, matrix)[0]

    best_index = similarities.argmax()

    return {
        "customer_text": resolutions.iloc[best_index]["customer_message"],
        "brand_reply": resolutions.iloc[best_index]["brand_reply"],
        "similarity": float(similarities[best_index]),
    }


def automated_evaluation():
    resolutions = pd.read_csv(RESOLUTIONS_PATH)
    golden = pd.read_csv(GOLDEN_PATH)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2
    )

    matrix = vectorizer.fit_transform(
        resolutions["customer_message"].fillna("")
    )

    results = []

    for _, row in golden.iterrows():
        message = str(row["text"])

        evidence = find_best_evidence(
            message,
            resolutions,
            vectorizer,
            matrix
        )

        results.append({
            "tweet_id": str(row["tweet_id"]),
            "customer_message": message,
            "gold_intent": row["gold_intent"],
            "evidence_similarity": evidence["similarity"],
            "historical_customer": evidence["customer_text"],
            "historical_reply": evidence["brand_reply"]
        })

    return results


def llm_judge(results):
    try:
        from openai import OpenAI
    except ImportError:
        print("OpenAI package is not installed.")
        print("Run: pip install openai")
        return None

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.")
        print("Automated evidence evaluation will still be saved.")
        return None

    client = OpenAI()

    judged = []

    # Judge a manageable subset.
    sample = results[:30]

    for item in sample:
        prompt = f"""
You are evaluating an AI customer-support agent.

Customer message:
{item["customer_message"]}

Historical customer message used as evidence:
{item["historical_customer"]}

Historical brand reply:
{item["historical_reply"]}

Evaluate the evidence and proposed historical reply.

Give scores from 1 to 5 for:

1. evidence_relevance:
How relevant is the historical customer message to the new customer message?

2. reply_helpfulness:
How useful would the historical brand reply be for addressing the new customer?

3. grounding:
Does the reply stay grounded in the historical evidence rather than inventing unsupported facts?

Return ONLY valid JSON:

{{
  "evidence_relevance": 1,
  "reply_helpfulness": 1,
  "grounding": 1,
  "reason": "short explanation"
}}
"""

        try:
            response = client.responses.create(
                model="gpt-5.6-luna",
                input=prompt
            )

            text = response.output_text.strip()

            judgment = json.loads(text)

            item["llm_judgment"] = judgment
            judged.append(item)

        except Exception as e:
            item["llm_judgment_error"] = str(e)
            judged.append(item)

    return judged


def main():
    print("=" * 70)
    print("REPLY + EVIDENCE EVALUATION")
    print("=" * 70)

    results = automated_evaluation()

    similarities = [
        r["evidence_similarity"]
        for r in results
    ]

    print()
    print("Examples evaluated:", len(results))
    print(
        "Average evidence similarity:",
        round(sum(similarities) / len(similarities), 3)
    )

    strong_matches = [
        s for s in similarities if s >= 0.45
    ]

    print(
        "Evidence matches >= 0.45:",
        len(strong_matches),
        f"({len(strong_matches) / len(similarities):.1%})"
    )

    judged = llm_judge(results)

    output = {
        "automated_evaluation": results
    }

    if judged is not None:
        output["llm_judged_examples"] = judged

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print()
    print("Saved evaluation to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()