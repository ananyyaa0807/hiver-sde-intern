# AI Support Agent for SpotifyCares

An AI-assisted customer support agent built for the Hiver SDE Intern 2027 take-home assessment.

The system uses historical customer-support conversations from the Customer Support on Twitter dataset to:

1. Classify incoming customer messages into support intents.
2. Retrieve similar historical SpotifyCares resolutions.
3. Draft a response grounded in the retrieved historical response.
4. Decide whether the request should be auto-handled or escalated to a human.

## 1. Dataset

Primary dataset:

Customer Support on Twitter (Kaggle)

The dataset contains customer and brand tweets connected through tweet IDs.

For this project, SpotifyCares was selected as the target brand.

The raw dataset is intentionally excluded from Git using `.gitignore`.

## 2. Project Pipeline

Customer message
→ Intent classification
→ Historical resolution retrieval
→ Evidence-grounded draft reply
→ Auto-handle / human escalation

## 3. Intent Categories

The project uses six support intents:

- `playback_issue`
- `account_subscription`
- `app_device_issue`
- `content_availability`
- `ads_promotions`
- `general_support`

## 4. Data Preparation

The original dataset was processed to identify customer messages that received direct responses from SpotifyCares.

Results:

- SpotifyCares brand replies: 43,265
- Customer messages with direct brand replies: 41,734
- Usable customer messages: 41,585
- Historical customer/reply pairs: 43,092

## 5. Intent Classification

Model:

TF-IDF + Logistic Regression

The initial automatically labelled training dataset contains 41,585 customer messages.

### Golden evaluation set

A 200-example evaluation set was sampled using a fixed random seed.

The examples were reviewed and labelled using the six project intents.

### Results

| Method | Accuracy |
|---|---:|
| Majority-class baseline | 38% |
| Keyword-rule baseline | 89% |
| TF-IDF + Logistic Regression | 83% |

The TF-IDF model substantially outperforms the trivial majority baseline but does not outperform the keyword baseline.

This is an important finding rather than a result that is hidden.

## 6. Support Agent

The support agent combines:

### Intent classification

Predicts the most likely support intent.

### Historical retrieval

Uses TF-IDF similarity to find historically similar customer messages and retrieves the corresponding SpotifyCares response.

### Escalation

The current policy auto-handles a request only when:

- intent confidence >= 0.70
- historical similarity >= 0.45

Otherwise, the request is escalated.

The escalation reason is explicitly returned to the user.

## 7. Example

Input:

"My Spotify app keeps crashing on my iPhone"

Example output:

- Intent: `app_device_issue`
- High intent confidence
- Strong historical similarity
- Decision: `AUTO-HANDLE`

The retrieved historical SpotifyCares response is used as the basis for the draft.

For an unusual message with weak historical evidence, the system can instead return:

- Decision: `ESCALATE`
- Reason: insufficient confidence or weak historical evidence

## 8. What "Good" Means

A useful support agent should not only classify messages accurately.

It should also:

- ground replies in historical brand behaviour
- avoid inventing unsupported solutions
- recognize when evidence is weak
- escalate uncertain cases
- provide an explicit reason for escalation
- be reproducible and measurable

## 9. Failure Modes

The current system has several known weaknesses.

### 1. Keyword baseline outperforms the ML classifier

The keyword baseline achieved 89% accuracy compared with 83% for TF-IDF + Logistic Regression.

Hypothesis: the training labels were initially generated using heuristic rules, which may cause the learned model to reproduce noisy or overlapping categories.

### 2. Playback issues are difficult to classify

The golden evaluation showed particularly weak performance for `playback_issue`.

Hypothesis: playback-related messages often overlap with app/device and content problems.

### 3. Ads/promotions can be confused with general support

Short messages about advertisements may contain little distinctive vocabulary.

### 4. Content availability can be ambiguous

A customer saying that a song is unavailable may be asking about licensing, search, playlists, or another issue.

### 5. Similarity retrieval can return a plausible but imperfect historical response

A high lexical similarity score does not guarantee that the historical situation is identical.

This is why retrieval evidence and escalation thresholds are important.

## 10. What Is Misleading About My Headline Number?

The 83% golden-set accuracy should not be interpreted as a complete measure of support-agent quality.

First, the golden set contains only 200 examples.

Second, the initial training labels were created using heuristic rules, meaning the training data is not equivalent to a fully human-labelled corpus.

Third, intent accuracy does not measure whether the retrieved response is actually helpful, safe, or appropriate.

Therefore, classification accuracy is only one component of system quality.

## 11. What Was Not Built

To keep the project focused, the following were not built:

- a production web application
- live Twitter integration
- automated customer messaging
- a large language model trained from scratch
- production authentication
- human-agent dashboard
- real-time monitoring infrastructure

The focus was on demonstrating the core decision-making pipeline and evaluating it.

## 12. Reproduction

Create and activate a Python virtual environment:

```bash
python -m venv .venv