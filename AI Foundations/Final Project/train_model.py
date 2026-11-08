from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "sample_tickets.csv"
MODEL_PATH = PROJECT_DIR / ".artifacts" / "ticket_classifier.joblib"


def build_model():
    return Pipeline(
        [
            ("vectorizer", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def train():
    tickets = pd.read_csv(DATA_PATH).drop_duplicates(subset=["text", "category"])
    tickets = tickets.dropna(subset=["text", "category"])
    tickets["text"] = tickets["text"].str.strip()
    tickets = tickets[tickets["text"] != ""]

    train_text, test_text, train_labels, test_labels = train_test_split(
        tickets["text"],
        tickets["category"],
        test_size=0.25,
        random_state=42,
        stratify=tickets["category"],
    )

    evaluation_model = build_model()
    evaluation_model.fit(train_text, train_labels)
    predictions = evaluation_model.predict(test_text)
    print(f"Holdout accuracy: {accuracy_score(test_labels, predictions):.2f}")
    print(classification_report(test_labels, predictions, zero_division=0))

    final_model = build_model()
    final_model.fit(tickets["text"], tickets["category"])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    train()
