from pathlib import Path

import pandas as pd


DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "Final Project"
    / "data"
    / "sample_tickets.csv"
)


def main():
    tickets = pd.read_csv(DATA_PATH).dropna(subset=["text", "category"])
    proportions = tickets["category"].value_counts(normalize=True).sort_index()
    majority_accuracy = tickets["category"].value_counts(normalize=True).max()

    print("Category proportions:")
    print((proportions * 100).round(1).astype(str).add("%").to_string())
    print(f"\nMajority-class baseline accuracy: {majority_accuracy:.1%}")


if __name__ == "__main__":
    main()
