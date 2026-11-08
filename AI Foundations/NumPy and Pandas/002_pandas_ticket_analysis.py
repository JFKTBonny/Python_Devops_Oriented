from pathlib import Path

import pandas as pd


DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "Final Project"
    / "data"
    / "sample_tickets.csv"
)


def main():
    tickets = pd.read_csv(DATA_PATH).drop_duplicates()
    tickets["text"] = tickets["text"].fillna("").str.strip()
    tickets = tickets[tickets["text"] != ""]
    tickets["character_count"] = tickets["text"].str.len()

    print("Tickets per category:")
    print(tickets["category"].value_counts().sort_index().to_string())
    print("\nTicket length by category:")
    print(tickets.groupby("category")["character_count"].agg(["mean", "median"]).round(1))


if __name__ == "__main__":
    main()
