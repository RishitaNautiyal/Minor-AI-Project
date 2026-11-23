import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Convert a local UCI SMS Spam Collection CSV for Scam Checker training."
    )
    parser.add_argument("--input", type=Path, default=Path("spam.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/uci.jsonl"))
    args = parser.parse_args()

    dataframe = pd.read_csv(
        args.input,
        encoding="latin-1",
        usecols=["v1", "v2"],
    ).rename(columns={"v1": "label", "v2": "text"})
    dataframe["label"] = dataframe["label"].map(
        {"spam": "spam", "ham": "legitimate"}
    )
    dataframe = dataframe.dropna(subset=["label", "text"])
    dataframe["text"] = dataframe["text"].astype(str).str.strip()
    dataframe = dataframe[dataframe["text"] != ""]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_json(
        args.output,
        orient="records",
        lines=True,
        force_ascii=False,
    )
    print(f"Converted {len(dataframe)} records to {args.output}")


if __name__ == "__main__":
    main()
