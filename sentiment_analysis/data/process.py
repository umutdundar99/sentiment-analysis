import os
import re
import shutil
from typing import Optional

import pandas as pd

KEEP_COLUMNS = ["customer_sentiment", "conversation"]


def preprocess_text(text: str) -> str:
    """
    Preprocesses the input text by removing unnecessary words, converting to lowercase, removing punctuation,
    and extra whitespaces.

    Args:
        text (str): The raw input text.

    Returns:
        str: The cleaned and preprocessed text.
    """

    text = text.lower()
    text_splitted = text.split("\n\n")
    text = [i for i in text_splitted if not i.startswith("agent")]
    text = " ".join(text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("customer", "")
    return text


def process_data(
    data: pd.DataFrame, name: str, val_split: Optional[float] = None
) -> pd.DataFrame:
    """
    Processes the input DataFrame by selecting specific columns, applying preprocessing to conversations,
    and saving the cleaned data to a CSV file.

    Args:
        data (pd.DataFrame): The raw input data.
        name (str): The filename to save the processed data.

    Returns:
        pd.DataFrame: The processed data.
    """

    data["conversation"] = data["conversation"].apply(preprocess_text)
    data = data[KEEP_COLUMNS].dropna()

    if val_split is not None:
        max_length = data["conversation"].apply(lambda x: len(x.split()))
        max_length = max_length.quantile(0.95)
        data = data[data["conversation"].apply(lambda x: len(x.split())) < max_length]
        data = data[~(data["conversation"] == "")]
        data_positive = data[data["customer_sentiment"] == "positive"]
        data_others = data[data["customer_sentiment"] != "positive"]
        val_data_positive = data_positive.sample(
            max(1, int(len(data_positive) * val_split)), random_state=42, replace=False
        )
        val_data_others = data_others.sample(
            max(1, int(len(data_others) * val_split)), random_state=42, replace=False
        )
        val_data = pd.concat([val_data_positive, val_data_others])
        train_data = data.drop(val_data.index)
        train_data.to_csv(
            os.path.join(os.path.dirname(__file__), "processed", "train.csv"),
            index=False,
        )
        val_data.to_csv(
            os.path.join(os.path.dirname(__file__), "processed", "val.csv"), index=False
        )
    else:
        data.to_csv(
            os.path.join(os.path.dirname(__file__), "processed", name), index=False
        )
    return data


if __name__ == "__main__":
    train_csv = pd.read_csv(os.path.join(os.path.dirname(__file__), "raw", "train.csv"))
    test_csv = pd.read_csv(os.path.join(os.path.dirname(__file__), "raw", "test.csv"))
    # delete processed folder
    if os.path.exists(os.path.join(os.path.dirname(__file__), "processed")):
        shutil.rmtree(os.path.join(os.path.dirname(__file__), "processed"))
    os.makedirs(os.path.join(os.path.dirname(__file__), "processed"))
    for csv, name in zip([train_csv, test_csv], ["train.csv", "test.csv"]):
        if "train" in name:
            process_data(csv, name, val_split=0.20)
        else:
            process_data(csv, name)
