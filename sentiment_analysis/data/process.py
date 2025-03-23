import os
import pandas as pd
import re
from typing import List
from nltk.corpus import stopwords
KEEP_COLUMNS = ["customer_sentiment", "conversation"]
STOPWORDS = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
    "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs",
    "themselves", "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "s", "t", "can", "will", "just", "don", "should", "now"
])



def preprocess_text(text: str) -> str:
    """
    Preprocesses the input text by removing unnecessary words, converting to lowercase, removing punctuation,
    and extra whitespaces.

    Args:
        text (str): The raw input text.

    Returns:
        str: The cleaned and preprocessed text.
    """
    text = re.sub(r'\b(Customer|Agent):\b', '', text, flags=re.IGNORECASE)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    #text = " ".join([word for word in text.split() if word not in STOPWORDS])
    return text

def process_data(data: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Processes the input DataFrame by selecting specific columns, applying preprocessing to conversations,
    and saving the cleaned data to a CSV file.

    Args:
        data (pd.DataFrame): The raw input data.
        name (str): The filename to save the processed data.

    Returns:
        pd.DataFrame: The processed data.
    """
    data = data[KEEP_COLUMNS].dropna()
    data["conversation"] = data["conversation"].apply(preprocess_text)
    data["conversation"] = data["conversation"].apply(lambda x: x.replace("customer", ""))
    data["conversation"] = data["conversation"].apply(lambda x: x.replace("agent", ""))
    data.to_csv(os.path.join(os.path.dirname(__file__), "processed", name), index=False)
    # drop the conversation rows that that is in %5 longest conversations
    data_long = data[~data["conversation"].str.len().isin(data["conversation"].str.len().nlargest(int(len(data)*0.05)).values)]
    return data

if __name__ == "__main__":
    train_csv = pd.read_csv(os.path.join(os.path.dirname(__file__), "raw", "train.csv"))
    test_csv = pd.read_csv(os.path.join(os.path.dirname(__file__), "raw", "test.csv"))
    for csv, name in zip([train_csv, test_csv], ["train.csv", "test.csv"]):
        process_data(csv, name)
