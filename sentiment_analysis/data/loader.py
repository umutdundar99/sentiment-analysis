import os

import lightning as L
import pandas as pd
import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset


class SentimentCharDataset(Dataset):
    def __init__(self, csv_path: str, max_length: int = 512):
        self.data = pd.read_csv(csv_path)
        texts = self.data["conversation"].tolist()
        self.labels = self.data["customer_sentiment"].tolist()
        all_text = "".join(texts)
        self.chars = sorted(list(set(all_text)))
        self.stoi = {ch: i + 1 for i, ch in enumerate(self.chars)}
        self.itos = {i + 1: ch for i, ch in enumerate(self.chars)}
        self.max_length = max_length
        self.vocab_size = len(self.stoi) + 1
        self.labels = [
            0 if label == "negative" else 1 if label == "neutral" else 2
            for label in self.labels
        ]

    def encode(self, text):
        encoded = [self.stoi.get(ch, 0) for ch in text]
        if len(encoded) > self.max_length:
            return encoded[: self.max_length]
        else:
            return encoded + [0] * (self.max_length - len(encoded))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]["conversation"]
        label = self.labels[idx]
        encoded_text = self.encode(text)

        input_tensor = torch.tensor(encoded_text, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return input_tensor, label_tensor


class SentimentGPT2Dataset(Dataset):
    def __init__(self, csv_path: str, max_length: int = 256):
        self.data = pd.read_csv(csv_path)
        self.max_length = max_length

        self.labels = self.data["customer_sentiment"].tolist()
        self.enc = tiktoken.get_encoding("gpt2")
        self.label_map = {"negative": 0, "neutral": 1, "positive": 2}
        self.data["label"] = self.data["customer_sentiment"].map(self.label_map)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        conversation = self.data.iloc[idx]["conversation"]
        label = self.data.iloc[idx]["label"]
        token_ids = self.enc.encode_ordinary(conversation)
        if len(token_ids) > self.max_length:
            token_ids = token_ids[: self.max_length]
        else:
            token_ids += [0] * (self.max_length - len(token_ids))
        input_tensor = torch.tensor(token_ids, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return input_tensor, label_tensor


class SentimentAnalysisDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule class for loading
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        max_length: int = 512,
        encode_type: str = "char",
    ):
        """
        Initializes the DataModule by setting the data directory and batch size.

        Args:
            data_dir (str): Directory containing the preprocessed CSV files.
            batch_size (int): Number of samples per batch.
            max_length (int): Maximum length of the input sequence.
            encode_type (str): Type of encoding to use. Either "char" or "gpt2".
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        if encode_type == "char":
            self.train = SentimentCharDataset(
                os.path.join(self.data_dir, "train.csv"), max_length
            )
            self.val = SentimentCharDataset(
                os.path.join(self.data_dir, "val.csv"), max_length
            )
            self.test = SentimentCharDataset(
                os.path.join(self.data_dir, "test.csv"), max_length
            )
        else:
            self.train = SentimentGPT2Dataset(
                os.path.join(self.data_dir, "train.csv"), max_length
            )
            self.val = SentimentGPT2Dataset(
                os.path.join(self.data_dir, "val.csv"), max_length
            )
            self.test = SentimentGPT2Dataset(
                os.path.join(self.data_dir, "test.csv"), max_length
            )

    def train_dataloader(self):
        """
        Returns a DataLoader for the training dataset.

        Returns:
            DataLoader: Training DataLoader.
        """
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            # pin_memory=True,
            num_workers=12,
            shuffle=True,
        )

    def val_dataloader(self):
        """
        Returns a DataLoader for the validation dataset.

        Returns:
            DataLoader: Validation DataLoader.
        """
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            # pin_memory=True,
            num_workers=12,
            shuffle=False,
        )

    def test_dataloader(self):
        """
        Returns a DataLoader for the validation dataset.

        Returns:
            DataLoader: Validation DataLoader.
        """
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            # pin_memory=True,
            num_workers=12,
            shuffle=False,
        )
