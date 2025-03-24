import os

import lightning as L
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class SentimentAnalysisDataset(Dataset):
    def __init__(self, csv_path: str, max_length: int = 512):
        self.data = pd.read_csv(csv_path)
        texts = self.data["text"].tolist()
        self.labels = self.data["label"].tolist()

        all_text = "".join(texts)
        self.chars = sorted(list(set(all_text)))
        self.stoi = {ch: i + 1 for i, ch in enumerate(self.chars)}
        self.itos = {i + 1: ch for i, ch in enumerate(self.chars)}
        self.max_length = max_length
        self.vocab_size = len(self.stoi) + 1

    def encode(self, text):
        encoded = [self.stoi.get(ch, 0) for ch in text]
        if len(encoded) > self.max_length:
            return encoded[: self.max_length]
        else:
            return encoded + [0] * (self.max_length - len(encoded))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]["text"]
        label = self.labels[idx]
        encoded_text = self.encode(text)

        input_tensor = torch.tensor(encoded_text, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return input_tensor, label_tensor


class SentimentAnalysisDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule class for loading
    """

    def __init__(self, data_dir: str, batch_size: int = 32):
        """
        Initializes the DataModule by setting the data directory and batch size.

        Args:
            data_dir (str): Directory containing the preprocessed CSV files.
            batch_size (int): Number of samples per batch.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train = SentimentAnalysisDataset(os.path.join(self.data_dir, "train.csv"))
        self.val = SentimentAnalysisDataset(os.path.join(self.data_dir, "val.csv"))
        self.test = SentimentAnalysisDataset(os.path.join(self.data_dir, "test.csv"))

    def train_dataloader(self):
        """
        Returns a DataLoader for the training dataset.

        Returns:
            DataLoader: Training DataLoader.
        """
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        """
        Returns a DataLoader for the validation dataset.

        Returns:
            DataLoader: Validation DataLoader.
        """
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        """
        Returns a DataLoader for the validation dataset.

        Returns:
            DataLoader: Validation DataLoader.
        """
        return DataLoader(self.test, batch_size=self.batch_size)
