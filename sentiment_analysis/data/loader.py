import os
import pandas as pd
from torch.utils.data import Dataset
import torch
import tiktoken
import pytorch_lightning as pl
from typing import Optional
from torch.utils.data import DataLoader


class SentimentAnalysisDataset(Dataset):
    """
    Torch Dataset class for loading sentiment analysis samples from preprocessed conversation data.
    Text is tokenized using GPT-2 tokenizer and padded/truncated to a uniform length.
    """
    def __init__(self, csv_path: str, max_length: int = 256):
        """
        Initializes the dataset from a preprocessed CSV file.

        Args:
            csv_path (str): File path to the processed CSV.
            max_length (int): Maximum token length for model input.
        """
        self.data = pd.read_csv(csv_path)
        self.max_length = max_length
        self.enc = tiktoken.get_encoding("gpt2")
        self.label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.data['label'] = self.data['customer_sentiment'].map(self.label_map)

    def __len__(self) -> int:
        """
        Returns the number of examples in the dataset.

        Returns:
            int: Dataset size.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves tokenized input tensor and target label tensor for a given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of input token tensor and sentiment class label tensor.
        """
        conversation = self.data.iloc[idx]['conversation']
        label = self.data.iloc[idx]['label']
        token_ids = self.enc.encode_ordinary(conversation)
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids += [0] * (self.max_length - len(token_ids))
        input_tensor = torch.tensor(token_ids, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return input_tensor, label_tensor


class SentimentAnalysisDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule class for loading
    """
    
    def __init__(self, data_dir:str, batch_size:int=32):
        """
        Initializes the DataModule by setting the data directory and batch size.

        Args:
            data_dir (str): Directory containing the preprocessed CSV files.
            batch_size (int): Number of samples per batch.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        
    def setup(self, stage: Optional[str] = None):
        """
        Loads the training and validation datasets from CSV files.

        Args:
            stage (Optional[str]): Optional argument to distinguish between training and validation.
        """
        if stage == 'fit' or stage is None:
            self.train = SentimentAnalysisDataset(os.path.join(self.data_dir, 'train.csv'))
            self.val = SentimentAnalysisDataset(os.path.join(self.data_dir, 'val.csv'))
            
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
      
        