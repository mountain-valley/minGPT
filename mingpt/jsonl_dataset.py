import json
import torch
from torch.utils.data import Dataset
import time
from transformers import GPT2TokenizerFast
from mingpt.utils import CfgNode as CN
import random
import os

start_time = time.time()

# Respect offline mode on compute nodes and prefer fast tokenizer to avoid protobuf dependency
_OFFLINE = os.environ.get("TRANSFORMERS_OFFLINE") == "1" or os.environ.get("HF_HUB_OFFLINE") == "1"
try:
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', local_files_only=_OFFLINE)
except Exception:
    if _OFFLINE:
        # In offline mode but cache missing -> re-raise for a clear failure
        raise
    # Fallback to online fetch if allowed
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

"""
NOTE: Each line in the JSONL file is a JSON object with a 'text' field. 
Each datapoint is the 'text' field and thus a string of text.
"""

class JSONLDataset(Dataset):
    @staticmethod
    def get_default_config():
        C = CN()
        C.file_path = '/nobackup/autodelete/usr/YOUR_RC_USERNAME/pile_data_10.jsonl'
        C.split = 'train'
        C.test_size = 10
        C.block_size = 128
        return C

    def __init__(self, config):
        self.config = config
        self.file_path = config.file_path
        self.split = config.split
        self.test_size = config.test_size
        self.block_size = config.block_size

        with open(self.file_path, 'r') as f:
            lines = f.readlines()
        
        print(f'Length of lines: {len(lines)}')
        prelength = len(lines)
        data = []
        for i, line in enumerate(lines):
            try:
                # Processes the line and extracts the 'text' field
                d = json.loads(line)['text']
                # Skips short text entries
                if len(d) < 10:
                    continue
                data.append(d)
            except json.JSONDecodeError:
                # Silently skip lines that fail to parse
                pass 

        print(f'Length of data: {len(data) / prelength * 100:.2f}%')
        
        # Implements the train/test split (robust to tiny files)
        total = len(data)
        if total == 0:
            raise ValueError(f"No usable lines found in {self.file_path}. "
                             f"Ensure each line has a JSON object with a non-trivial 'text' field.")
        if self.split == 'train':
            split_idx = max(1, total - self.test_size)
            self.data = data[:split_idx]
        else:
            take = min(total, self.test_size)
            self.data = data[-take:] if take > 0 else []

    def __len__(self):
        # Returns the total number of items in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # grab the line at the given index
        line = self.data[idx]
        # tokenize/encode the line
        dix = tokenizer.encode(line)

        # resample short docs
        # TODO: Is there a better way to handle this? Padding?
        if len(dix) < self.block_size + 1:
            return self.__getitem__(random.randrange(len(self)))

        # random in-doc window
        start = random.randint(0, len(dix) - (self.block_size + 1))
        window = dix[start : start + self.block_size + 1]
        
        x = torch.tensor(window[:-1], dtype=torch.long)
        y = torch.tensor(window[1:], dtype=torch.long)
        return x, y

    def get_vocab_size(self):
        return len(tokenizer.get_vocab())

    def get_block_size(self):
        return self.block_size

# file_path = '/nobackup/autodelete/usr/YOUR_RC_USERNAME/pile_data_10.jsonl'
# file_path = 'pile_data_10_first_50000.jsonl'
# # file_path = '100.jsonl'

# def get_elapsed_time():
#     return time.time() - start_time

# # Initialize the dataset with a test size of 1000 lines
# print(f'Making dataset... {get_elapsed_time() / 60} minutes')
# train_dataset = JSONLDataset(file_path, split='train', test_size=10)
# print(len(train_dataset))