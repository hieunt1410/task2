from settings import *
from preprocess import build_dataset

import torch
from torch.utils.data import Dataset
import copy


class SiameseProcessor:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, item):
        query, paragraph, label, _ = item

        # not specifying return_tensors here. we do not want to wrap an extra layer around the original list
        encoded_query = self.tokenizer(
            query, padding="max_length", truncation=True, max_length=MAX_SEQUENCE_LENGTH
        )
        encoded_paragraph = self.tokenizer(
            paragraph, padding="max_length", truncation=True, max_length=MAX_SEQUENCE_LENGTH
        )

        encoded_query = {  # list to tensor
            "input_ids": torch.LongTensor(encoded_query["input_ids"]).cuda(),
            "attention_mask": torch.LongTensor(encoded_query["attention_mask"]).cuda(),
        }

        encoded_paragraph = {  # list to tensor
            "input_ids": torch.LongTensor(encoded_paragraph["input_ids"]).cuda(),
            "attention_mask": torch.LongTensor(encoded_paragraph["attention_mask"]).cuda(),
        }

        return encoded_query, encoded_paragraph, label


class MyDataSet(Dataset):
    def __init__(
        self, tokenizer, dataset_path, num_pairs_per_batch, ns_strategy, 
        year, training_samples_file=None
    ) -> None:
        super().__init__()
        self.data = build_dataset(dataset_path, year, training_samples_file)
        self.processor = SiameseProcessor(tokenizer)
        self.num_pairs_per_batch = num_pairs_per_batch
        self.ns_strategy = ns_strategy

        self.ps = {
            sample["id"]: [pos["id"] for pos in sample["pos_candidates"]]
            for sample in self.data
        }
        self.ps_iter = copy.deepcopy(self.ps)
        self.ns = {
            sample["id"]: [neg["id"] for neg in sample["neg_candidates"]]
            for sample in self.data
        }
        self.ns_iter = copy.deepcopy(self.ns)
        
        self.training_data = self.create_training_dataset()

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, index):
        data = self.training_data[index]
        return self.processor(data)

    def create_training_dataset(self):
        if self.ns_strategy == "hard":
            return self.create_hard_training_dataset()
        else:
            raise ValueError(f"Invalid negative sampling strategy: {self.ns_strategy}")
        
    def create_hard_training_dataset(self):
        training_data = []
        for sample in self.data:
            batch = []
            for cand in sample["pos_candidates"]:
                batch.append((sample["text"], cand["text"], 1, 1.0))
            num_neg_pairs = max(self.num_pairs_per_batch - len(sample["pos_candidates"]), 0)

            for cand in sample["neg_candidates"][:num_neg_pairs]:
                batch.append((sample["text"], cand["text"], 0, 1.0))

            training_data.append(batch)
        return training_data
            