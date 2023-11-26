from transformers import RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, AutoTokenizer, AdamW
from mtl_model import MTLRobertaForSequenceClassification
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import json
import random
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler, Dataset
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from utils import get_base_path
import os
import sys
sys.path.append('../')

# from mtl_dataloader import MTLTasks


class CustomDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.targets = labels

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        input_id = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]
        label = self.targets[idx]

        return input_id, attention_mask, label


def few_shot_sample(args, df, sample_strategy, mtl_tasks):
    if sample_strategy == 'mv':
        df_mtl_tasks = load_data(args, 'train', mtl_tasks)
        # sample the data which the mtl was trained with
        mtl_task_sample_size = (
            args.budget - (args.n_fewshot_tasks * args.k_shot)) // len(mtl_tasks.split(","))

        # print(f"Sample size: {mtl_task_sample_size}")

        df_mtl_tasks = df_mtl_tasks.sample(mtl_task_sample_size,
                                           random_state=args.seed).reset_index(drop=True)

        subset_label_0 = df_mtl_tasks[df_mtl_tasks[args.label] == 0]
        subset_label_1 = df_mtl_tasks[df_mtl_tasks[args.label] == 1]
        # Sample half from each subset
        sample_1 = subset_label_1.sample(
            min(len(subset_label_1), args.k_shot // 2), random_state=args.seed)
        sample_0 = subset_label_0.sample(
            args.k_shot - len(sample_1), random_state=args.seed)
        sampled_df = pd.concat([sample_1, sample_0]).reset_index()
        sample_ids = sampled_df['id'].tolist()
        df = df[df['id'].isin(sample_ids)].reset_index(drop=True)
    else:
        raise NotImplementedError
    return df


def load_data(args, split, annotators):
    data_file = f"data/{args.dataset}/{args.label}/annotators/{annotators}/{split}.csv"
    data_file = os.path.join(get_base_path(), data_file)
    df = pd.read_csv(data_file)
    return df


def get_dataset(args, df, split, tokenizer, mtl_tasks=None, sample_strategy='mv', logger= None):

    if split == 'train':
        df = few_shot_sample(args, df, sample_strategy, mtl_tasks)
        logger.info(f"distribution of {split} data: {df[args.label].value_counts()}")
        # df = df.sample(args.k_shot,random_state=args.seed).reset_index(drop=True)
    texts = df[args.text_col].tolist()
    labels = df[args.label].tolist()
    encoded_texts = tokenizer(
        texts, padding=True, truncation=True, return_tensors="pt")
    input_ids = encoded_texts["input_ids"]
    attention_mask = encoded_texts['attention_mask']
    # labels = torch.tensor(labels)
    dataset = CustomDataset(input_ids, attention_mask, labels)

    return dataset
