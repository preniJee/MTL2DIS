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

from fewshot_dataloader import load_data, get_dataset
from configs import get_args
from utils import get_fewshot_result_path, create_logger


def get_weighted_sampler(args, data):
    pos_ratio = np.sum(data.targets) / len(data)
    perfect_balance_weights = [
        1.0/(1-pos_ratio), 1.0/pos_ratio]
    class_wieghts = [(1-args.balance_ratio)*perfect_balance_weights[0],
                     args.balance_ratio*perfect_balance_weights[1]]
    sample_weights = [class_wieghts[t]
                      for t in data.targets]

    w_sampler = WeightedRandomSampler(
        sample_weights, len(data.targets), replacement=True)
    return w_sampler


def few_shot_train(args, model, train_dataloader, device, optimizer):
    losses = []
    for batch in train_dataloader:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model.forward(input_ids=input_ids,
                                attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    return model, np.mean(losses)


def evaluate(data_loader, model, device):
    predictions = []
    test_labels = []
    test_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            # TODO set classification head
            outputs = model.forward(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            test_loss += loss.item()

            logits = outputs.logits
            predicted = torch.argmax(logits, dim=1)
            predictions.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    # import IPython; IPython.embed()
    avg_loss = test_loss / len(data_loader)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, predictions, average='binary')
    auc = roc_auc_score(test_labels, predictions)
    return [precision, recall, f1, auc, test_loss]


# Generate and save the evaluation result

def run_fewshot(args, base_models_path,  task_result_path):

    logger = create_logger(save_path=os.path.join(task_result_path, "log.log"))
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"running for {args.few_shot_task} for {args.k_shot}  shots")
    config = RobertaConfig.from_pretrained(
        args.model_name, num_labels=2)  # add label2id id2label!

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    df_train = load_data(args, 'train', annotators=args.few_shot_task)
    df_val = load_data(args, 'val', annotators=args.few_shot_task)
    df_test = load_data(args, 'test',  annotators=args.few_shot_task)

    few_shot_val_dataset = get_dataset(
        args, df_val, split='val', tokenizer=tokenizer)
    val_data_loader = DataLoader(
        few_shot_val_dataset, shuffle=True, batch_size=32)

    few_shot_test_dataset = get_dataset(
        args, df_test, split='test', tokenizer=tokenizer)
    test_data_loader = DataLoader(
        few_shot_test_dataset, shuffle=True, batch_size=32)

    # device = torch.device("cuda:3")

    for root, dirs, files in os.walk(base_models_path,):
        for dir in dirs:
            if dir.startswith('mtl') and not args.few_shot_task in dir:
                result_path = os.path.join(task_result_path, dir)
                if os.path.exists(os.path.join(result_path, "model")):
                    continue

                model_path = os.path.join(base_models_path, dir, 'best_model')
                model = RobertaForSequenceClassification.from_pretrained(
                    model_path, config=config)

                mtl_tasks = dir.split('_')[1]
                few_shot_dataset = get_dataset(
                    args, df_train, split='train', tokenizer=tokenizer,
                    mtl_tasks=mtl_tasks, sample_strategy=args.few_shot_sample_strategy, logger=logger)

                # TODO: weighted sampler to handle heavy dataset imbalance
                pos_ratio = np.sum(few_shot_dataset.targets) / \
                    len(few_shot_dataset)
                if args.balance_ratio > 0 and pos_ratio < args.balance_ratio:
                    w_sampler = get_weighted_sampler(args, few_shot_dataset)
                    train_data_loader = DataLoader(
                        few_shot_dataset, batch_size=64, sampler=w_sampler)
                else:
                    train_data_loader = DataLoader(
                        few_shot_dataset, batch_size=64, shuffle=True)

                model.to(device)
                optimizer = AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)

                # freeze roberta layers
                if args.freeze_roberta:
                    for param in model.roberta.parameters():
                        param.requires_grad = False
                best_val_f1 = float('-inf')

                best_result = {'precision': 0, 'recall': 0,
                               'f1': 0, 'auc': 0, 'loss': 0}
                logger.info(
                    f"\n running for model on MTL tasks : {mtl_tasks} : \n")
                for epoch in range(args.epochs):
                    model.train()
                    model, train_loss = few_shot_train(
                        args, model, train_data_loader, device, optimizer)

                    model.eval()
                    precision, recall, f1, auc, val_loss = evaluate(
                        val_data_loader, model, device)

                    if f1 > best_val_f1:
                        best_val_f1 = f1
                        best_result['precision'] = precision
                        best_result['recall'] = recall
                        best_result['f1'] = f1
                        best_result['auc'] = auc
                        best_result['loss'] = val_loss
                        model.save_pretrained(f"{result_path}/model")
                        logger.info(
                            f"epoch: {epoch} train_loss: {train_loss} val_loss: {val_loss} val_f1: {f1}")

                with open(f"{result_path}/{args.val_result_file_name}", "w") as report_file:
                    json.dump(best_result, report_file, indent=4)

                # evaluate on test set
                best_model = RobertaForSequenceClassification.from_pretrained(
                    f"{result_path}/model", config=config)
                best_model.to(device)
                test_precision, test_recall, test_f1, test_auc, test_loss = evaluate(
                    test_data_loader, best_model, device)
                test_result = {'precision': test_precision, 'recall': test_recall,
                               'f1': test_f1, 'auc': test_auc, 'loss': test_loss}
                with open(f"{result_path}/test_result.json", "w") as report_file:
                    json.dump(test_result, report_file, indent=4)


if __name__ == "__main__":
    args = get_args()
    base_models_path = f"results/{args.model_name}/{args.dataset}/{args.label}/seed_{args.seed}/budget_{args.budget}/mtl_{args.n_mtl_tasks}"
    taks_result_path = get_fewshot_result_path(args)

    run_fewshot(args, base_models_path, taks_result_path)
