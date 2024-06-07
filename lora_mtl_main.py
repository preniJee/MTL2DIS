from sklearn.metrics import f1_score
import torch.nn as nn
import torch
import numpy as np
import os
import json

from tqdm import tqdm
import pandas as pd
from collections import defaultdict

from mtl_dataloader import MTLTasks
from mtl_main import evaluate
from configs import get_args, merge_args_into_config
from peft import LoraConfig
from transformers import (LlamaConfig, AutoTokenizer)
from utils import (create_logger, get_hp_tuning_result_path, set_seed, load_yaml_file,
                   get_mtl_best_model_path, keep_best_model, get_base_path)
import wandb

from lora_mtl_model import MTLPeftLlamaForSequenceClassification

from torch.optim.lr_scheduler import ReduceLROnPlateau, ConstantLR, ExponentialLR


def merge_args_into_config(args, config):
    config.tasks = args.mtl_tasks.split(',')


def lora_mtl_train(model, batch, device, optimizer):

    (model_task_id, model_task_name), batch = batch
    
    model.set_classification_head(model_task_name)
    # import IPython; IPython.embed()
    
    input_ids, attention_mask, labels = batch
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()

    outputs = model(input_ids=input_ids,
                    attention_mask=attention_mask, labels=labels, return_dict=True)
    
    loss = outputs.loss

    loss.backward()
    optimizer.step()

    return model, outputs, labels


def run(args, result_path):

    logger = create_logger(save_path=os.path.join(result_path, "log.log"))

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, add_prefix_space=True, device=device)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    llama_config = LlamaConfig.from_pretrained(args.model_name)

    merge_args_into_config(args, llama_config)

    print("start loading model")
    model = MTLPeftLlamaForSequenceClassification.from_pretrained(args.model_name,
                                                                  config=llama_config)
    print('model loaded')

    loraconfig = LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias="none",
        target_modules=[
            "q_proj",
            "v_proj"]
 
    ) #        
    model.create_peft_model(loraconfig)

  


    logger.info(model.model.print_trainable_parameters())

    model.to(device)

    # create mtl tasks
    mtl_tasks = args.mtl_tasks.split(",")
    main_tasks = MTLTasks(args=args, mtl_tasks=mtl_tasks,
                          tokenizer=tokenizer, few_shot=False)

    mtl_train_dataloader = main_tasks.get_mtl_dataloader(
        split='train')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay, )

    # scheduler = ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    # scheduler = ConstantLR(optimizer)

    # Initialize variables to keep track of the best model and its validation loss
    best_avg_val_f1 = float('-inf')
    best_loss = float('inf')
    best_result_dict = defaultdict()

    # early stopping
    counter = 0
    early_stop = False

    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")

        log_dict = {}

        model.train()

        losses = []
        all_preds = []
        all_labels = []

        for step, batch in enumerate(tqdm(mtl_train_dataloader)):

            model, outputs , labels = lora_mtl_train(model, batch, device, optimizer)

            loss = outputs.loss

            # Get the predictions
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            # Append predictions and true labels for F1 score calculation
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            losses.append(loss.item())



            if ((step + 1) % args.eval_steps == 0) or ((step + 1) == len(mtl_train_dataloader)):
                loss = np.mean(losses)
                
                f1_train = f1_score(all_labels, all_preds, average='binary')
                logger.info(
                    f"Epoch {epoch +1} Step {step+1}/{len(mtl_train_dataloader)}: Train Loss: {loss:.4f}, train f1 : {f1_train}, LR: {optimizer.param_groups[0]['lr']:.8f}")

                # validation loop

                model.eval()
                task_iterator = main_tasks.get_dataloader_sequence_iterator()
                result_dict = evaluate(
                    task_iterator, model, device, "val", logger)

                # scheduler.step(result_dict['avg_val_loss'])
                # scheduler.step()

                log_dict.update(result_dict)
                log_dict.update({"train_loss": loss})
                log_dict.update({"train_f1": f1_train})
                
                # Check if the current model has a lower validation loss than the best model
                if result_dict['avg_val_f1'] > best_avg_val_f1:
                    best_avg_val_f1 = result_dict['avg_val_f1']
                    best_result_dict = result_dict
                    best_result_dict['best_epoch'] = epoch + 1
                    best_result_dict['best_step'] = step + 1

#                   # get test results
                    test_task_iterator = main_tasks.get_dataloader_sequence_iterator()
                    test_result_dict = evaluate(
                        test_task_iterator, model, device, "test", logger)
                    
                    merged_model = model.merge_and_unload_model()
                    merged_model.save_pretrained(f"{result_path}/model")

                if result_dict['avg_val_loss'] < best_loss:
                    best_loss = result_dict['avg_val_loss']
                    counter = 0

                else: 
                    counter += 1
                    if counter == args.patience:
                        early_stop = True
                        logger.info(f"Early stopping at epoch {epoch+1} step {step+1}")
                        break

                wandb.log(log_dict)

                losses = []
                all_preds = []
                all_labels = []
                model.train()

      
        if early_stop:
                logger.info(f"Early stopping at epoch {epoch+1} step {step+1}")
                break
   

    config_dict = {'lr': args.lr, 'train_batch_size': args.train_batch_size, "epochs": args.epochs,
                   'balance_ratio': args.balance_ratio, 'weight_decay': args.weight_decay,
                   'lora_alpha': args.lora_alpha, 'lora_dropout': args.lora_dropout,
                   'lora_r': args.lora_r}

    best_result_dict.update(config_dict)

    with open(f"{result_path}/{args.val_result_file_name}", "w") as report_file:
        json.dump(best_result_dict, report_file, indent=4)
    
    with open(f"{result_path}/test_result.json", "w") as report_file:
        json.dump(test_result_dict, report_file, indent=4)
    # import IPython; IPython.embed()
    
    
def sweep_main():

    model_with_current_hps = get_hp_tuning_result_path(args)
    if not os.path.exists(os.path.join(model_with_current_hps, 'model')):
        wandb_run = wandb.init(tags=[
            f"DATASET_{args.dataset}",
            # f"tasks_{args.mtl_tasks}",
            f"budget{args.budget}"], project="MTL2DIS")
        config = wandb.config
        args.lr = config.lr
        args.weight_decay = config.weight_decay
        args.lora_alpha = config.lora_alpha
        args.lora_r = config.lora_r
        args.lora_dropout = config.lora_dropout
        args.epochs = config.epochs

        # args.balance_ratio = config.balance_ratio

        wandb.log({"lr": args.lr, "train_batch_size": args.train_batch_size,
                   'balance_ratio': args.balance_ratio, 'tasks': args.mtl_tasks})

        run(args, model_with_current_hps)

        wandb_run.finish()
        
        
def main():

    result_path = get_hp_tuning_result_path(args)
    if not os.path.exists(os.path.join(result_path, 'model')):
        wandb_run = wandb.init(name=f"{args.mtl_tasks}_lr_{args.lr}_epochs_{args.epochs}",
                               tags=[f"model_{args.model_name}",
                                     f"lora_alpha_{args.lora_alpha}",
                                     f"lora_r_{args.lora_r}",
                                     f"lora_dropout_{args.lora_dropout}",
                                     f"DATASET_{args.dataset}",
                                     f"budget{args.budget}"],
                               project="MTL2DIS")

        wandb.log({"lr": args.lr, "train_batch_size": args.train_batch_size,
                   'balance_ratio': args.balance_ratio, 'tasks': args.mtl_tasks})

        run(args, result_path)

        wandb.finish()


if __name__ == "__main__":

    args = get_args()
    combinations = args.mtl_tasks.split("-")
    for comb in combinations:
        args.mtl_tasks = comb

        # check if the best mtl model for currect combination of tasks exist then skip running
        best_model_for_current_tasks = get_mtl_best_model_path(args)

        if not os.path.exists(best_model_for_current_tasks):

            set_seed(args.seed)

            if args.run_sweep:
                sweep_config = load_yaml_file("yaml_configs/llama_hp.yaml")

                # sweep_config = load_yaml_file(f"yaml_configs/mtl_12_sweep_configs/{args.mtl_tasks}.yml")
                sweep_config['name'] = f"{args.dataset}_{args.budget}_{args.mtl_tasks}"
                sweep_id = wandb.sweep(sweep_config, project="MTL2DIS")
                # run the sweep
                wandb.agent(sweep_id, function=sweep_main)

                keep_best_model(args)
            else:
                if args.load_hp:
                    raise NotImplementedError

                main()
                keep_best_model(args)

        else:
            print(
                f"Best {args.model_name} for {args.mtl_tasks} already exists!")
