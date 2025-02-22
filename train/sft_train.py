from __init__ import *
from utils.utils import *
from transformers import Trainer, TrainingArguments,DataCollatorForSeq2Seq
import wandb
from utils.data_utils import load_data
from utils.load_model import *
import argparse
import os
from train.names import model_names, dataset_names

os.environ['WANDB_MODE'] = 'offline'

def init_train_model(model_name=''):
    model, tokenizer = init_model(model_name=model_names[model_name], eval=False)
    model.enable_input_require_grads()
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        token = tokenizer.convert_ids_to_tokens(2)
        print(f"Token for ID 2: {token}")
        tokenizer.pad_token_id = 2 
    return model, tokenizer

def preprocess_function(example):
    global MAX_LENGTH
    input_ids, attention_mask, labels = [], [], []
    system, conversations = example["system"], example["conversations"]
    targets = conversations[1]['value'].strip()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": conversations[0]['value']}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    instruction, response = tokenizer(inputs+'\n'), tokenizer(targets)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def load_train_data(dataset_name):
    if dataset_name == 'NuminaMath':
        dataset = load_data(set_global(dataset_names[dataset_name]), 'json')
        train_dataset = dataset['train']
    if dataset_name == 'Bespoke':
        dataset = load_data(dataset_names[dataset_name], 'huggingface')
        train_dataset = dataset['train'] 
    return train_dataset.map(preprocess_function, remove_columns=train_dataset.column_names)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--MAX_LENGTH', type=int, default=1024*10)
    parser.add_argument('--model_name', type=str, default='Qwen2.5-14B-Instruct', help='Qwen2.5-3B, Qwen2.5-7B, Qwen2.5-32B')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=96)
    parser.add_argument('--dataset_name', type=str, default='Bespoke', help='Bespoke, NuminaMath')
    parser.add_argument('--deepspeed', type=str, default=None) #
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=1)
    return parser.parse_args(args)

args = parse_args()

lr = args.lr
MAX_LENGTH = args.MAX_LENGTH
model_name = args.model_name
dataset_name = args.dataset_name
wandb_name = f"{dataset_name}_{model_name}_sft_lr{lr}"

model, tokenizer = init_train_model(model_name=model_name)
train_dataset = load_train_data(dataset_name)

if args.local_rank == 0:
    wandb.login()
    wandb.init(project="ThinkPO-SFT", name=wandb_name)
    
training_args = TrainingArguments(
    save_only_model=True,
    output_dir=set_global(f"./train/models/sft/{wandb_name}"),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    save_total_limit=1,
    num_train_epochs=args.epoch,
    learning_rate=lr,
    gradient_checkpointing=True,
    lr_scheduler_type='cosine',
    warmup_ratio=0.1,
    logging_steps=10,
    save_strategy='epoch',
    report_to="wandb",
    bf16=True,
    # dataloader_num_workers=4,
    deepspeed= set_global(args.deepspeed) if args.deepspeed is not None else None,
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
)
trainer.train()
