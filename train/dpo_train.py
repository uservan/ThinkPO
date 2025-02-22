import __init__
from utils.utils import *
import os
import warnings
from trl import DPOConfig, DPOTrainer
import wandb
from utils.load_model import *
import argparse
from train.names import *

warnings.filterwarnings('ignore')
os.environ["WANDB_MODE"] = "offline"

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=5e-7)
    parser.add_argument('--beta', type=float, default=0.05)
    parser.add_argument('--model', type=str, default='Bespoke-7b')
    parser.add_argument('--dataset', type=str, default='Bespoke_dpo')
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--MAX_LENGTH', type=int, default=1024 * 8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=12)
    parser.add_argument('--deepspeed', type=str, default=None) # 
    parser.add_argument('--local_rank', type=int, default=0)
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()
    lr, beta = args.lr, args.beta
    model_name = model_names[args.model] 
    MAX_LENGTH = args.MAX_LENGTH
    dataset_name = set_global(dataset_names[args.dataset]) 
    wandb_name = f"{args.model}_{args.dataset}_dpo_lr{lr}_beta{beta}"

    training_config = DPOConfig(
        save_only_model=True,
        output_dir=set_global(f"./train/models/dpo/{wandb_name}"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        save_total_limit=1,
        num_train_epochs=args.epoch,
        report_to="wandb",
        save_strategy='epoch',
        logging_steps=1,
        learning_rate=lr,
        beta=beta,
        bf16=True,
        lr_scheduler_type='cosine',
        warmup_ratio = 0.1,
        max_length=MAX_LENGTH,
        deepspeed= set_global(args.deepspeed) if args.deepspeed is not None else None,
    )

    train_dataset = load_train_data(dataset_name)
    model, tokenizer = init_train_model(model_name)
    ref_model = init_train_model(model_name)[0]
    ref_model.eval()
    # ref_model = None

    if args.local_rank == 0:
        wandb.login()  
        wandb.init(project="ThinkPO", name=wandb_name)   

    dpo_trainer = DPOTrainer(
        model,
        ref_model=ref_model,
        args=training_config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    dpo_trainer.train()