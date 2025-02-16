import __init__
from utils.utils import *
import os
import warnings
import torch
from transformers import OPTForCausalLM, TrainingArguments
from trl import DPOConfig, DPOTrainer
from datasets import Dataset
import wandb
warnings.filterwarnings('ignore')
from utils.data_utils import save_all_results, read_saved_results, load_data
from utils.load_model import *
import argparse
os.environ["WANDB_MODE"] = "offline"

def init_train_model(model_name=''):
    model, tokenizer = init_model(model_name, eval=False, low_cpu_mem_usage=False)
    model.enable_input_require_grads()
    if tokenizer.pad_token is None:
        token = tokenizer.convert_ids_to_tokens(2)
        print(f"Token for ID 2: {token}")
        tokenizer.pad_token_id = 2 
    return model, tokenizer

def load_train_data(choose_data_name):
    choose_data = load_data(choose_data_name)
    choose_data = choose_data['train']
    return choose_data

dataset_names = {
    # 'NuminaMath': 'AI-MO/NuminaMath-CoT',
    'openo1':'data/final/OpenO1-SFT-Pro-Filter.jsonl',
    'sky':'data/final/SKY-SFT.jsonl',

    'Bespoke':'bespokelabs/Bespoke-Stratos-17k',
    'NuminaMath': 'data/final/NuminaMath-SFT',
    'Bespoke_dpo':f'data/final/Bespoke_dpo_filter.jsonl',
    'Bespoke_dpo_long':f'data/final/Bespoke_dpo_filter_len_long.jsonl',
    'Bespoke_dpo_short':f'data/final/Bespoke_dpo_filter_len_short.jsonl',
    'Bespoke_dpo_middle':'data/final/Bespoke_dpo_filter_len_middle.jsonl'
}
model_names = {
    'OpenThinker-7B':'open-thoughts/OpenThinker-7B',
    'Deepseek-7b':'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
    'Bespoke-old-7b': 'bespokelabs/Bespoke-Stratos-7B', 

    'Bespoke-7b-3epoch': '/scratch/pbsjobs/wxy320/models/dpo/steps/Bespoke_Qwen2.5-7B-Instruct_sft_lr1e-05/1566', #'bespokelabs/Bespoke-Stratos-7B', 
    'Bespoke-7b': '/scratch/pbsjobs/wxy320/models/dpo/1epoch/Bespoke_Qwen2.5-7B_sft_lr3e-05/checkpoint-522', #'bespokelabs/Bespoke-Stratos-7B', 
    'Instruct-7b':'Qwen/Qwen2.5-7B-Instruct',
    'NuminaMath-7b': set_global(f'./train/models/sft/NuminaMath_Qwen2.5-7B_sft_lr3e-05/checkpoint-522') ,
    'NuminaMath-7b-3epoch': '/scratch/pbsjobs/wxy320/models/dpo/3epochnumath/NuminaMath_Qwen2.5-7B_sft_lr1e-05/checkpoint-1566' ,


    'Bespoke-3b-3epoch': set_global('./train/models/sft/Bespoke_Qwen2.5-3B_sft_lr1e-05/checkpoint-1566'), #'bespokelabs/Bespoke-Stratos-7B', 
    'Bespoke-3b': set_global('./train/models/sft/Bespoke_Qwen2.5-3B_sft_lr3e-05/checkpoint-522'), 
    'Instruct-3b':'Qwen/Qwen2.5-3B-Instruct',
    'NuminaMath-3b': set_global(f'./train/models/sft/NuminaMath_Qwen2.5-3B-Instruct_sft_lr1e-05/checkpoint-522') ,

    'Bespoke-14b-3epoch': set_global('./train/models/sft/Bespoke_Qwen2.5-3B_sft_lr1e-05/checkpoint-1566'), #'bespokelabs/Bespoke-Stratos-7B', 
    'Bespoke-14b': '/scratch/pbsjobs/wxy320/models/dpo/1epoch/Bespoke_Qwen2.5-14B_sft_lr3e-05/checkpoint-522', 
    'Instruct-14b':'Qwen/Qwen2.5-14B-Instruct',

    'Bespoke-32b': 'bespokelabs/Bespoke-Stratos-32B', 
    'Instruct-32b':'Qwen/Qwen2.5-32B-Instruct',
    'NuminaMath-32b': set_global(f'./models/sft/NuminaMath_Qwen2.5-3B_sft_lr1e-5/checkpoint-2633') ,
}

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
    parser.add_argument('--deepspeed', type=str, default="./train/deepseed/zero3_config2.json") # 
    parser.add_argument('--local_rank', type=int, default=0)
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()
    lr, beta = args.lr, args.beta
    model_name = model_names[args.model] # # 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B' # 
    MAX_LENGTH = args.MAX_LENGTH # tokenizer.model_max_length
    dataset_name = set_global(dataset_names[args.dataset]) # set_global(f'data/final/NuminaMath-CoT_O1_Qwq{i+1}.jsonl')
    wandb_name = f"{args.model}_{args.dataset}_dpo_lr{lr}_beta{beta}"

    training_config = DPOConfig(
        save_only_model=True,
        output_dir=set_global(f"./train/models/dpo/1epoch/{wandb_name}"),
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
        wandb.init(project="MiniMind-Full-R1", name=wandb_name)   

    dpo_trainer = DPOTrainer(
        model,
        ref_model=ref_model,
        args=training_config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    dpo_trainer.train()