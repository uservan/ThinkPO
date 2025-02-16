import os
import warnings
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from datasets import load_dataset
from datasets import Dataset
import wandb
warnings.filterwarnings('ignore')

cache_dir = '/users/PDS0352/wyang107/project/LCEG/model_cache'
def Logger(content):
    print(content)
def set_global(path):
    return os.path.join('/users/PDS0352/wyang107/project/LCEG/o1', path)
from huggingface_hub import login
login(token='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')


def init_model(model_from=1, model_name='', special_tokens=[]):
    if model_from == 1:
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, device_map="auto", low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=cache_dir,)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if special_tokens:
        special_tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=cache_dir,)
        add_tokens = []
        for token in special_tokens:
            token_id = special_tokenizer.convert_tokens_to_ids(token)
            if token_id == special_tokenizer.unk_token_id:
                add_tokens.append(token)
            else:
                Logger(f"Token '{token}' is recognized with ID: {token_id}")
        if len(add_tokens) >0:
            special_tokenizer.add_special_tokens({"additional_special_tokens": add_tokens})
            # model.resize_token_embeddings(len(tokenizer))
            Logger(f"Special tokens added: {special_tokens}")
            for token in special_tokens:
                token_id = special_tokenizer.convert_tokens_to_ids(token)
                Logger(f"Token '{token}' is recognized with ID: {token_id}, rather than {tokenizer.unk_token_id}")   
    model.enable_input_require_grads()
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = 2
    return model, tokenizer,special_tokenizer

def load_data(dataset_name):
    if 'NuminaMath' in dataset_name:
        dataset = load_dataset('AI-MO/NuminaMath-CoT', cache_dir=cache_dir)
        train_dataset = dataset['train']
        choose_data = load_dataset('json', data_files=dataset_name)
        choose_data = choose_data['train']
        new_dataset = []
        i = 0 
        for new_example in choose_data:
            while train_dataset[i]['problem'] != new_example['prompt']:
                i+=1
            example = train_dataset[i]
            prompt, rejected = example['problem'], example['solution']
            new_dataset.append({
                'prompt': prompt,
                'chosen': new_example['generate'],
                'rejected': rejected,
            })
    data_dict = {
        "prompt": [item["prompt"] for item in new_dataset],
        "chosen": [item["chosen"] for item in new_dataset],
        "rejected": [item["rejected"] for item in new_dataset],
    }
    return Dataset.from_dict(data_dict)

special_tokens=['<Thought>', '</Thought>','<Output>', '</Output>']
MAX_LENGTH = 1024 * 8 # tokenizer.model_max_length


if __name__ == '__main__':
    lr, beta = 5e-6, 0.5
    model_from = 1
    model_name =set_global('./models/openo1_sft_lr3e-05_8b_no_chinese_pro/checkpoint-2638') 
    # "meta-llama/Llama-3.1-8B-Instruct"
    #   # "meta-llama/Llama-3.2-3B-Instruct"
    dataset_name = set_global('data/NuminaMath-CoT_O1_Qwq.jsonl')
    wandb_name = f"numinaMath_dpo_lr{lr}_beta{beta}_8B"

    train_dataset = load_data(dataset_name)

    model, tokenizer, special_tokenizer = init_model(model_from=model_from,  model_name=model_name, special_tokens=special_tokens)
    training_config = DPOConfig(
        output_dir=set_global(f"./minimind_dpo/{wandb_name}"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        save_total_limit=1,
        num_train_epochs=1,
        remove_unused_columns=False,
        report_to="wandb",
        # save_steps=10,
        save_strategy='epoch',
        beta=beta,
        logging_steps=1,
        learning_rate=lr,
        weight_decay=0.01,
        max_length=MAX_LENGTH,
        # warmup_ratio=0.01,
    )


    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_config,
        train_dataset=train_dataset,
        tokenizer=tokenizer
    )

    wandb.login() 
    wandb.init(project="MiniMind-Full-SFT", name=wandb_name)    
    dpo_trainer.train()