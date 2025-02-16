from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import wandb
from ..utils.utils import * 
from ..utils.load_model import init_model

def load_data(dataset_name, mode='json'):
    if mode == 'huggingface':
        dataset = load_dataset(dataset_name, cache_dir=cache_dir)
        train_dataset = dataset['train'].select(range(84271))
        train_dataset = train_dataset.rename_column("text", "review")
        train_dataset = train_dataset.rename_column("label", "sentiment")
    if mode == 'json':
        dataset = load_dataset('json', data_files=dataset_name)
        train_dataset = dataset['train']
    return train_dataset

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['prompt'])):
        text = f"{example['prompt'][i]} \n >>> Response:\n {example['response'][i]} <|end_of_text|>"
        output_texts.append(text)
    return output_texts



save_steps, lr = 1000, 3e-5
model_name = "meta-llama/Llama-3.1-8B" # 'meta-llama/Llama-3.2-3B'
dataset_name =set_global('data/final/OpenO1-SFT-Pro-Filter.jsonl') # 'AI-MO/NuminaMath-CoT' # 
wandb_name = f"openo1_sft_lr{lr}_8b_no_chinese_pro"
MAX_LENGTH = 1024 * 8 # tokenizer.model_max_length

model, tokenizer = init_model(model_name=model_name)
train_dataset = load_data(dataset_name, mode='json')


response_template_with_context =" >>> Response:\n "  # We added context here: "\n". This is enough for this tokenizer
response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]  # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]`
data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

Ssft_config = SFTConfig(
        output_dir=set_global(f"./models/sft_trainer/{wandb_name}"),
        max_seq_length=MAX_LENGTH,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        save_total_limit=1,
        num_train_epochs=1,
        learning_rate=lr,
        # lr_scheduler_type='constant',
        warmup_ratio=0.01,
        logging_steps=10,
        save_steps=save_steps,
        gradient_checkpointing=True,
        report_to="wandb"
    )
trainer = SFTTrainer(
    model,
    train_dataset=train_dataset,
    args=Ssft_config,
    formatting_func=formatting_prompts_func,
    data_collator=data_collator,
)

wandb.login() 
wandb.init(project="MiniMind-Full-SFT-Trainer", name=wandb_name)
trainer.train()