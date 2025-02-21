import __init__
from utils.utils import *
from utils.data_utils import save_all_results, read_saved_results, load_data
from utils.load_model import *

def init_train_model(model_name=''):
    model, tokenizer = init_model(model_name, eval=False, low_cpu_mem_usage=False)
    model.enable_input_require_grads()
    if tokenizer.pad_token is None:
        token = tokenizer.convert_ids_to_tokens(2)
        print(f"Token for ID 2: {token}")
        tokenizer.pad_token_id = 2 
    return model, tokenizer

def load_train_data(choose_data_name):
    if choose_data_name in ['Bespoke_dpo', 'Bespoke', 'Deepseek']:
        choose_data_name = dataset_names[choose_data_name]
        choose_data = load_data(choose_data_name, 'huggingface')
    else:
        choose_data_name = set_global(dataset_names[choose_data_name])
        choose_data = load_data(choose_data_name, 'json')
    choose_data = choose_data['train']
    return choose_data

dataset_names = {
    'NuminaMath': 'AI-MO/NuminaMath-CoT',
    'Bespoke':'bespokelabs/Bespoke-Stratos-17k',
    
    'Bespoke_dpo':f'VanWang/Bespoke_dpo_filter',
    'Bespoke_dpo_long':f'data/final/Bespoke_dpo_filter_len_long.jsonl',
    'Bespoke_dpo_short':f'data/final/Bespoke_dpo_filter_len_short.jsonl',
    'Bespoke_dpo_middle':'data/final/Bespoke_dpo_filter_len_middle.jsonl'
}
model_names = {
    'Deepseek-7b':'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',

    'Instruct-1.5b':'Qwen/Qwen2.5-1.5B-Instruct',
    'Instruct-3b':'Qwen/Qwen2.5-3B-Instruct',
    'Instruct-7b':'Qwen/Qwen2.5-7B-Instruct',
    'Instruct-3b':'Qwen/Qwen2.5-3B-Instruct',
    'Instruct-14b':'Qwen/Qwen2.5-14B-Instruct',
    'Instruct-32b':'Qwen/Qwen2.5-32B-Instruct',

    'Bespoke-32b': 'bespokelabs/Bespoke-Stratos-32B', 
    'Bespoke-7b': 'bespokelabs/Bespoke-Stratos-7B', 

    'OpenThinker-7B':'open-thoughts/OpenThinker-7B',
}