import __init__
from utils.data_utils import save_all_results, read_saved_results, load_dataset
from utils.utils import cache_dir, set_global
from utils.load_model import *
from utils.eval.eval_utils import check_math_correctness
from qwq_generate import qwq_generate
import random
from tqdm import tqdm

def generate_is_ok(predict, refrence):
    if not check_math_correctness(refrence, predict): return False
    return True

MAX_LENGTH = 1024 * 4
model_name = "Qwen/QwQ-32B-Preview"
each_run_size = [5000]
data_num = sum(each_run_size)
save_file = set_global(f'./data/generate/NuminaMath-CoT_O1_Qwq_olympiads.jsonl')

save_data = []
all_data = read_saved_results(save_file)
for data in tqdm(all_data):
    prompt = data['prompt']
    chosen = data['chosen']
    rejected = data['rejected']
    if generate_is_ok(chosen, rejected): 
        save_data.append(data) 
if len(save_data) < data_num:
    save_sets = set([d['prompt'] for d in save_data])
    dataset = load_dataset('AI-MO/NuminaMath-CoT', cache_dir=cache_dir)['train'].filter(lambda x: x['source'] == 'olympiads' and x['problem'] not in save_sets)
    dataset = dataset.select(random.sample(range(len(dataset)), data_num - len(save_data)))
    model, tokenizer = init_model(model_name, eval=True)
    for data in dataset:
        prompt = data['problem']
        for i in range(5):
            chosen_ans = qwq_generate(model, tokenizer,max_length=MAX_LENGTH)
            if generate_is_ok(chosen_ans, data['solution']): 
                save_data.append({
                    'prompt': prompt,
                    'chosen': chosen_ans,
                    'rejected': data['solution']
                })
                break
save_all_results(save_file, save_data)


cumsum = 0
for i, num in enumerate(each_run_size):
    save_file = set_global(f'./data/final/NuminaMath-CoT_O1_Qwq{i+1}.jsonl')
    save_all_results(save_file, save_data[cumsum: cumsum+num])
    cumsum = cumsum+num