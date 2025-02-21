import __init__
from utils.utils import *
from utils.data_utils import save_all_results, read_saved_results
from utils.load_model import *
from utils.data_utils import *
from utils.eval.eval_utils import check_math_correctness
from tqdm import tqdm


def get_middle_1000(lst):
    n = len(lst)
    if n <= 1000:
        return lst  
    start = (n - 1000) // 2 
    return lst[start:start + 1000]

def generate_is_ok(predict, refrence):
    if not check_math_correctness(refrence, predict): return False
    return True

save_file = set_global(f'./data/final/Bespoke_dpo.jsonl')
all_data = read_saved_results(save_file)
tokenizer = load_tokenizer("NovaSky-AI/Sky-T1-32B-Preview")

lengths, save_data = [], []
for d in tqdm(all_data):
    chosen, rejected = d['chosen'][2]['content'], d['rejected'][2]['content']
    chosen_len, rejected_len = len(tokenizer.encode(chosen, add_special_tokens=True)), len(tokenizer.encode(rejected, add_special_tokens=True))
    if rejected_len < 8*1024:
        d['dealta'] = rejected_len - chosen_len
        chosen, rejected = d['chosen'], d['rejected']
        d['chosen'], d['rejected'] = rejected, chosen
        save_data.append(d)
        lengths.append(d['dealta'])

all_data = sorted(save_data, key=lambda x: x["dealta"])

middle_1000 = get_middle_1000(all_data)
save_all_results(set_global(f'./data/final/Bespoke_dpo_filter_len_middle.jsonl'), middle_1000)
save_all_results(set_global(f'./data/final/Bespoke_dpo_filter_len_short.jsonl'), all_data[:1000])
save_all_results(set_global(f'./data/final/Bespoke_dpo_filter_len_long.jsonl'), all_data[-1000:])
