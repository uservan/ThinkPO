import __init__
from utils.data_utils import save_all_results, read_saved_results, load_dataset, save_results
from utils.utils import cache_dir, set_global
from utils.load_model import *
from utils.data_utils import *
from utils.eval.eval_utils import check_math_correctness
import random
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt


def get_middle_1000(lst):
    n = len(lst)
    if n <= 1000:
        return lst  
    start = (n - 1000) // 2 
    return lst[start:start + 1000]

def generate_is_ok(predict, refrence):
    if not check_math_correctness(refrence, predict): return False
    return True

save_file = set_global(f'./data/final/Bespoke_dpo_filter_short.jsonl')
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

plt.figure(figsize=(8, 6))
plt.hist(lengths, bins=1000, edgecolor='black', alpha=0.7)
plt.xlabel("Token Length")
plt.ylabel("Frequency")
plt.title("Distribution of Text Token Lengths")
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig("token_length_distribution.png")
plt.show()
