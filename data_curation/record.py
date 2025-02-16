import __init__
import os
from utils.data_utils import *
from utils.eval.eval_utils import find_box


def count_total_occurrences(text, words):
    return sum(text.lower().count(word) for word in words)
word_list = ["wait", "hmm"]

folder_path = "/home/wxy320/ondemand/program/SkyThought/skythought/tools/results/steps"  
subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

results = dict()
for step in subdirs:
    step = os.path.join(folder_path, step)
    json_lens = dict()
    json_files = [f for f in os.listdir(step) if f.endswith(".json") and os.path.isfile(os.path.join(step, f))]
    for file_path in json_files:
        with open(os.path.join(step,file_path), "r", encoding="utf-8") as f:
            data = json.load(f)
        all_prompts_len, num, counts = 0, 0 ,0
        for prompt, d in data.items():
            if '<|end_of_solution|>' in d['responses']['0.7']['content']: 
                all_prompts_len = all_prompts_len + d['token_usages']['0.7']['completion_tokens']
                num = num +1
            if d['responses']['0.7']['correctness']: 
                counts =counts+count_total_occurrences(d['responses']['0.7']['content'], word_list)
        json_lens[file_path] = {'lengths':all_prompts_len / num, "wait":counts}
    results[step.split('/')[-1]] = json_lens


with open('results.json', "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4) 