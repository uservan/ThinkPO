import os
import numpy as np
import random
import torch
from utils.settings import *


def set_global(path):
    return os.path.join(project_dir, path)

from huggingface_hub import login
login(token=hug_token)

api_keys = {
    'openai': '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'
}


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def Logger(content):
    print(content)