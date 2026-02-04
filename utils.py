import torch
import numpy as np
import random
import os


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def ensure_cache_dir(cache_dir_path):
    if not os.path.exists(cache_dir_path):
        os.makedirs(cache_dir_path)
        os.makedirs(cache_dir_path + '/train_cache')
        os.makedirs(cache_dir_path + '/results')

