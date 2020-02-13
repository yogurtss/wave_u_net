import numpy as np

def cfg():
    model_cfg = {'musdb_path': 'musdb18',
                 'batch_size': 16,
                 'lr': 1e-4,
                 'epochs': 2000,
                 'cache_size': 4000
                 'num_snippets_per_track': 100,
                 'excepted size': 22050

    }