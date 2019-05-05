from typing import List, Union
import os

import numpy as np

DATA_DIR: str = os.environ.get("DATA_DIR", "data")
DATASET_NAMES: List[str] = ["train", "test"]
FORCE_DOWNLOAD: bool = bool(os.environ.get("FORCE_DOWNLOAD", False))
SEED: Union[int, np.random.RandomState] = int(os.environ.get("SEED", 42))

np.random.seed(seed=SEED)


def data_filename(dataset_name: str) -> str:
    return os.path.join(DATA_DIR, f"{dataset_name}.parq")
