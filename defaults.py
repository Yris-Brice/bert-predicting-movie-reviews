from typing import List, Union
import os

import numpy as np

BERT_MODEL_HUB: str = os.environ.get(
    "BERT_MODEL_HUB", "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
)
DATA_DIR: str = os.environ.get("DATA_DIR", "data")
DATASET_NAMES: List[str] = ["train", "test"]
FORCE_DOWNLOAD: bool = bool(os.environ.get("FORCE_DOWNLOAD", False))
MAX_SEQ_LENGTH: int = int(os.environ.get("MAX_SEQ_LENGTH", 128))
SAMPLE_SIZE: int = 5_000
SEED: Union[int, np.random.RandomState] = int(os.environ.get("SEED", 42))


np.random.seed(seed=SEED)


def data_filename(dataset_name: str) -> str:
    return os.path.join(DATA_DIR, f"{dataset_name}.parq")
