from typing import Union
import os

import numpy as np

BERT_MODEL_HUB: str = os.environ.get(
    "BERT_MODEL_HUB", "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
)
DATA_DIR = os.environ.get("DATA_DIR", "data")
DATASET_NAMES = ["train", "test"]
FORCE_DOWNLOAD = bool(os.environ.get("FORCE_DOWNLOAD", False))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 2e-5))
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", 128))
SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", 5_000))
SEED: Union[int, np.random.RandomState] = int(os.environ.get("SEED", 42))

LABEL_LIST = [0, 1]

# Compute train and warmup steps from batch size
# These hyperparameters are copied from this colab notebook
# (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
MODEL_DIR = os.environ.get("MODEL_DIR", "model")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
NUM_TRAIN_EPOCHS = float(os.environ.get("NUM_TRAIN_EPOCHS", 3.0))
# Warmup is a period of time where hte learning rate
# is small and gradually increases--usually helps training.
WARMUP_PROPORTION = float(os.environ.get("WARMUP_PROPORTION", 0.1))
# Model configs
SAVE_CHECKPOINTS_STEPS = int(os.environ.get("SAVE_CHECKPOINTS_STEPS", 500))
SAVE_SUMMARY_STEPS = int(os.environ.get("SAVE_SUMMARY_STEPS", 100))


np.random.seed(seed=SEED)


def data_filename(dataset_name: str, serializer: str = "parq") -> str:
    return os.path.join(DATA_DIR, f"{dataset_name}.{serializer}")
