# Copy+pasted from
# https://colab.research.google.com/github/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb

from typing import Dict, List, Tuple
import os
import re

import fastparquet
import pandas as pd
import tensorflow as tf

from exceptions import FilePathFormatException
from defaults import DATA_DIR, DATASET_NAMES, FORCE_DOWNLOAD, data_filename


# Load all files from a directory in a DataFrame.
def load_directory_data(directory: str) -> pd.DataFrame:
    data: Dict[str, List[str]] = {}
    data["sentence"] = []
    data["sentiment"] = []
    for file_path in sorted(os.listdir(directory)):
        with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
            data["sentence"].append(f.read())
            m = re.match("\d+_(\d+)\.txt", file_path)
            if m:
                data["sentiment"].append(m.group(1))
            else:
                msg = f"could not extract sentiment from file_path='{file_path}"
                raise FilePathFormatException(msg)
    return pd.DataFrame.from_dict(data)


# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory: str) -> pd.DataFrame:
    pos_df = load_directory_data(os.path.join(directory, "pos"))
    neg_df = load_directory_data(os.path.join(directory, "neg"))
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)


# Download and process the dataset files.
def download_and_load_datasets(
    force_download: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dataset = tf.keras.utils.get_file(
        fname="aclImdb.tar.gz",
        origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
        extract=True,
    )

    train_df, test_df = (
        load_dataset(os.path.join(os.path.dirname(dataset), "aclImdb", subdir))
        for subdir in DATASET_NAMES
    )

    return train_df, test_df


if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    dataframes = download_and_load_datasets(force_download=FORCE_DOWNLOAD)

    for df, dataset_name in zip(dataframes, DATASET_NAMES):
        fastparquet.write(data_filename(dataset_name), df)
