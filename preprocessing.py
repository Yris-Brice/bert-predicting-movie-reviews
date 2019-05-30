import pickle
from typing import Tuple

from bert import run_classifier, tokenization
from fastparquet import ParquetFile
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

from defaults import (
    BERT_MODEL_HUB,
    DATASET_NAMES,
    LABEL_LIST,
    MAX_SEQ_LENGTH,
    SAMPLE_SIZE,
    data_filename,
)


DATA_COLUMN = "sentence"
LABEL_COLUMN = "polarity"


def create_bert_input_example(row: pd.Series) -> run_classifier.InputExample:
    return run_classifier.InputExample(
        guid=None,  # Globally unique ID for bookkeeping, unused in this example
        text_a=row[DATA_COLUMN],
        text_b=None,
        label=row[LABEL_COLUMN],
    )


def create_tokenizer_from_hub_module() -> tokenization.FullTokenizer:
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run(
                [tokenization_info["vocab_file"], tokenization_info["do_lower_case"]]
            )
    return tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case
    )


def create_train_test_features(
    tokenizer: tokenization.FullTokenizer
) -> Tuple[run_classifier.InputFeatures, run_classifier.InputFeatures]:
    train_input_examples, test_input_examples = (
        ParquetFile(data_filename(dataset_name))
        .to_pandas()
        .sample(SAMPLE_SIZE)
        .apply(create_bert_input_example, axis=1)
        for dataset_name in DATASET_NAMES
    )

    train_features, test_features = (
        run_classifier.convert_examples_to_features(
            input_examples, LABEL_LIST, MAX_SEQ_LENGTH, tokenizer
        )
        for input_examples in (train_input_examples, test_input_examples)
    )

    return train_features, test_features


if __name__ == "__main__":
    tokenizer = create_tokenizer_from_hub_module()
    train_features, test_features = create_train_test_features(tokenizer)
    train_features_file, test_features_file = (
        open(data_filename(f"{mode}_features", serializer="pkl"), "wb")
        for mode in ["train", "test"]
    )
    pickle.dump(train_features, train_features_file)
    pickle.dump(test_features, test_features_file)
    print(
        "Stored train and test features files=", train_features_file, test_features_file
    )
