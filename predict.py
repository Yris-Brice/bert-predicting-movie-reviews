import sys
from typing import List

from bert import run_classifier
import numpy as np

from defaults import LABEL_LIST, MAX_SEQ_LENGTH
from preprocessing import create_tokenizer_from_hub_module
from train import create_estimator

LABELS = ["Negative", "Positive"]

tokenizer = create_tokenizer_from_hub_module()
estimator = create_estimator()


def predict(sentences: List[str]):
    input_examples = [
        run_classifier.InputExample(guid="", text_a=sent, text_b=None, label=0)
        for sent in sentences
    ]  # here, "" is just a dummy label
    input_features = run_classifier.convert_examples_to_features(
        input_examples, LABEL_LIST, MAX_SEQ_LENGTH, tokenizer
    )
    predict_input_fn = run_classifier.input_fn_builder(
        features=input_features,
        seq_length=MAX_SEQ_LENGTH,
        is_training=False,
        drop_remainder=False,
    )
    predictions = estimator.predict(
        predict_input_fn, yield_single_examples=len(sentences) > 1
    )
    return [
        {
            "sentence": sent,
            "probabilities": pred["probabilities"],
            "label": LABELS[pred["labels"]],
        }
        for sent, pred in zip(sentences, predictions)
    ]


if __name__ == "__main__":
    sentences = sys.argv[1:]
    # sentences = ["Foo"] #, "Bar"]
    print("Predict sentences=%r" % sentences)
    print(predict(sentences))
