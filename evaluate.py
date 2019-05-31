import pickle

import bert

import defaults
from train import create_estimator


def evaluate(model_dir: str = defaults.MODEL_DIR):
    test_features = pickle.load(
        open(defaults.data_filename("test_features", serializer="pkl"), "rb")
    )

    test_input_fn = bert.run_classifier.input_fn_builder(
        features=test_features,
        seq_length=defaults.MAX_SEQ_LENGTH,
        is_training=False,
        drop_remainder=False,
    )

    estimator = create_estimator(num_train_steps=int(5_000 / 16 * 3.0))
    print(estimator.evaluate(input_fn=test_input_fn, steps=None))


if __name__ == "__main__":
    evaluate()
