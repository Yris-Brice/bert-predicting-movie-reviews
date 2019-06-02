import json_tricks as json
import pickle

import bert
import click

import defaults
from train import create_estimator


def evaluate():
    test_features = pickle.load(
        open(defaults.data_filename("test_features", serializer="pkl"), "rb")
    )

    test_input_fn = bert.run_classifier.input_fn_builder(
        features=test_features,
        seq_length=defaults.MAX_SEQ_LENGTH,
        is_training=False,
        drop_remainder=False,
    )

    estimator = create_estimator()
    return estimator.evaluate(input_fn=test_input_fn, steps=None)


@click.command()
@click.option("-o", "--output-file", type=click.Path(writable=False))
def cli(output_file):
    evaluation = evaluate()
    if output_file:
        with open(output_file, "w") as f:
            json.dump(evaluation, f, indent=4, sort_keys=True)

    print(evaluation)


if __name__ == "__main__":
    cli()
