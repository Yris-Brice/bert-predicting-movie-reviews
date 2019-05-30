import os
import pickle

import bert
import pendulum
import tensorflow as tf

import defaults
from model import model_fn_builder


def train(
    model_dir: str = defaults.MODEL_DIR,
    data_dir: str = defaults.DATA_DIR,
    batch_size: int = defaults.BATCH_SIZE,
    num_train_epochs: float = defaults.NUM_TRAIN_EPOCHS,
    save_summary_steps: int = defaults.SAVE_SUMMARY_STEPS,
    save_checkpoints_steps: int = defaults.SAVE_CHECKPOINTS_STEPS,
    warmup_proportion: float = defaults.WARMUP_PROPORTION,
    learning_rate: float = defaults.LEARNING_RATE,
):
    features_files = (
        open(defaults.data_filename(f"{mode}_features", serializer="pkl"), "rb")
        for mode in ["train", "test"]
    )
    train_features, test_features = (pickle.load(f) for f in features_files)

    # Compute # train and warmup steps from batch size
    num_train_steps = int(len(train_features) / batch_size * num_train_epochs)
    num_warmup_steps = int(num_train_steps * warmup_proportion)

    # Specify output directory and number of checkpoint steps to save
    run_config = tf.estimator.RunConfig(
        model_dir=model_dir,
        save_summary_steps=save_summary_steps,
        save_checkpoints_steps=save_checkpoints_steps,
    )

    model_fn = model_fn_builder(
        num_labels=len(defaults.LABEL_LIST),
        learning_rate=learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn, config=run_config, params={"batch_size": batch_size}
    )

    # Create an input function for training. drop_remainder = True for using TPUs.
    train_input_fn = bert.run_classifier.input_fn_builder(
        features=train_features,
        seq_length=defaults.MAX_SEQ_LENGTH,
        is_training=True,
        drop_remainder=False,
    )

    print(f"Beginning Training!")
    current_time = pendulum.now()
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    print("Training took time ", pendulum.now() - current_time)


if __name__ == "__main__":
    if not os.path.exists(defaults.MODEL_DIR):
        os.makedirs(defaults.MODEL_DIR)

    train()
