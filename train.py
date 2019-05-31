import os
import pickle

import bert
import pendulum
import tensorflow as tf

import defaults
from model import model_fn_builder


def create_estimator(
    model_dir: str = defaults.MODEL_DIR,
    batch_size: int = defaults.BATCH_SIZE,
    save_summary_steps: int = defaults.SAVE_SUMMARY_STEPS,
    save_checkpoints_steps: int = defaults.SAVE_CHECKPOINTS_STEPS,
    learning_rate: float = defaults.LEARNING_RATE,
    num_train_steps: int = 0,
    warmup_proportion: float = defaults.WARMUP_PROPORTION,
) -> tf.estimator.Estimator:
    # Specify output directory and number of checkpoint steps to save
    run_config = tf.estimator.RunConfig(
        model_dir=model_dir,
        save_summary_steps=save_summary_steps,
        save_checkpoints_steps=save_checkpoints_steps,
    )

    num_warmup_steps = int(num_train_steps * warmup_proportion)
    model_fn = model_fn_builder(
        num_labels=len(defaults.LABEL_LIST),
        learning_rate=learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
    )

    return tf.estimator.Estimator(
        model_fn=model_fn, config=run_config, params={"batch_size": batch_size}
    )


def train(
    batch_size: int = defaults.BATCH_SIZE,
    num_train_epochs: float = defaults.NUM_TRAIN_EPOCHS,
):
    train_features = pickle.load(
        open(defaults.data_filename(f"train_features", serializer="pkl"), "rb")
    )

    # Compute # train and warmup steps from batch size
    num_train_steps = int(len(train_features) / batch_size * num_train_epochs)

    estimator = create_estimator(num_train_steps=num_train_steps)

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
