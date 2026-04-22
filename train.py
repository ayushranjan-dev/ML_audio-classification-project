
import os
import argparse
import tensorflow as tf
from data_loader import build_tf_dataset
from model import build_cnn_model
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_DIR, MODEL_NAME
from keras.callbacks import ModelCheckpoint, EarlyStopping

def main(train_dir, val_dir):
    train_ds, classes = build_tf_dataset(train_dir, batch_size=BATCH_SIZE, shuffle=True, augment=True)
    val_ds, _ = build_tf_dataset(val_dir, batch_size=BATCH_SIZE, shuffle=False, augment=False)

   
    for x_batch, y_batch in train_ds.take(1):
        input_shape = x_batch.shape[1:]  # (n_mels, t, 1)

    num_classes = len(classes)
    model = build_cnn_model(input_shape, num_classes if num_classes > 2 else 2)
    model.summary()

    # compile
    if num_classes == 2:
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        metrics = ['accuracy']
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        metrics = ['accuracy']

    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                  loss=loss,
                  metrics=metrics)

    os.makedirs(MODEL_DIR, exist_ok=True)
    ckpt_path = os.path.join(MODEL_DIR, MODEL_NAME)
    callbacks = [
        ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print(f"Model saved to {ckpt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="data/train", help="training dataset folder")
    parser.add_argument("--val_dir", default="data/val", help="validation dataset folder")
    args = parser.parse_args()
    main(args.train_dir, args.val_dir)