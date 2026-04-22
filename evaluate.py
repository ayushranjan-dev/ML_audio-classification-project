import numpy as np
import argparse
import tensorflow as tf
from data_loader import build_tf_dataset
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from config import MODEL_DIR, MODEL_NAME

def main(val_dir, model_path=None):
    ds, classes = build_tf_dataset(val_dir, batch_size=32, shuffle=False)
    if model_path is None:
        model_path = f"{MODEL_DIR}/{MODEL_NAME}"
    model = load_model(model_path, compile=True)

    y_true = []
    y_pred = []
    for X, y in ds:
        preds = model.predict(X)
        if preds.shape[-1] == 1:
            pred_labels = (preds.ravel() > 0.5).astype(int)
        else:
            pred_labels = np.argmax(preds, axis=1)
        y_true.extend(y.numpy().tolist())
        y_pred.extend(pred_labels.tolist())

    print("Classes:", classes)
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=classes))
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_dir", default="data/val", help="validation dataset folder")
    parser.add_argument("--model_path", default=None, help="path to saved model (.h5)")
    args = parser.parse_args()
    main(args.val_dir, args.model_path)