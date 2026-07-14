# model/train_model.py
"""
Train a small CNN on the Quick, Draw! .npy bitmaps or synthetic data.

Outputs:
- model/doodle_cnn.h5
- model/metrics/confusion_matrix.png
- model/metrics/training_curves.png
- model/metrics/metrics.json
"""
import os
from pathlib import Path
import numpy as np
import json
import matplotlib.pyplot as plt
import itertools

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
METRICS_DIR = MODEL_DIR / "metrics"
CATEGORIES = ["cat", "house", "tree", "car", "fish", "star", "umbrella", "banana", "bicycle", "clock"]

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# configurable
CAP_PER_CLASS = 5000  # cap for speed
TEST_SIZE = 0.2
RANDOM_STATE = 42
IMG_SIZE = 28
BATCH_SIZE = 64
EPOCHS = 12

def load_all_data(cap_per_class=CAP_PER_CLASS):
    X_list = []
    y_list = []
    for idx, cat in enumerate(CATEGORIES):
        path = DATA_DIR / f"{cat}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing data file: {path} - run data_prep.py first or download real files.")
        arr = np.load(path)
        if arr.ndim == 3:
            pass
        elif arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        else:
            raise ValueError("Unexpected array shape for " + cat)
        if cap_per_class:
            arr = arr[:cap_per_class]
        X_list.append(arr)
        y_list.append(np.full(len(arr), idx, dtype=np.int32))
        print(f"Loaded {len(arr)} samples for {cat}")
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y

def preprocess(X):
    X = X.astype("float32") / 255.0
    X = X.reshape((-1, IMG_SIZE, IMG_SIZE, 1))
    return X

def build_model(input_shape=(IMG_SIZE,IMG_SIZE,1), n_classes=len(CATEGORIES)):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
        MaxPool2D(),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPool2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(n_classes, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_confusion_matrix(cm, classes, outpath):
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(outpath)
    plt.close()

def plot_training(history, outpath):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='train_loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.title("Loss")
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='train_acc')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.title("Accuracy")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def main():
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.utils import to_categorical
    from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

    X_raw, y_raw = load_all_data()
    X = preprocess(X_raw)
    y = y_raw.copy()

    # stratified split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    y_train_cat = to_categorical(y_train, num_classes=len(CATEGORIES))
    y_test_cat = to_categorical(y_test, num_classes=len(CATEGORIES))

    model = build_model(input_shape=X.shape[1:])
    print(model.summary())

    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    ckpt = ModelCheckpoint(str(MODEL_DIR / "doodle_cnn.h5"), save_best_only=True, monitor='val_accuracy', mode='max')
    es = EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True)
    history = model.fit(X_train, y_train_cat, validation_split=0.1, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[ckpt, es])

    # save training curves
    plot_training(history, METRICS_DIR / "training_curves.png")

    # evaluate
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report = classification_report(y_test, y_pred, target_names=CATEGORIES, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, CATEGORIES, METRICS_DIR / "confusion_matrix.png")

    # compute precision/recall/f1 (macro)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    metrics = {
        "accuracy": float(np.mean(y_pred == y_test)),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f1),
        "classification_report": report
    }
    with open(METRICS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Metrics:")
    print(json.dumps(metrics, indent=2))
    print("Model and metrics saved to", MODEL_DIR)

if __name__ == "__main__":
    main()
