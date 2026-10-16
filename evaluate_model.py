
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

BASE_DIR = Path(r"D:\BrainTumourDetection")
TEST_DIR = BASE_DIR / "braintumour_cleaned" / "Testing"
MODEL_PATH = BASE_DIR / "brain_tumor_efficientnetb0.h5"
CLASS_NAMES_PATH = BASE_DIR / "class_names_efficientnet.json"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8

def main():
    for path in (TEST_DIR, MODEL_PATH, CLASS_NAMES_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
        class_names = json.load(f)

    print("Model class order:", class_names)

    # Match the class order used by the original training script.
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    if test_ds.class_names != class_names:
        raise ValueError(
            "Class order mismatch!\n"
            f"Model: {class_names}\n"
            f"Test folders: {test_ds.class_names}"
        )

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    y_true = np.concatenate([labels.numpy() for _, labels in test_ds])

    # Preserve 0–255 pixel values, as in the training script.
    probabilities = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)

    print("\n" + "=" * 55)
    print("HELD-OUT TEST SET EVALUATION")
    print("=" * 55)
    print("Model:", MODEL_PATH.name)
    print("Test images:", len(y_true))
    print(f"Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%")
    print(f"Macro F1: {f1_score(y_true, y_pred, average='macro') * 100:.2f}%")
    print(f"Weighted F1: {f1_score(y_true, y_pred, average='weighted') * 100:.2f}%")

    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_true, y_pred, labels=range(len(class_names))))

    print("\nPer-class classification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=range(len(class_names)),
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

if __name__ == "__main__":
    main()
