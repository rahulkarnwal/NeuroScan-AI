
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)

BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "braintumour_cleaned" / "Testing"
MODEL_PATH = BASE_DIR / "brain_tumor_efficientnetb0.h5"
CLASS_PATH = BASE_DIR / "class_names_efficientnet.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8

print("=" * 55)
print("EFFICIENTNETB0 - TEST EVALUATION")
print("=" * 55)

# Check required files and folders
for path in [TEST_DIR, MODEL_PATH, CLASS_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Required path not found: {path}")

# Load the saved class names
with open(CLASS_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

print("\nClass names:", class_names)

# Load test data without shuffling
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int"
)

# Verify class order
if test_ds.class_names != class_names:
    raise ValueError(
        f"Class order mismatch!\n"
        f"Model classes: {class_names}\n"
        f"Test classes: {test_ds.class_names}"
    )

# Load the trained model
print("\nLoading trained EfficientNetB0...")
model = tf.keras.models.load_model(str(MODEL_PATH))

# Collect true labels in dataset order
y_true = np.concatenate([
    labels.numpy() for _, labels in test_ds
])

# Predict class probabilities
print("\nPredicting test images...")
probabilities = model.predict(test_ds, verbose=1)
y_pred = np.argmax(probabilities, axis=1)

# Overall accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"\nTest accuracy: {accuracy * 100:.2f}%")

# Detailed metrics
print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)

# Confusion matrix
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(class_names)))
)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)
disp.plot(cmap="Blues", xticks_rotation=30, values_format="d")
plt.title("EfficientNetB0 - Cleaned Test Set")
plt.tight_layout()

output_path = BASE_DIR / "confusion_matrix_efficientnet.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"\nConfusion matrix saved to: {output_path}")
print("\nEvaluation completed successfully.")
