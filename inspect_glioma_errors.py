
import csv
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "braintumour_cleaned" / "Testing"
MODEL_PATH = BASE_DIR / "brain_tumor_efficientnetb0.h5"
CLASS_NAMES_PATH = BASE_DIR / "class_names_efficientnet.json"

OUTPUT_DIR = BASE_DIR / "glioma_error_analysis"
OVERLAY_DIR = OUTPUT_DIR / "gradcam_overlays"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

OUTPUT_DIR.mkdir(exist_ok=True)
OVERLAY_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# 2. LOAD EXISTING MODEL AND CLASS ORDER
# --------------------------------------------------
print("Loading existing EfficientNetB0 model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

print("Classes:", class_names)

if not TEST_DIR.exists():
    raise FileNotFoundError(f"Test folder not found: {TEST_DIR}")

if "glioma" not in class_names:
    raise ValueError("glioma is missing from the class-name file.")

glioma_index = class_names.index("glioma")

# --------------------------------------------------
# 3. LOAD TEST IMAGES IN A STABLE ORDER
# --------------------------------------------------
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    class_names=class_names,
    shuffle=False,
)

# Keep the file order aligned with predictions and labels.
file_paths = list(test_ds.file_paths)

all_labels = []
all_images = []

for images, labels in test_ds:
    all_images.append(images.numpy())
    all_labels.append(labels.numpy())

X = np.concatenate(all_images, axis=0).astype(np.float32)
y_true = np.concatenate(all_labels, axis=0)

print(f"Loaded {len(file_paths)} test images.")
print("Predicting without retraining...")

probabilities = model.predict(X, batch_size=BATCH_SIZE, verbose=1)
y_pred = np.argmax(probabilities, axis=1)

# Actual glioma images that the model predicted incorrectly.
error_indices = [
    i for i in range(len(y_true))
    if y_true[i] == glioma_index and y_pred[i] != glioma_index
]

print(f"\nMisclassified glioma images: {len(error_indices)}")

# --------------------------------------------------
# 4. FIND THE LAST CONVOLUTIONAL LAYER
# --------------------------------------------------
# In this model, EfficientNetB0 is nested inside the
# outer classification model.
base_model = next(
    (layer for layer in model.layers
     if isinstance(layer, tf.keras.Model)
     and "efficientnet" in layer.name.lower()),
    None,
)

if base_model is None:
    raise RuntimeError(
        "Could not find the nested EfficientNet model. "
        "Inspect model.summary() before adapting Grad-CAM."
    )

last_conv = base_model.get_layer("top_conv")

# This model returns the final convolutional feature maps
# and the nested EfficientNet output.
conv_model = tf.keras.Model(
    inputs=base_model.input,
    outputs=[last_conv.output, base_model.output],
)

# --------------------------------------------------
# 5. GRAD-CAM FOR THE MODEL'S PREDICTED CLASS
# --------------------------------------------------
def make_gradcam(image_batch, predicted_class):
    """
    image_batch: one image, shape (1, 224, 224, 3),
    with pixel values in the original 0-255 range.
    """
    with tf.GradientTape() as tape:
        conv_outputs, base_outputs = conv_model(
            image_batch, training=False
        )
        tape.watch(conv_outputs)

        # Pass EfficientNet features through the existing
        # trained classification head.
        x = base_outputs
        found_base = False

        for layer in model.layers:
            if layer is base_model:
                found_base = True
                continue
            if found_base:
                x = layer(x, training=False)

        class_score = x[:, predicted_class]

    grads = tape.gradient(class_score, conv_outputs)

    if grads is None:
        raise RuntimeError(
            "Gradients were not computed. "
            "Check the loaded model architecture."
        )

    # Average gradients over spatial dimensions.
    weights = tf.reduce_mean(grads, axis=(1, 2))

    # Weighted combination of convolutional feature maps.
    heatmap = tf.reduce_sum(
        conv_outputs * weights[:, tf.newaxis, tf.newaxis, :],
        axis=-1,
    )

    heatmap = tf.nn.relu(heatmap)[0]
    max_value = tf.reduce_max(heatmap)
    heatmap = heatmap / (max_value + tf.keras.backend.epsilon())

    return heatmap.numpy()


# --------------------------------------------------
# 6. SAVE PROBABILITIES, HEATMAPS, AND OVERLAYS
# --------------------------------------------------
csv_path = OUTPUT_DIR / "misclassified_glioma_probabilities.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow([
        "filename",
        "actual_class",
        "predicted_class",
        "predicted_probability",
        "glioma_probability",
        "meningioma_probability",
        "notumor_probability",
        "pituitary_probability",
    ])

    for number, idx in enumerate(error_indices, start=1):
        path = Path(file_paths[idx])
        actual_class = class_names[int(y_true[idx])]
        predicted_class = class_names[int(y_pred[idx])]
        scores = probabilities[idx]

        writer.writerow([
            str(path),
            actual_class,
            predicted_class,
            f"{scores[y_pred[idx]]:.6f}",
            *[f"{score:.6f}" for score in scores],
        ])

        # Generate Grad-CAM for the class the model predicted.
        image_batch = X[idx:idx + 1]
        heatmap = make_gradcam(
            image_batch,
            int(y_pred[idx]),
        )

        # Read original image for a visual overlay.
        with Image.open(path) as im:
            original = im.convert("RGB").resize(
                (224, 224), Image.Resampling.BILINEAR
            )
        original_array = np.asarray(original)

        overlay_path = OVERLAY_DIR / (
            f"{number:03d}_{path.stem}_as_{predicted_class}.png"
        )

        # Save original, Grad-CAM, and overlay side by side.
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(original_array)
        axes[0].set_title("Original MRI")

        axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
        axes[1].set_title("Grad-CAM heatmap")

        axes[2].imshow(original_array)
        axes[2].imshow(
            heatmap,
            cmap="jet",
            alpha=0.4,
            vmin=0,
            vmax=1,
            extent=(0, 224, 224, 0),
        )
        axes[2].set_title("Overlay")

        for ax in axes:
            ax.axis("off")

        fig.suptitle(
            f"Actual: {actual_class} | Predicted: {predicted_class}\n"
            f"Predicted probability: {scores[y_pred[idx]]:.2%}",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(overlay_path, dpi=130, bbox_inches="tight")
        plt.close(fig)

        if number % 10 == 0:
            print(f"Processed {number}/{len(error_indices)} errors")

print("\nAnalysis complete.")
print(f"CSV report: {csv_path}")
print(f"Grad-CAM images: {OVERLAY_DIR}")
print("No model weights were changed.")
