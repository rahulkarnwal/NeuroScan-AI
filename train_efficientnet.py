
import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(r"D:\BrainTumourDetection")
TRAIN_DIR = BASE_DIR / "braintumour_cleaned" / "Training"

CHECKPOINT_PATH = BASE_DIR / "efficientnet_best.weights.h5"
MODEL_PATH = BASE_DIR / "brain_tumor_efficientnetb0.h5"
CLASS_NAMES_PATH = BASE_DIR / "class_names_efficientnet.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 12
SEED = 42

tf.keras.utils.set_random_seed(SEED)

if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Training dataset not found: {TRAIN_DIR}"
    )

print("=" * 55)
print("BRAIN TUMOR CLASSIFICATION")
print("EfficientNetB0 Transfer Learning")
print("=" * 55)

# ==================================================
# 2. LOAD TRAINING AND VALIDATION DATA
# ==================================================

print("\nLoading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True
)

print("\nLoading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as file:
    json.dump(class_names, file, indent=4)

print("\nClass names:", class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ==================================================
# 3. LOAD PRETRAINED EFFICIENTNETB0
# ==================================================

print("\nLoading pretrained EfficientNetB0...")

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers during initial training.
base_model.trainable = False

# ==================================================
# 4. BUILD CLASSIFICATION MODEL
# ==================================================

inputs = layers.Input(shape=(224, 224, 3))

# EfficientNetB0 includes input rescaling.
# Dataset images remain in the 0-255 pixel range.
x = base_model(inputs, training=False)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.2)(x)

outputs = layers.Dense(
    num_classes,
    activation="softmax"
)(x)

model = models.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==================================================
# 5. CONFIGURE CALLBACKS
# ==================================================

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),

    ModelCheckpoint(
        filepath=str(CHECKPOINT_PATH),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
]

# ==================================================
# 6. TRAIN THE MODEL
# ==================================================

print("\nStarting EfficientNetB0 training...\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ==================================================
# 7. RESTORE BEST CHECKPOINT AND SAVE FULL MODEL
# ==================================================

print("\nRestoring the best saved checkpoint...")

model.load_weights(str(CHECKPOINT_PATH))

print("Saving complete model...")

# HDF5 format avoids the native .keras saving path
# that previously raised the 'options' error.
model.save(str(MODEL_PATH), save_format="h5")

# ==================================================
# 8. DISPLAY TRAINING RESULTS
# ==================================================

best_epoch = (
    history.history["val_loss"].index(
        min(history.history["val_loss"])
    ) + 1
)

print("\n" + "=" * 55)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 55)

print(
    f"Best validation accuracy: "
    f"{max(history.history['val_accuracy']) * 100:.2f}%"
)

print(
    f"Best validation loss: "
    f"{min(history.history['val_loss']):.4f}"
)

print(f"Epoch with lowest validation loss: {best_epoch}")

print(f"\nBest weights saved at:\n{CHECKPOINT_PATH}")
print(f"Complete model saved at:\n{MODEL_PATH}")
print(f"Class names saved at:\n{CLASS_NAMES_PATH}")

print("\nYour previous CNN models remain unchanged.")
