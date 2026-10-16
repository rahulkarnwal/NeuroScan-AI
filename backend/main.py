
from pathlib import Path
from datetime import datetime, timezone
import io
import json
import os
import sqlite3

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "brain_tumor_efficientnetb0.h5"
CLASS_NAMES_PATH = BASE_DIR / "class_names_efficientnet.json"
DATABASE_PATH = BACKEND_DIR / "history.db"

app = FastAPI(title="Brain Tumor Detection API")

FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:5173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

model = None
class_names = []


def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                probabilities TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


initialize_database()


@app.on_event("startup")
def load_model():
    global model, class_names

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    if model.output_shape[-1] != len(class_names):
        raise RuntimeError(
            "Model output count and class names do not match."
        )

    print("EfficientNetB0 model loaded successfully.")
    print("Classes:", class_names)
    print("Prediction history database:", DATABASE_PATH)


@app.get("/")
def home():
    return {
        "message": "Brain Tumor Detection API is running",
        "classes": class_names,
    }


@app.get("/health")
def health():
    return {"model_loaded": model is not None}



def make_gradcam_overlay(image_rgb, image_array, class_index):
    """Create a Grad-CAM overlay for the selected class using EfficientNetB0."""
    if model is None:
        raise RuntimeError("Model is not loaded.")

    # The trained classifier has EfficientNetB0 as its second (nested) layer.
    base_model = model.get_layer("efficientnetb0")
    conv_layer = base_model.get_layer("top_conv")
    feature_extractor = tf.keras.Model(
        inputs=base_model.input,
        outputs=[conv_layer.output, base_model.output],
    )

    tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_features, base_output = feature_extractor(tensor, training=False)
        tape.watch(conv_features)
        x = base_output
        # Apply the existing classifier head, preserving its trained weights.
        for layer in model.layers[2:]:
            x = layer(x, training=False) if isinstance(layer, tf.keras.layers.Dropout) else layer(x)
        class_score = x[:, class_index]

    gradients = tape.gradient(class_score, conv_features)
    if gradients is None:
        raise RuntimeError("Could not calculate Grad-CAM gradients for this model.")

    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    cam = tf.reduce_sum(weights * conv_features, axis=-1)[0]
    cam = tf.maximum(cam, 0)
    max_value = tf.reduce_max(cam)
    cam = tf.where(max_value > 0, cam / (max_value + tf.keras.backend.epsilon()), tf.zeros_like(cam))
    cam = tf.image.resize(cam[..., tf.newaxis], (224, 224)).numpy()[..., 0]

    # Build a simple JET-like colour map without requiring OpenCV or Matplotlib.
    value = np.clip(cam, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * value - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * value - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * value - 1.0), 0.0, 1.0)
    heatmap = (np.stack([red, green, blue], axis=-1) * 255).astype(np.uint8)

    original = np.asarray(image_rgb.resize((224, 224)), dtype=np.uint8)
    overlay = np.clip(original.astype(np.float32) * 0.55 + heatmap.astype(np.float32) * 0.45, 0, 255).astype(np.uint8)

    def png_data_uri(array):
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="PNG")
        import base64
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    return png_data_uri(overlay), png_data_uri(heatmap)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    if file.content_type not in (
        "image/jpeg",
        "image/png",
        "image/webp",
    ):
        raise HTTPException(
            status_code=400,
            detail="Upload a JPEG, PNG, or WebP image.",
        )

    try:
        contents = await file.read()

        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image must be smaller than 10 MB.",
            )

        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((224, 224))

        image_array = np.asarray(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        scores = model.predict(image_array, verbose=0)[0]
        index = int(np.argmax(scores))

        predicted_class = class_names[index]
        confidence = round(float(scores[index]) * 100, 2)

        probabilities = {
            name: round(float(score) * 100, 2)
            for name, score in zip(class_names, scores)
        }

        try:
            gradcam_overlay, gradcam_heatmap = make_gradcam_overlay(
                image, image_array, index
            )
        except Exception as exc:
            # Keep ordinary predictions available even if explanation generation fails.
            print(f"Grad-CAM generation failed: {exc}")
            gradcam_overlay, gradcam_heatmap = None, None

        created_at = datetime.now(timezone.utc).isoformat()

        # Save this prediction in SQLite.
        with get_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO prediction_history
                (filename, predicted_class, confidence,
                 probabilities, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    file.filename or "uploaded_image",
                    predicted_class,
                    confidence,
                    json.dumps(probabilities),
                    created_at,
                ),
            )

            prediction_id = cursor.lastrowid

        return {
            "id": prediction_id,
            "filename": file.filename or "uploaded_image",
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities,
            "gradcam_overlay": gradcam_overlay,
            "gradcam_heatmap": gradcam_heatmap,
            "gradcam_notice": (
                "Grad-CAM highlights image regions that influenced this model prediction; "
                "it is not a tumor segmentation or a medical diagnosis."
            ),
            "created_at": created_at,
            "notice": (
                "Educational prototype only; "
                "not a medical diagnosis."
            ),
        }

    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )
    finally:
        await file.close()


@app.get("/history")
def get_history():
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, filename, predicted_class,
                   confidence, probabilities, created_at
            FROM prediction_history
            ORDER BY id DESC
            """
        ).fetchall()

    history = []

    for row in rows:
        item = dict(row)
        item["probabilities"] = json.loads(item["probabilities"])
        history.append(item)

    return {"history": history}


@app.delete("/history")
def clear_history():
    with get_connection() as connection:
        connection.execute("DELETE FROM prediction_history")

    return {"message": "Prediction history cleared successfully."}
