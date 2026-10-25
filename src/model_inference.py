import io
import json
import os
from fastapi import FastAPI, File, HTTPException, UploadFile
import numpy as np
import onnxruntime as ort
from PIL import Image

app = FastAPI(
    title="E-Commerce Fashion ONNX Inference Engine",
    version="1.0.0",
    description="Production MLOps microservice serving MobileNetV3 ONNX model for 123 fashion categories.",
)

# 1. Dynamic Directory Resolution (Platform Agnostic)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

MODEL_PATH = os.path.join(PROJECT_DIR, "models", "fashion_cnn.onnx")
CLASSES_PATH = os.path.join(PROJECT_DIR, "data", "class_names.json")

# 2. Load Class Labels & ONNX Inference Session
try:
    if not os.path.exists(CLASSES_PATH):
        raise FileNotFoundError(
            f"Class mapping file not found at '{CLASSES_PATH}'."
        )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"ONNX model file not found at '{MODEL_PATH}'."
        )

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        CLASS_NAMES = json.load(f)

    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
except Exception as e:
    raise RuntimeError(f"Failed to load ONNX model or class mapping: {str(e)}")


def pad_to_square(image: Image.Image, background_color=(255, 255, 255)) -> Image.Image:
    """Pads image with white borders to make it square while preserving its original aspect ratio."""
    w, h = image.size
    max_dim = max(w, h)
    padded_img = Image.new("RGB", (max_dim, max_dim), background_color)
    padded_img.paste(image, ((max_dim - w) // 2, (max_dim - h) // 2))
    return padded_img


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess raw HTTP image payload into RGB 224x224 tensor with aspect-ratio preservation."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # 1. Pad to square to prevent stretching tall images (e.g., pants, dresses)
    image = pad_to_square(image)
    
    # 2. Resize to model input dimensions
    image = image.resize((224, 224))

    # Standard ImageNet Normalization
    img_np = np.array(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std

    # Reshape array dimensions: [H, W, C] -> [C, H, W] -> [1, C, H, W]
    img_np = np.transpose(img_np, (2, 0, 1))
    img_np = np.expand_dims(img_np, axis=0)
    return img_np.astype(np.float32)


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "engine": "ONNX Runtime",
        "model": "fashion_cnn.onnx",
        "supported_categories": len(CLASS_NAMES),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/jpg",
    ]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format. Please upload a JPG, JPEG, PNG, or WEBP image.",
        )

    try:
        image_bytes = await file.read()
        tensor = preprocess_image(image_bytes)

        outputs = session.run(None, {input_name: tensor})[0]

        # Softmax Probability Calculation
        exp_outputs = np.exp(outputs - np.max(outputs))
        probs = (exp_outputs / exp_outputs.sum(axis=1, keepdims=True))[0]

        pred_class_idx = int(np.argmax(probs))
        confidence_score = float(probs[pred_class_idx])

        # Extract Top-5 Ranked Predictions
        top5_indices = np.argsort(probs)[-5:][::-1]
        top5_predictions = [
            {
                "category": CLASS_NAMES[idx],
                "confidence": round(float(probs[idx]), 4),
            }
            for idx in top5_indices
        ]

        return {
            "prediction": CLASS_NAMES[pred_class_idx],
            "confidence": round(confidence_score, 4),
            "top_5": top5_predictions,
            "status": "success",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
