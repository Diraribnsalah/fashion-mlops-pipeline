import io
import json
import os
import sys
import numpy as np
import onnxruntime as ort
import pytest
from fastapi.testclient import TestClient
from PIL import Image
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
# Import elements from project source
from src.model_inference import app, pad_to_square, preprocess_image

# Dynamic Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fashion_cnn.onnx")
CLASSES_PATH = os.path.join(BASE_DIR, "data", "class_names.json")

client = TestClient(app)


# Test 1: Verify Dataset Artifacts
def test_class_names_exist_and_valid():
    assert os.path.exists(
        CLASSES_PATH
    ), f"Missing class mapping file at {CLASSES_PATH}"

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    assert isinstance(class_names, list)
    assert len(class_names) > 0, "Class list should not be empty."


# Test 2: Preprocessing Logic (pad_to_square)
def test_pad_to_square():
    # Create a non-square test image (Tall image: 100x300)
    tall_img = Image.new("RGB", (100, 300), color="red")
    padded_img = pad_to_square(tall_img)

    # Output should be square (300x300)
    assert padded_img.size == (300, 300)


# Test 3: ONNX Model File & Inference Execution
def test_onnx_model_inference():
    assert os.path.exists(
        MODEL_PATH
    ), f"Missing ONNX model weight file at {MODEL_PATH}"

    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name

    # Generate synthetic RGB input tensor (1, 3, 224, 224)
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    outputs = session.run(None, {input_name: dummy_input})[0]

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    assert outputs.shape == (
        1,
        len(class_names),
    ), "Model output shape must match class count."


# Test 4: FastAPI Health Check Endpoint
def test_fastapi_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["engine"] == "ONNX Runtime"


# Test 5: FastAPI /predict Endpoint with Synthetic Image Upload
def test_fastapi_predict_endpoint():
    # Create synthetic JPEG in-memory
    img = Image.new("RGB", (224, 224), color="blue")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_bytes = img_byte_arr.getvalue()

    response = client.post(
        "/predict", files={"file": ("test_shirt.jpg", img_bytes, "image/jpeg")}
    )

    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "success"
    assert "prediction" in json_data
    assert "confidence" in json_data
    assert len(json_data["top_5"]) == 5
