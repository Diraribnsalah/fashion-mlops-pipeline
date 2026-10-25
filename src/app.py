import json
import os
import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image

#  Dynamic Directory Resolution (Platform Agnostic)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

MODEL_PATH = os.path.join(PROJECT_DIR, "models", "fashion_cnn.onnx")
CLASSES_PATH = os.path.join(PROJECT_DIR, "data", "class_names.json")

#  Load Class Labels & ONNX Inference Session
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


def preprocess_rgb(image_array):
    """Aspect-ratio preserving ImageNet RGB preprocessing (224x224) for MobileNetV3 ONNX model."""
    if image_array is None:
        return None

    # Convert array to PIL Image
    img = Image.fromarray(image_array).convert("RGB")
    
    #  Pad to square to prevent stretching tall images (e.g., pants, dresses)
    img = pad_to_square(img)
    
    #  Resize to model input dimensions
    img = img.resize((224, 224))

    # Standard ImageNet Normalization
    img_np = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_np = (img_np - mean) / std

    # Transpose array from (H, W, C) to (C, H, W) and add batch dimension
    img_np = np.transpose(img_np, (2, 0, 1))
    tensor = np.expand_dims(img_np, axis=0).astype(np.float32)

    return tensor


def predict_fashion(image):
    if image is None:
        return None

    tensor = preprocess_rgb(image)
    outputs = session.run(None, {input_name: tensor})[0]

    # Compute Softmax probabilities
    exp_outputs = np.exp(outputs - np.max(outputs))
    probs = (exp_outputs / exp_outputs.sum(axis=1, keepdims=True))[0]

    # Map all classes with their respective confidence scores
    confidences = {
        CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
    }
    return confidences


# Dashboard Layout & Custom Styling
custom_css = """
#header { text-align: center; margin-bottom: 20px; }
#header h1 { font-size: 2.2rem; font-weight: 700; color: #1E293B; }
#header p { font-size: 1rem; color: #64748B; }
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo"), css=custom_css) as demo:
    gr.HTML(
        """
        <div id="header">
            <h1> E-Commerce Fashion AI Classifier</h1>
            <p>Production MLOps Pipeline powered by MobileNetV3 & ONNX Runtime (123 Categories)</p>
        </div>
        """
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Upload Product Image (RGB)",
                type="numpy",
                sources=["upload", "clipboard"],
                height=320,
            )
            submit_btn = gr.Button(
                " Run AI Inference", variant="primary", size="lg"
            )

        with gr.Column(scale=1):
            output_labels = gr.Label(
                num_top_classes=5, label=" Top Predictions"
            )

    submit_btn.click(
        fn=predict_fashion, inputs=input_image, outputs=output_labels
    )

if __name__ == "__main__":
    demo.launch(share=True)
