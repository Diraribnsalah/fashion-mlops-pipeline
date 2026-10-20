import io
import numpy as np
import onnxruntime as ort
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException

app = FastAPI(
    title="FashionMNIST ONNX Inference Engine",
    version="1.0.0",
    description="Production MLOps microservice serving ONNX model."
)

CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

try:
    session = ort.InferenceSession("fashion_cnn.onnx")
    input_name = session.get_inputs()[0].name
except Exception as e:
    raise RuntimeError(f" the dowload operation of ONNX model was failed: {str(e)}")

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = image.resize((25, 25))
    img_np = np.array(image, dtype=np.float32) / 255.0
    img_np = (img_np - 0.5) / 0.5
    img_np = np.expand_dims(img_np, axis=(0, 1))
    return img_np.astype(np.float32)

@app.get("/")
def health_check():
    return {"status": "healthy", "engine": "ONNX Runtime", "model": "fashion_cnn.onnx"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="the image type must be PNG and JPEG")

    try:
        image_bytes = await file.read()
        tensor = preprocess_image(image_bytes)

        outputs = session.run(None, {input_name: tensor})[0]

        exp_outputs = np.exp(outputs - np.max(outputs))
        probs = exp_outputs / exp_outputs.sum(axis=1, keepdims=True)

        pred_class_idx = int(np.argmax(probs, axis=1)[0])
        confidence_score = float(probs[0][pred_class_idx])

        return {
            "prediction": CLASS_NAMES[pred_class_idx],
            "confidence": round(confidence_score, 4),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
