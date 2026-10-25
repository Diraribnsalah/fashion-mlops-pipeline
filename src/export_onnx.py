import json
import os
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small


def export_to_onnx():
    #  Dynamic Directory Resolution (Platform Agnostic)
    BASE_DIR = os.path.abspath(os.getcwd())
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODELS_DIR = os.path.join(BASE_DIR, "models")

    CLASSES_PATH = os.path.join(DATA_DIR, "class_names.json")
    MODEL_PATH = os.path.join(MODELS_DIR, "fashion_mobilenet.pth")
    ONNX_PATH = os.path.join(MODELS_DIR, "fashion_cnn.onnx")

    # Verify input artifacts exist before processing
    if not os.path.exists(CLASSES_PATH):
        raise FileNotFoundError(
            f"Class mapping file not found at '{CLASSES_PATH}'. Ensure 'class_names.json' exists in data directory."
        )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at '{MODEL_PATH}'. Run training pipeline first to generate PyTorch weights."
        )

    #  Read unique class count dynamically
    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    num_classes = len(class_names)
    print(
        f" Exporting MobileNetV3 model ({num_classes} classes) to ONNX format..."
    )

    #  Reconstruct architecture & load trained weights
    model = mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    #  Dummy Input Tensor (Batch=1, Channels=3, H=224, W=224 RGB)
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)

    #  Export Graph to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print(f" ONNX model successfully exported to: {ONNX_PATH}")


if __name__ == "__main__":
    export_to_onnx()
