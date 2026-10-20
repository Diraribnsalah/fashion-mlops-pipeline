import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import torch.onnx
from model import CNN

# 1 the envirement setup 
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_and_export(test_loader, weights_path="fashion_cnn.pth", onnx_path="fashion_cnn.onnx"):
    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]

    model = CNN(num_classes=10)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    #  confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - FashionMNIST')
    plt.savefig("confusion_matrix.png")
    plt.close()

    #  export the model to ONNX
    dummy_input = torch.randn(1, 1, 25, 25).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input_image'],
        output_names=['output_logits'],
        dynamic_axes={'input_image': {0: 'batch_size'}, 'output_logits': {0: 'batch_size'}}
    )
    print(f"the model was exported successfully {onnx_path}")
