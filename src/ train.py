import json
import os
import kagglehub
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


#  Custom Dataset Definition
class FastFashionDataset(Dataset):

    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.loc[idx, "id"]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        label = self.df.loc[idx, "label_idx"]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def main():
    #  Dynamic Directory Resolution (Platform Agnostic)
    BASE_DIR = os.path.abspath(os.getcwd())
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODELS_DIR = os.path.join(BASE_DIR, "models")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    CSV_PATH = os.path.join(DATA_DIR, "fashion_full_dataset.csv")
    CLASSES_PATH = os.path.join(DATA_DIR, "class_names.json")
    MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "fashion_mobilenet.pth")

    #  Download & Verify Kaggle Dataset
    print(" Downloading/verifying dataset via KaggleHub...")
    dataset_path = kagglehub.dataset_download(
        "paramaggarwal/fashion-product-images-small"
    )
    LOCAL_IMAGES_DIR = os.path.join(dataset_path, "images")
    print(f" Images located at: {LOCAL_IMAGES_DIR}")

    #  Load CSV & Filter Dataset
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Dataset CSV not found at '{CSV_PATH}'. Please ensure 'fashion_full_dataset.csv' is inside the 'data/' directory."
        )

    df = pd.read_csv(CSV_PATH)

    df["file_exists"] = df["id"].apply(
        lambda img_id: os.path.exists(
            os.path.join(LOCAL_IMAGES_DIR, f"{img_id}.jpg")
        )
    )
    df = df[df["file_exists"]].copy()

    # Filter rare classes (< 5 samples)
    class_counts = df["articleType"].value_counts()
    valid_classes = class_counts[class_counts >= 5].index
    df = df[df["articleType"].isin(valid_classes)].copy()

    unique_types = sorted(df["articleType"].unique())
    label_to_idx = {label: idx for idx, label in enumerate(unique_types)}
    df["label_idx"] = df["articleType"].map(label_to_idx)

    # Export synchronized class mapping
    with open(CLASSES_PATH, "w", encoding="utf-8") as f:
        json.dump(unique_types, f, ensure_ascii=False, indent=4)

    num_classes = len(unique_types)
    print(
        f" Valid Dataset Samples: {len(df)} | Unique Categories: {num_classes}"
    )

    #  Augmentation & Preprocessing Pipeline
    transform_train = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    transform_val = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    #  Data Partitioning & Loaders
    train_df, val_df = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df["label_idx"]
    )

    # Safe worker allocation for Windows/Linux/macOS
    num_workers = 2 if os.name != "nt" else 0

    train_loader = DataLoader(
        FastFashionDataset(
            train_df, LOCAL_IMAGES_DIR, transform=transform_train
        ),
        batch_size=64,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        FastFashionDataset(val_df, LOCAL_IMAGES_DIR, transform=transform_val),
        batch_size=64,
        shuffle=False,
        num_workers=num_workers,
    )

    #  Hardware Accelerator Selection (CUDA / MPS / CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f" Compute Hardware In Use: {device}")

    #  Model Initialization
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    #  Training Hyperparameters & Optimization Setup
    epochs = 12
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0002, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    #  Training Loop
    print(f" Starting model training for {epochs} epochs...")
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (step + 1) % 100 == 0 or (step + 1) == len(train_loader):
                print(
                    f"Epoch [{epoch+1}/{epochs}] | Batch [{step+1}/{len(train_loader)}] | Loss: {loss.item():.4f}"
                )

        train_acc = correct / total

        # Validation Phase
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = val_correct / val_total
        current_lr = scheduler.get_last_lr()[0]

        print(
            f" Epoch [{epoch+1}/{epochs}] - Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | Current LR: {current_lr:.6f}\n"
        )

        scheduler.step()

        # Save Best Performing Weights
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  Saved best model checkpoint to: {MODEL_SAVE_PATH}\n")

    print(
        f" Training Complete! Highest Validation Accuracy: {best_acc*100:.2f}%"
    )


if __name__ == "__main__":
    main()
