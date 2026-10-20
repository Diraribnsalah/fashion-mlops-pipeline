import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_transforms():
  """  transforme the dataset to (25 , 25) and compresse the data value btween (-1 , 1)  """
  return transforms.Compose([
      transforms.Resize((25, 25)),
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,)),
  ])


def get_dataloaders(batch_size=32, val_ratio=0.2, data_dir='./data'):
  """ load the data and split it to (test , train , val ) and return DataLoaders"""
  data_transform = get_transforms()

  raw_train_dataset = datasets.FashionMNIST(
      root=data_dir, train=True, transform=data_transform, download=True
  )

  test_dataset = datasets.FashionMNIST(
      root=data_dir, train=False, transform=data_transform, download=True
  )

  total_size = len(raw_train_dataset)
  val_size = int(total_size * val_ratio)
  train_size = total_size - val_size

  train_data, val_data = random_split(
      raw_train_dataset, [train_size, val_size]
  )

  train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  return train_loader, val_loader, test_loader


 
if __name__ == '__main__':
  train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)

  image_batch, labels_batch = next(iter(train_loader))
  print(f'(Image Batch Shape): {image_batch.shape}')
  print(f'(Labels Batch Shape): {labels_batch.shape}')
