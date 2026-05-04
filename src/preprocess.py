from torchvision import datasets, transforms
from torch.utils.data import DataLoader

MEAN = 0.5077
STD  = 0.2120

def get_dataloaders(batch_size=64):

    train_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,))
    ])

    test_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,))
    ])

    train_data = datasets.ImageFolder(
        root='train',
        transform=train_transforms
    )
    test_data = datasets.ImageFolder(
        root='test',
        transform=test_transforms
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader, train_data.classes