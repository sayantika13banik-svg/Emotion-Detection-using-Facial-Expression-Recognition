from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# calculated from FER2013 training set
MEAN = 0.5077
STD  = 0.2120

def get_dataloaders(batch_size=64):
    # function that returns train and test loaders
    # batch_size=64 — default value, can be changed when calling

    train_transforms = transforms.Compose([
        transforms.Grayscale(),
        # convert to 1 channel grayscale

        transforms.Resize((48, 48)),
        # resize to 48×48

        transforms.RandomHorizontalFlip(),
        # randomly flip left-right during training

        transforms.RandomRotation(10),
        # randomly rotate up to 10 degrees

        transforms.ToTensor(),
        # convert to tensor, pixels 0-255 → 0.0-1.0

        transforms.Normalize((MEAN,), (STD,))
        # normalize using FER2013 statistics
    ])

    test_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,))
        # no augmentation for test set
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
        # shuffle=True — randomise order every epoch
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
        # shuffle=False — order doesn't matter for evaluation
    )

    return train_loader, test_loader, train_data.classes
    # returns both loaders AND class names
    # class names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']