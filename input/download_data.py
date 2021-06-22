from torchvision  import datasets
from torchvision.transforms import ToTensor


def download_mnist_datasets():
    train_data = datasets.MNIST(
        root="input",
        download=True,
        train=True,
        transform=ToTensor()
    )
    validation_data = datasets.MNIST(
        root="input",
        download=True,
        train=False,
        transform=ToTensor()
    )
    return train_data, validation_data