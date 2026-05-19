import mindspore.nn as nn


class LeNet5(nn.Cell):
    """LeNet-5 for 32x32 grayscale image classification."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, pad_mode="valid")
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, pad_mode="valid")
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Dense(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Dense(84, num_classes)

    def construct(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        return self.fc3(x)


def freeze_feature_extractor(network: LeNet5) -> None:
    for layer in [network.conv1, network.conv2, network.fc1, network.fc2]:
        for param in layer.get_parameters():
            param.requires_grad = False


class ImprovedGarbageCNN(nn.Cell):
    """A stronger RGB CNN for garbage classification."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.SequentialCell(
            nn.Conv2d(3, 32, kernel_size=3, pad_mode="pad", padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, pad_mode="pad", padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, pad_mode="pad", padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, pad_mode="pad", padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.SequentialCell(
            nn.Dense(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Dense(256, num_classes),
        )

    def construct(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return self.classifier(x)
