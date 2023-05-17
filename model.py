import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.cv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU()
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.cv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )
        self.cv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.linear1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(6272, 2048),
            nn.ReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.ReLU()
        )
        self.linear3 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        self.linear4 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cv1(x)
        x = self.cv2(x)
        x = self.cv3(x)
        x = self.cv4(x)
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        return x
class CNNForNaturalDataset(nn.Module):
    def _makeBlog(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
    def __init__(self, num_classes=6):
        super().__init__()
        self.conv1 = self._makeBlog(3, 32)
        self.conv2 = self._makeBlog(32, 64)
        self.conv3 = self._makeBlog(64, 64)
        self.conv4 = self._makeBlog(64, 128)
        self.linear_1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(6272, 2048),
            nn.ReLU()
        )
        self.linear_2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.ReLU()
        )
        self.linear_3 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        self.linear_4 = nn.Linear(512, num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.shape[0], -1)
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = self.linear_3(x)
        x = self.linear_4(x)
        return x
if __name__ == "__main__":
    image = torch.rand((8, 3, 128, 128))
    model = CNNForNaturalDataset()
    result = model(image)
    print(result.shape)
