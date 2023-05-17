from torch.utils.data import Dataset
import numpy as np
import pickle
import os
from PIL import Image
class CIFFA_Dataset(Dataset):
    def __init__(self, root, train=True):
        self.images = []
        self.labels = []
        data_path = os.path.join(root, "cifar-10-batches-py")
        if train:
            data_files = [os.path.join(data_path, "data_batch_{}".format(i)) for i in range(1, 6)]
        else:
            data_files = [os.path.join(data_path, "test_batch")]
        for data_file in data_files:
            with open(data_file, 'rb') as fo:
                dict = pickle.load(fo, encoding="bytes")
                self.images.extend(dict[b'data'])
                self.labels.extend(dict[b'labels'])
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        image = self.images[idx].reshape((3, 32, 32)).astype(np.float32)
        label = self.labels[idx]
        return image/255, label
class Animals_Dataset(Dataset):
    def __init__(self, root="data", train=True, transform=None):
        self.path_images = []
        self.labels = []
        self.classes = []
        self.transform = transform
        if train:
            data_path = os.path.join(root, "animals", "train")
            self.classes = os.listdir(data_path)
        else:
            data_path = os.path.join(root, "animals", "test")
        for i, category in enumerate(self.classes):
            data_sub_path = os.path.join(data_path, category)
            for image in os.listdir(data_sub_path):
                image_path = os.path.join(data_sub_path, image)
                self.path_images.append(image_path)
                self.labels.append(i)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        image = Image.open(self.path_images[idx]).convert("RGB") # dùng convert vì trong data set có một số ảnh có 4 kênh màu
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label
class NaturalDataset(Dataset):
    def __init__(self, root="data", train=True, transform=None):
        self.path_images = []
        self.labels = []
        self.transform = transform
        self.classes__ = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

        if train:
            data_path = os.path.join(root, "natural_scenes", "train")
        else:
            data_path = os.path.join(root, "natural_scenes", "valid")
        for i, category in enumerate(os.listdir(data_path)):
            sub_path = os.path.join(data_path, category)
            for img in os.listdir(sub_path):
                img_path = os.path.join(sub_path, img)
                self.path_images.append(img_path)
                self.labels.append(i)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        img = Image.open(self.path_images[idx])
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label
if __name__ == "__main__":
    # data = CIFFA_Dataset("data")
    data = NaturalDataset(train=False)
    idx = 1000
    image, label = data.__getitem__(idx)
    image.show()
    print(label)
