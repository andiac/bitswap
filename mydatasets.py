import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class MyCeleba(Dataset):
    def __init__(self, data_dir="./model/data/celeba", transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # npy file is NCHW
        self.data = np.transpose(np.load(os.path.join(data_dir, "celeba10000.npy")), (0, 2, 3, 1))
        self.data = (self.data * 255.).astype("uint8")

    def __getitem__(self, index):
        x = Image.fromarray(self.data[index, :, :, :]).convert("RGB")

        if self.transform is not None:
            x = self.transform(x)

        return x, x

    def __len__(self):
        return self.data.shape[3]

class Imgnet32Val(Dataset):
    def __init__(self, data_dir="./model/data/Imagenet32_val_npz", transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.data = np.load(os.path.join(data_dir, "val_data.npz"))["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    def __getitem__(self, index):
        x = Image.fromarray(self.data[index, :, :, :]).convert("RGB")

        if self.transform is not None:
            x = self.transform(x)

        return x, x

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    a = Imgnet32Val()
