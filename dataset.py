import os
from glob import glob
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.nn.functional import normalize


class AudioSnippetDataset(Dataset):
    def __init__(self, training_folder, subset_size=None, transform=normalize):
        super(AudioSnippetDataset, self).__init__()
        self.transform = transform
        self.files = glob(os.path.join(training_folder, "*.npy"))
        if subset_size and subset_size < len(self.files):
            self.files = np.random.choice(self.files, subset_size)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = self.files[idx]
        item = np.load(file_name)
        real = np.real(item)
        imag = np.imag(item)
        item = np.array([real, imag])
        item = Variable(torch.Tensor(item))

        if self.transform:
            item = self.transform(item)

        return item
