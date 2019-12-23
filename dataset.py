import os
from glob import glob
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset

class AudioSnippetDataset(Dataset):
    def __init__(self, training_folder, transform=None):
        super(AudioSnippetDataset, self).__init__()
        self.training_folder = training_folder
        self.transform = transform
        self.len = len(glob(os.path.join(training_folder, "*.npy")))
        print(self.len)
        

    def __len__(self):
        return self.len


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = "arr_" + str(idx) + ".npy"
        item = np.load(os.path.join(self.training_folder, file_name))
        real = np.real(item)
        imag = np.imag(item)
        item = np.array([real, imag])
        item = Variable(torch.Tensor(item))

        if self.transform:
           item = self.transform(item)

        return item
